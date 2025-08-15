import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import xarray as xr

# Local
from .. import metrics, utils, vis
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from .ar_model import ARModel
from .constant_latent_encoder import ConstantLatentEncoder
from .graph_latent_decoder import GraphLatentDecoder
from .graph_latent_encoder import GraphLatentEncoder
from .hi_graph_latent_decoder import HiGraphLatentDecoder
from .hi_graph_latent_encoder import HiGraphLatentEncoder


class GraphEFM(ARModel):
    """
    Graph-based Ensemble Forecasting Model
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
    ):
        super().__init__(args, config, datastore)

        assert (
            args.n_example_pred <= args.batch_size
        ), "Can not plot more examples than batch size in GraphEFM"
        self.sample_obs_noise = bool(args.sample_obs_noise)
        self.ensemble_size = args.ensemble_size
        self.num_latents_plot = args.num_latents_plot
        self.var_leads_val_plot = args.var_leads_val_plot
        self.kl_beta = args.kl_beta
        self.crps_weight = args.crps_weight

        # Load graph with static features
        graph_dir_path = datastore.root_path / "graph" / args.graph
        self.hierarchical_graph, graph_ldict = utils.load_graph(
            graph_dir_path=graph_dir_path
        )

        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        # grid_dim from data + static
        grid_state_dim = self._datastore.get_num_data_vars("state")
        grid_current_dim = self.grid_input_dim + grid_state_dim
        g2m_dim = self.g2m_features.shape[1]
        m2g_dim = self.m2g_features.shape[1]

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)

        self.grid_prev_embedder = utils.make_mlp(
            [self.grid_input_dim] + self.mlp_blueprint_end
        )  # For states up to t-1
        self.grid_current_embedder = utils.make_mlp(
            [grid_current_dim] + self.mlp_blueprint_end
        )  # For states including t
        # Embedders for mesh
        self.g2m_embedder = utils.make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] + self.mlp_blueprint_end)
        if self.hierarchical_graph:
            # Print some useful info
            print("Loaded hierarchical graph with structure:")
            level_mesh_sizes = [
                mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
            ]
            self.num_mesh_nodes = level_mesh_sizes[-1]
            num_levels = len(self.mesh_static_features)
            for level_index, level_mesh_size in enumerate(level_mesh_sizes):
                same_level_edges = self.m2m_features[level_index].shape[0]
                print(
                    f"level {level_index} - {level_mesh_size} nodes, "
                    f"{same_level_edges} same-level edges"
                )

                if level_index < (num_levels - 1):
                    up_edges = self.mesh_up_features[level_index].shape[0]
                    down_edges = self.mesh_down_features[level_index].shape[0]
                    print(f"  {level_index}<->{level_index+1}")
                    print(f" - {up_edges} up edges, {down_edges} down edges")
            # Embedders
            # Assume all levels have same static feature dimensionality
            mesh_dim = self.mesh_static_features[0].shape[1]
            m2m_dim = self.m2m_features[0].shape[1]
            mesh_up_dim = self.mesh_up_features[0].shape[1]
            mesh_down_dim = self.mesh_down_features[0].shape[1]

            # Separate mesh node embedders for each level
            self.mesh_embedders = torch.nn.ModuleList(
                [
                    utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels)
                ]
            )
            self.mesh_up_embedders = torch.nn.ModuleList(
                [
                    utils.make_mlp([mesh_up_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels - 1)
                ]
            )
            self.mesh_down_embedders = torch.nn.ModuleList(
                [
                    utils.make_mlp([mesh_down_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels - 1)
                ]
            )
            # If not using any processor layers, no need to embed m2m
            self.embedd_m2m = (
                max(
                    args.prior_processor_layers,
                    args.encoder_processor_layers,
                    args.processor_layers,
                )
                > 0
            )
            if self.embedd_m2m:
                self.m2m_embedders = torch.nn.ModuleList(
                    [
                        utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)
                        for _ in range(num_levels)
                    ]
                )
        else:
            (
                self.num_mesh_nodes,
                mesh_static_dim,
            ) = self.mesh_static_features.shape
            print(
                f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes}"
                f"nodes ({self.num_grid_nodes} grid, "
                f"{self.num_mesh_nodes} mesh)"
            )
            mesh_static_dim = self.mesh_static_features.shape[1]
            self.mesh_embedder = utils.make_mlp(
                [mesh_static_dim] + self.mlp_blueprint_end
            )
            m2m_dim = self.m2m_features.shape[1]
            self.m2m_embedder = utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)

        latent_dim = args.latent_dim if args.latent_dim is not None else args.hidden_dim
        # Prior
        if args.learn_prior:
            if self.hierarchical_graph:
                self.prior_model = HiGraphLatentEncoder(
                    latent_dim,
                    self.g2m_edge_index,
                    self.m2m_edge_index,
                    self.mesh_up_edge_index,
                    args.hidden_dim,
                    args.prior_processor_layers,
                    hidden_layers=args.hidden_layers,
                    output_dist=args.prior_dist,
                )
            else:
                self.prior_model = GraphLatentEncoder(
                    latent_dim,
                    self.g2m_edge_index,
                    self.m2m_edge_index,
                    args.hidden_dim,
                    args.prior_processor_layers,
                    hidden_layers=args.hidden_layers,
                    output_dist=args.prior_dist,
                )
        else:
            self.prior_model = ConstantLatentEncoder(
                latent_dim,
                self.num_mesh_nodes,
                output_dist=args.prior_dist,
            )

        # Enc. + Dec.
        if self.hierarchical_graph:
            # Encoder
            self.encoder = HiGraphLatentEncoder(
                latent_dim,
                self.g2m_edge_index,
                self.m2m_edge_index,
                self.mesh_up_edge_index,
                args.hidden_dim,
                args.encoder_processor_layers,
                hidden_layers=args.hidden_layers,
                output_dist="diagonal",
            )
            # Decoder
            self.decoder = HiGraphLatentDecoder(
                self.g2m_edge_index,
                self.m2m_edge_index,
                self.m2g_edge_index,
                self.mesh_up_edge_index,
                self.mesh_down_edge_index,
                args.hidden_dim,
                latent_dim,
                grid_state_dim,
                args.processor_layers,
                hidden_layers=args.hidden_layers,
                output_std=bool(args.output_std),
            )
        else:
            # Encoder
            self.encoder = GraphLatentEncoder(
                latent_dim,
                self.g2m_edge_index,
                self.m2m_edge_index,
                args.hidden_dim,
                args.encoder_processor_layers,
                hidden_layers=args.hidden_layers,
                output_dist="diagonal",
            )
            # Decoder
            self.decoder = GraphLatentDecoder(
                self.g2m_edge_index,
                self.m2m_edge_index,
                self.m2g_edge_index,
                args.hidden_dim,
                latent_dim,
                grid_state_dim,
                args.processor_layers,
                hidden_layers=args.hidden_layers,
                output_std=bool(args.output_std),
            )

        # Add lists for val and test errors of ensemble prediction
        self.val_metrics.update(
            {
                "spread_squared": [],
                "ens_mse": [],
            }
        )
        self.test_metrics.update(
            {
                "ens_mae": [],
                "ens_mse": [],
                "crps_ens": [],
                "spread_squared": [],
            }
        )

    def sample_next_state(self, pred_mean, pred_std):
        """
        Sample state at next time step given Gaussian distribution.
        If self.sample_obs_noise is False, only return mean.

        pred_mean: (B, num_grid_nodes, d_state)
        pred_std: (B, num_grid_nodes, d_state) or
            None (if not output_std)

        Return:
        next_state: (B, num_grid_nodes, d_state)
        """
        if not self.output_std:
            pred_std = self.per_var_std  # (d_f,)

        if self.sample_obs_noise:
            return torch.distributions.Normal(pred_mean, pred_std).rsample()
            # (B, num_grid_nodes, d_state)

        return pred_mean  # (B, num_grid_nodes, d_state)

    def embedd_current(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        current_state,
    ):
        """
        embed grid representation including current (target) state. Used as
        input to the encoder, which is conditioned also on the target.

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        current_state: (B, num_grid_nodes, feature_dim), X_{t+1}

        Returns:
        current_emb: (B, num_grid_nodes, d_h)
        """
        batch_size = prev_state.shape[0]

        grid_current_features = torch.cat(
            (
                prev_prev_state,
                prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
                current_state,
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_current_dim)

        return self.grid_current_embedder(
            grid_current_features
        )  # (B, num_grid_nodes, d_h)

    def embedd_all(self, prev_state, prev_prev_state, forcing):
        """
        embed all node and edge representations

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)

        Returns:
        grid_emb: (B, num_grid_nodes, d_h)
        graph_embedding: dict with entries of shape (B, *, d_h)
        """
        batch_size = prev_state.shape[0]

        grid_features = torch.cat(
            (
                prev_prev_state,
                prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_dim)

        grid_emb = self.grid_prev_embedder(grid_features)
        # (B, num_grid_nodes, d_h)

        # Graph embedding
        graph_emb = {
            "g2m": self.expand_to_batch(
                self.g2m_embedder(self.g2m_features), batch_size
            ),  # (B, M_g2m, d_h)
            "m2g": self.expand_to_batch(
                self.m2g_embedder(self.m2g_features), batch_size
            ),  # (B, M_m2g, d_h)
        }

        if self.hierarchical_graph:
            graph_emb["mesh"] = [
                self.expand_to_batch(emb(node_static_features), batch_size)
                for emb, node_static_features in zip(
                    self.mesh_embedders,
                    self.mesh_static_features,
                )
            ]  # each (B, num_mesh_nodes[l], d_h)

            if self.embedd_m2m:
                graph_emb["m2m"] = [
                    self.expand_to_batch(emb(edge_feat), batch_size)
                    for emb, edge_feat in zip(self.m2m_embedders, self.m2m_features)
                ]
            else:
                # Need a placeholder otherwise, just use raw features
                graph_emb["m2m"] = list(self.m2m_features)

            graph_emb["mesh_up"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(self.mesh_up_embedders, self.mesh_up_features)
            ]
            graph_emb["mesh_down"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_down_embedders, self.mesh_down_features
                )
            ]
        else:
            graph_emb["mesh"] = self.expand_to_batch(
                self.mesh_embedder(self.mesh_static_features), batch_size
            )  # (B, num_mesh_nodes, d_h)
            graph_emb["m2m"] = self.expand_to_batch(
                self.m2m_embedder(self.m2m_features), batch_size
            )  # (B, M_m2m, d_h)

        return grid_emb, graph_emb

    def compute_step_loss(
        self,
        prev_states,
        current_state,
        forcing_features,
    ):
        """
        Perform forward pass and compute loss for one time step

        prev_states: (B, 2, num_grid_nodes, d_features), X^{t-p}, ..., X^{t-1}
        current_state: (B, num_grid_nodes, d_features) X^t
        forcing_features: (B, num_grid_nodes, d_forcing) corresponding to
            index 1 of prev_states
        """
        # embed all features
        grid_prev_emb, graph_emb = self.embedd_all(
            prev_states[:, 1],
            prev_states[:, 0],
            forcing_features,
        )
        # embed also including current grid state, for encoder
        grid_current_emb = self.embedd_current(
            prev_states[:, 1],
            prev_states[:, 0],
            forcing_features,
            current_state,
        )  # (B, num_grid_nodes, d_h)

        # Compute variational approximation (encoder)
        var_dist = self.encoder(
            grid_current_emb, graph_emb=graph_emb
        )  # Gaussian, (B, num_mesh_nodes, d_latent)

        # Compute likelihood
        last_state = prev_states[:, -1]
        likelihood_term, pred_mean, pred_std = self.estimate_likelihood(
            var_dist, current_state, last_state, grid_prev_emb, graph_emb
        )
        if self.kl_beta > 0:
            # Compute prior
            prior_dist = self.prior_model(
                grid_prev_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)

            # Compute KL
            kl_term = torch.sum(
                torch.distributions.kl_divergence(var_dist, prior_dist),
                dim=(1, 2),
            )  # (B,)
        else:
            # If beta=0, do not need to even compute prior nor KL
            kl_term = None  # Set to None to crash if erroneously used

        return likelihood_term, kl_term, pred_mean, pred_std

    def estimate_likelihood(
        self, latent_dist, current_state, last_state, grid_prev_emb, graph_emb
    ):
        """
        Estimate (masked) likelihood using given distribution over
        latent variables

        latent_dist: distribution, (B, num_mesh_nodes, d_latent)
        current_state: (B, num_grid_nodes, d_state)
        last_state: (B, num_grid_nodes, d_state)
        grid_prev_emb: (B, num_grid_nodes, d_state)
        g2m_emb: (B, M_g2m, d_h)
        m2m_emb: (B, M_m2m, d_h)
        m2g_emb: (B, M_m2g, d_h)

        Returns:
        likelihood_term: (B,)
        pred_mean: (B, num_grid_nodes, d_state)
        pred_std: (B, num_grid_nodes, d_state) or (d_state,)
        """
        # Sample from variational distribution
        latent_samples = latent_dist.rsample()  # (B, num_mesh_nodes, d_latent)

        # Compute reconstruction (decoder)
        pred_mean, model_pred_std = self.decoder(
            grid_prev_emb, latent_samples, last_state, graph_emb
        )  # both (B, num_grid_nodes, d_state)

        if self.output_std:
            pred_std = model_pred_std  # (B, num_grid_nodes, d_state)
        else:
            # Use constant set std.-devs.
            pred_std = self.per_var_std  # (d_f,)

        # Compute likelihood (negative loss, exactly likelihood for nll loss)
        # Note: There are some round-off errors here due to float32
        # and large values
        entry_likelihoods = -self.loss(
            pred_mean,
            current_state,
            pred_std,
            mask=self.interior_mask_bool,
            average_grid=False,
            sum_vars=False,
        )  # (B, num_grid_nodes', d_state)
        likelihood_term = torch.sum(entry_likelihoods, dim=(1, 2))  # (B,)
        return likelihood_term, pred_mean, pred_std

    def training_step(self, batch):
        """
        Train on single batch

        batch, containing:
        init_states: (B, 2, num_grid_nodes, d_state)
        target_states: (B, pred_steps, num_grid_nodes, d_state)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing), where
            index 0 corresponds to index 1 of init_states
        """
        init_states, target_states, forcing_features, _ = batch

        prev_prev_state = init_states[:, 0]  # (B, num_grid_nodes, d_state)
        prev_state = init_states[:, 1]  # (B, num_grid_nodes, d_state)
        pred_steps = forcing_features.shape[1]

        loss_like_list = []
        loss_kl_list = []

        for i in range(pred_steps):
            forcing = forcing_features[:, i]  # (B, num_grid_nodes, d_forcing)
            target_state = target_states[:, i]  # (B, num_grid_nodes, d_state)

            prev_states_stacked = torch.stack(
                (prev_prev_state, prev_state), dim=1
            )  # (B, 2, num_grid_nodes, d_state)
            (
                loss_like_term,
                loss_kl_term,
                pred_mean,
                pred_std,
            ) = self.compute_step_loss(
                prev_states_stacked,
                target_state,
                forcing,
            )
            # (B,), (B,), (B, num_grid_nodes, d_state),
            # pred_std is (B, num_grid_nodes, d_state) or (d_state)

            loss_like_list.append(loss_like_term)
            loss_kl_list.append(loss_kl_term)

            # Get predicted next state (sample or mean)
            predicted_state = self.sample_next_state(pred_mean, pred_std)

            # Overwrite border with true state
            new_state = (
                self.boundary_mask * target_state + self.interior_mask * predicted_state
            )

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        # Compute final ELBO and loss, sum over time, mean over batch
        per_sample_likelihood = torch.sum(
            torch.stack(loss_like_list, dim=1), dim=1
        )  # (B,)
        mean_likelihood = torch.mean(per_sample_likelihood)
        log_dict = {
            "elbo_likelihood": mean_likelihood,
        }

        if self.kl_beta > 0:
            # Only compute full KL + ELBO if beta > 0
            per_sample_kl = torch.sum(torch.stack(loss_kl_list, dim=1), dim=1)  # (B,)
            mean_kl = torch.mean(per_sample_kl)
            elbo = mean_likelihood - mean_kl
            loss = -mean_likelihood + self.kl_beta * mean_kl

            log_dict["elbo"] = elbo
            log_dict["elbo_kl"] = mean_kl
        else:
            # Pure auto-encoder training
            loss = -mean_likelihood

        # Optionally sample trajectories and compute CRPS loss
        if self.crps_weight > 0:
            # Sample trajectories using prior
            pred_traj_means, pred_traj_stds = self.sample_trajectories(
                init_states,
                forcing_features,
                target_states,
                2,
            )
            # (B, S=2, pred_steps, num_grid_nodes, d_f), always 2 samples

            # Compute CRPS
            crps_estimate = metrics.crps_ens(
                pred_traj_means,
                target_states,
                pred_traj_stds,
                mask=self.interior_mask_bool,
            )  # (B, pred_steps)
            crps_loss = torch.mean(crps_estimate)

            # Add onto loss
            loss = loss + self.crps_weight * crps_loss
            log_dict["crps_loss"] = crps_loss

        log_dict["train_loss"] = loss
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss

    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Sample one time step prediction

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)

        Returns:
        new_state: (B, num_grid_nodes, feature_dim)
        """
        # embed all features
        grid_prev_emb, graph_emb = self.embedd_all(prev_state, prev_prev_state, forcing)

        # Compute prior
        prior_dist = self.prior_model(
            grid_prev_emb, graph_emb=graph_emb
        )  # (B, num_mesh_nodes, d_latent)

        # Sample from prior
        latent_samples = prior_dist.rsample()
        # (B, num_mesh_nodes, d_latent)

        # Compute reconstruction (decoder)
        last_state = prev_state
        pred_mean, pred_std = self.decoder(
            grid_prev_emb, latent_samples, last_state, graph_emb
        )  # (B, num_grid_nodes, d_state)

        return self.sample_next_state(pred_mean, pred_std), pred_std

    def sample_trajectories(
        self,
        init_states,
        forcing_features,
        true_states,
        num_traj,
        use_encoder=False,
    ):
        """
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        num_traj: S, number of trajectories to sample
        use_encoder: bool, if latent variables should be sampled from
            var. distribution

        Returns
        traj_means: (B, S, pred_steps, num_grid_nodes, d_f)
        traj_stds: (B, S, pred_steps, num_grid_nodes, d_f) or (d_f)
        """
        unroll_func = (
            self.unroll_prediction_vi if use_encoder else self.unroll_prediction
        )
        traj_list = [
            unroll_func(
                init_states,
                forcing_features,
                true_states,
            )
            for _ in range(num_traj)
        ]
        # List of tuples, each containing
        # mean: (B, pred_steps, num_grid_nodes, d_f) and
        # std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        traj_means = torch.stack([pred_pair[0] for pred_pair in traj_list], dim=1)
        if self.output_std:
            traj_stds = torch.stack([pred_pair[1] for pred_pair in traj_list], dim=1)
        else:
            traj_stds = self.per_var_std

        return traj_means, traj_stds

    def unroll_prediction_vi(self, init_states, forcing_features, true_states):
        """
        Roll out prediction, sampling latent var. from variational
        encoder distribution

        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            # Compute 1-step prediction, but using encoder
            forcing = forcing_features[:, i]
            current_state = true_states[:, i]

            # embed all features
            grid_prev_emb, graph_emb = self.embedd_all(
                prev_state, prev_prev_state, forcing
            )

            # embed also including current grid state, for encoder
            grid_current_emb = self.embedd_current(
                prev_state,
                prev_prev_state,
                forcing,
                current_state,
            )

            # Compute variational distribution
            var_dist = self.encoder(
                grid_current_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)

            # Sample from var. dist.
            latent_samples = var_dist.rsample()
            # (B, num_mesh_nodes, d_latent)

            # Compute reconstruction (decoder)
            pred_mean, pred_std = self.decoder(
                grid_prev_emb, latent_samples, prev_state, graph_emb
            )  # (B, num_grid_nodes, d_state)

            pred_state = self.sample_next_state(pred_mean, pred_std)
            # pred_state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            # Overwrite border with true state
            new_state = (
                self.boundary_mask * current_state + self.interior_mask * pred_state
            )

            prediction_list.append(new_state)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std

    def plot_examples(self, batch, n_examples, split, prediction=None):
        """
        Plot ensemble forecast + mean and std
        (split argument should be unused, only for compatibility with ARModel)
        """
        init_states, target_states, forcing_features, time = batch

        trajectories, _ = self.sample_trajectories(
            init_states,
            forcing_features,
            target_states,
            self.ensemble_size,
        )
        # (B, S, pred_steps, num_grid_nodes, d_f)

        # Rescale to original data scale
        traj_rescaled = trajectories * self.state_std + self.state_mean
        target_rescaled = target_states * self.state_std + self.state_mean

        # Compute mean and std of ensemble
        ens_mean = torch.mean(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        # Iterate over the examples
        for (
            traj_slice,
            target_slice,
            ens_mean_slice,
            ens_std_slice,
            time_slice,
        ) in zip(
            traj_rescaled[:n_examples],
            target_rescaled[:n_examples],
            ens_mean[:n_examples],
            ens_std[:n_examples],
            time[:n_examples],
        ):

            # Create xarray for plotting
            da_samples = [
                self._create_dataarray_from_tensor(
                    tensor=traj_slice[i, ...],
                    time=time_slice,
                    split=split,
                    category="state",
                ).unstack("grid_index")
                for i in range(traj_slice.shape[0])
            ]

            da_target = self._create_dataarray_from_tensor(
                tensor=target_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            da_ens_mean = self._create_dataarray_from_tensor(
                tensor=ens_mean_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            da_ens_std = self._create_dataarray_from_tensor(
                tensor=ens_std_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")
            # traj_slice is (S, pred_steps, num_grid_nodes, d_f)
            # others are (pred_steps, num_grid_nodes, d_f)
            self.plotted_examples += 1  # Increment already here

            # Note: min and max values can not be in ensemble mean
            var_vmin = (
                torch.minimum(
                    traj_slice.flatten(0, 2).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    traj_slice.flatten(0, 2).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, (samples_t, target_t, ens_mean_t, ens_std_t) in enumerate(
                zip(
                    traj_slice.transpose(0, 1),
                    # (pred_steps, S, num_grid_nodes, d_f)
                    target_slice,
                    ens_mean_slice,
                    ens_std_slice,
                ),
                start=1,
            ):
                time_title_part = f"t={t_i} ({self._datastore.step_length*t_i} s)"
                # Create one figure per variable at this time step
                var_figs = [
                    vis.plot_ensemble_prediction(
                        [
                            da.isel(state_feature=var_i, time=t_i - 1)
                            for da in da_samples
                        ],
                        da_target.isel(state_feature=var_i, time=t_i - 1),
                        da_ens_mean.isel(state_feature=var_i, time=t_i - 1),
                        da_ens_std.isel(state_feature=var_i, time=t_i - 1),
                        self._datastore,
                        title=f"{var_name} ({var_unit}), {time_title_part}",
                        var_name=var_name,
                        var_unit=var_unit,
                        vrange=var_vrange,
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            self._datastore.get_vars_names("state"),
                            self._datastore.get_vars_units("state"),
                            var_vranges,
                        )
                    )
                ]

                example_title = f"example_{self.plotted_examples}"

                save_dir = os.path.join(wandb.run.dir, "example_plots")
                os.makedirs(save_dir, exist_ok=True)
                for var_name, fig in zip(
                    self._datastore.get_vars_names("state"), var_figs
                ):
                    filename = os.path.join(
                        save_dir, f"{var_name}_{example_title}_t{t_i}.pdf"
                    )
                    fig.savefig(filename, format="pdf", bbox_inches="tight")

                wandb.log(
                    {
                        f"{var_name}_{example_title}": wandb.Image(fig)
                        for var_name, fig in zip(
                            self._datastore.get_vars_names("state"), var_figs
                        )
                    }
                )
                plt.close("all")  # Close all figs for this time step, saves memory

    def ensemble_common_step(self, batch):
        """
        Perform ensemble forecast and compute basic metrics.
        Common step done during both evaluation and testing

        batch: tuple of tensors, batch to perform ensemble forecast on

        Returns:
        trajectories: (B, S, pred_steps, num_grid_nodes, d_f)
        traj_stds: (B, S, pred_steps, num_grid_nodes, d_f)
        target_states: (B, pred_steps, num_grid_nodes, d_f)
        spread_squared_batch: (B, pred_steps, d_f)
        ens_mse_batch: (B, pred_steps, d_f)
        """
        # Compute and store metrics for ensemble forecast
        init_states, target_states, forcing_features, _ = batch

        trajectories, traj_stds = self.sample_trajectories(
            init_states,
            forcing_features,
            target_states,
            self.ensemble_size,
        )
        # (B, S, pred_steps, num_grid_nodes, d_f)

        spread_squared_batch = metrics.spread_squared(
            trajectories,
            target_states,
            traj_stds,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )
        # (B, pred_steps, d_f)

        ens_mean = torch.mean(
            trajectories, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_mse_batch = metrics.mse(
            ens_mean,
            target_states,
            None,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)

        return (
            trajectories,
            traj_stds,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        )

    def validation_step(self, batch, *args):
        """
        Run validation on single batch
        """
        super().validation_step(batch, *args)
        batch_idx = args[0]

        # Run ensemble forecast
        (
            prior_trajectories,
            _,
            _,
            spread_squared_batch,
            ens_mse_batch,
        ) = self.ensemble_common_step(batch)
        self.val_metrics["spread_squared"].append(spread_squared_batch)
        self.val_metrics["ens_mse"].append(ens_mse_batch)

        # Plot some example predictions using prior and encoder
        if self.trainer.is_global_zero and batch_idx == 0 and self.n_example_pred > 0:
            # Roll out trajectories using variational distribution (encoder)
            (init_states, target_states, forcing_features, time) = batch
            # Only create ens. forecast for as many examples as needed
            init_states = init_states[: self.n_example_pred]
            target_states = target_states[: self.n_example_pred]
            forcing_features = forcing_features[: self.n_example_pred]
            time = time[: self.n_example_pred]

            # Sample trajectories using variational dist. for latent var.
            enc_trajectories, _ = self.sample_trajectories(
                init_states,
                forcing_features,
                target_states,
                self.ensemble_size,
                use_encoder=True,
            )

            # Only need n_example_pred prior trajectories
            prior_trajectories = prior_trajectories[: self.n_example_pred]

            # Rescale to original data scale
            prior_rescaled = prior_trajectories * self.state_std + self.state_mean
            enc_rescaled = enc_trajectories * self.state_std + self.state_mean
            target_rescaled = target_states * self.state_std + self.state_mean

            # Plot samples
            log_plot_dict = {}
            for example_i, (
                prior_traj,
                enc_traj,
                target_traj,
                time_slice,
            ) in enumerate(
                zip(prior_rescaled, enc_rescaled, target_rescaled, time),
                start=1,
            ):
                # prior_traj and enc traj are
                # (S, pred_steps, num_grid_nodes, d_f)

                var_name_list = self._datastore.get_vars_names("state")
                var_unit_list = self._datastore.get_vars_units("state")

                # Make Xarray.DA
                da_target = self._create_dataarray_from_tensor(
                    tensor=target_traj,
                    time=time_slice,
                    split="val",
                    category="state",
                ).unstack("grid_index")
                da_prior_traj = [
                    self._create_dataarray_from_tensor(
                        tensor=prior_traj[i],
                        time=time_slice,
                        split="val",
                        category="state",
                    ).unstack("grid_index")
                    for i in range(prior_traj.shape[0])
                ]
                da_enc_traj = [
                    self._create_dataarray_from_tensor(
                        tensor=enc_traj[i],
                        time=time_slice,
                        split="val",
                        category="state",
                    ).unstack("grid_index")
                    for i in range(enc_traj.shape[0])
                ]

                for var_i, timesteps in self.var_leads_val_plot.items():
                    var_name = var_name_list[var_i]
                    var_unit = var_unit_list[var_i]
                    for step in timesteps:
                        prior_states = [
                            da_prior_sample.isel(state_feature=var_i, time=step - 1)
                            for da_prior_sample in da_prior_traj
                        ]  # (S, num_grid_nodes)
                        enc_states = [
                            da_enc_sample.isel(state_feature=var_i, time=step - 1)
                            for da_enc_sample in da_enc_traj
                        ]  # (S, num_grid_nodes)

                        target_state = da_target.isel(
                            state_feature=var_i, time=step - 1
                        )  # (num_grid_nodes,)

                        # Concatenate along ens member dim for stats compute
                        prior_states_cat = xr.concat(prior_states, dim="ensemble")
                        enc_states_cat = xr.concat(enc_states, dim="ensemble")

                        plot_title = (
                            f"{var_name} ({var_unit}), t={step} "
                            f"({self._datastore.step_length*step} s)"
                        )

                        # Make plots
                        with np.testing.suppress_warnings() as sup:
                            # Numpy will complain when we do the .std
                            # for dimensions only containing NaNs, and this is
                            # very noisy. As we will anyhow filter out these
                            # dimensions later we suppress them here.
                            sup.filter(
                                RuntimeWarning,
                                "Degrees of freedom <= 0 for slice.",
                            )

                            log_plot_dict[
                                f"prior_{var_name}_step_{step}_ex{example_i}"
                            ] = vis.plot_ensemble_prediction(
                                prior_states,
                                target_state,
                                prior_states_cat.mean(dim="ensemble"),
                                prior_states_cat.std(dim="ensemble", ddof=1),
                                self._datastore,
                                title=f"{plot_title} (prior)",
                                var_name=var_name,
                                var_unit=var_unit,
                            )
                            log_plot_dict[
                                f"vi_{var_name}_step_{step}_ex{example_i}"
                            ] = vis.plot_ensemble_prediction(
                                enc_states,
                                target_state,
                                enc_states_cat.mean(dim="ensemble"),
                                enc_states_cat.std(dim="ensemble", ddof=1),
                                self._datastore,
                                title=f"{plot_title} (vi)",
                                var_name=var_name,
                                var_unit=var_unit,
                            )

            # Sample latent variable and plot
            # embed all features
            grid_prev_emb, graph_emb = self.embedd_all(
                init_states[:, 1],
                init_states[:, 0],
                forcing_features[:, 0],
            )  # (B, num_grid_nodes, d_h)
            # embed also including current grid state, for encoder
            grid_current_emb = self.embedd_current(
                init_states[:, 1],
                init_states[:, 0],
                forcing_features[:, 0],
                target_states[:, 0],
            )  # (B, num_grid_nodes, d_h)

            # Create latent variable samples
            prior_dist = self.prior_model(
                grid_prev_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)
            prior_samples = prior_dist.rsample((self.num_latents_plot,)).transpose(
                0, 1
            )  # (B, samples, num_mesh_nodes, d_latent)

            vi_dist = self.encoder(
                grid_current_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)
            vi_samples = vi_dist.rsample((self.num_latents_plot,)).transpose(
                0, 1
            )  # (B, samples, num_mesh_nodes, d_latent)

            # Make plot for each example
            for example_i, (prior_ex_samples, vi_ex_samples) in enumerate(
                zip(prior_samples, vi_samples), start=1
            ):
                log_plot_dict[f"latent_samples_ex{example_i}"] = (
                    vis.plot_latent_samples(prior_ex_samples, vi_ex_samples)
                )

            if not self.trainer.sanity_checking:
                # Log all plots to wandb
                wandb.log(log_plot_dict)

            plt.close("all")

    def log_spsk_ratio(self, metric_vals, prefix):
        """
        Compute the mean spread-skill ratio for logging in evaluation

        metric_vals: dict with all metric values
        prefix: string, prefix to use for logging
        """
        # Compute mean spsk_ratio
        spread_squared_tensor = self.all_gather_cat(
            torch.cat(metric_vals["spread_squared"], dim=0)
        )  # (N_eval, pred_steps, d_f)
        ens_mse_tensor = self.all_gather_cat(
            torch.cat(metric_vals["ens_mse"], dim=0)
        )  # (N_eval, pred_steps, d_f)

        # Do not log during sanity check?
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            # Note that spsk_ratio is scale-invariant, so do not have to rescale
            spread = torch.sqrt(torch.mean(spread_squared_tensor, dim=0))
            skill = torch.sqrt(torch.mean(ens_mse_tensor, dim=0))
            # Both (pred_steps, d_f)

            # Include finite sample correction
            spsk_ratios = np.sqrt((self.ensemble_size + 1) / self.ensemble_size) * (
                spread / skill
            )  # (pred_steps, d_f)
            log_dict = self.create_metric_log_dict(spsk_ratios, prefix, "spsk_ratio")

            log_dict[f"{prefix}_mean_spsk_ratio"] = torch.mean(spsk_ratios)  # log mean
            wandb.log(log_dict)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Must log before super call, as metric lists are cleared at end of step
        self.log_spsk_ratio(self.val_metrics, "val")
        super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        Include metrics computation for ensemble mean prediction
        """
        super().test_step(batch, batch_idx)

        (
            trajectories,
            traj_stds,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        ) = self.ensemble_common_step(batch)
        self.test_metrics["spread_squared"].append(spread_squared_batch)
        self.test_metrics["ens_mse"].append(ens_mse_batch)

        # Compute additional ensemble metrics
        ens_mean = torch.mean(
            trajectories, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(trajectories, dim=1)
        # (B, pred_steps, num_grid_nodes, d_f)

        # Compute MAE for ensemble mean + ensemble CRPS
        ens_maes = metrics.mae(
            ens_mean,
            target_states,
            ens_std,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["ens_mae"].append(ens_maes)
        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            traj_stds,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["crps_ens"].append(crps_batch)

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        super().on_test_epoch_end()
        self.log_spsk_ratio(self.test_metrics, "test")
