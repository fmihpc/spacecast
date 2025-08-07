# Third-party
import numpy as np
import torch
import torch_geometric as pyg
from torch import nn
from torch.nn.functional import silu

# Local
from .. import utils
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from .ar_model import ARModel


class GraphDiff(ARModel):
    """
    Hierarchical Graph-based Diffusion Forecasting Model
    with message passing that goes sequentially down
    and up the hierarchy during processing.
    """

    def __init__(self, args, config: NeuralLAMConfig, datastore: BaseDatastore):
        super().__init__(args, config=config, datastore=datastore)
        self.map_noise = NoiseEmbedding()

        num_state_vars = datastore.get_num_data_vars(category="state")
        num_forcing_vars = datastore.get_num_data_vars(category="forcing")
        num_past_forcing_steps = args.num_past_forcing_steps
        num_future_forcing_steps = args.num_future_forcing_steps

        # grid_dim from data + static
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape

        self.grid_output_dim = num_state_vars  # We only output the denoised state

        self.grid_dim = (
            3 * self.grid_output_dim  # prev_prev, prev, diffusion
            + grid_static_dim
            + num_forcing_vars * (num_past_forcing_steps + num_future_forcing_steps + 1)
        )

        # ----------------------------------------------------------------------------
        # BaseGraphModel parameters
        assert (
            args.eval is None or args.n_example_pred <= args.batch_size
        ), "Can not plot more examples than batch size during validation"

        # Load graph with static features
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first
        # num_mesh_nodes indices,
        graph_dir_path = datastore.root_path / "graph" / args.graph
        self.hierarchical, graph_ldict = utils.load_graph(graph_dir_path=graph_dir_path)
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        self.num_mesh_nodes, _ = self.get_num_mesh()
        utils.rank_zero_print(
            f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # grid_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.grid_embedder = make_mlp([self.grid_dim] + self.mlp_blueprint_end)
        self.g2m_embedder = make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = make_mlp([m2g_dim] + self.mlp_blueprint_end)

        # GNNs
        gnn_class = PropagationNet if args.vertical_propnets else InteractionNet
        # encoder
        self.g2m_gnn = gnn_class(
            self.g2m_edge_index,
            args.hidden_dim,
            hidden_layers=args.hidden_layers,
            update_edges=False,
        )
        self.encoding_grid_mlp = make_mlp([args.hidden_dim] + self.mlp_blueprint_end)

        # decoder
        self.m2g_gnn = gnn_class(
            self.m2g_edge_index,
            args.hidden_dim,
            hidden_layers=args.hidden_layers,
            update_edges=False,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = make_mlp(
            [args.hidden_dim] * (args.hidden_layers + 1) + [self.grid_output_dim],
            layer_norm=False,
        )  # No layer norm on this one

        # ---------------------------------------------------------------------
        # BaseHIGraphModel parameters# Track number of nodes, edges on each lvl
        # Flatten lists for efficient embedding
        self.num_levels = len(self.mesh_static_features)

        # Number of mesh nodes at each level
        self.level_mesh_sizes = [
            mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
        ]  # Needs as python list for later

        # Print some useful info
        print("Loaded hierarchical graph with structure:")
        for level_index, level_mesh_size in enumerate(self.level_mesh_sizes):
            same_level_edges = self.m2m_features[level_index].shape[0]
            print(
                f"level {level_index} - {level_mesh_size} nodes, "
                f"{same_level_edges} same-level edges"
            )

            if level_index < (self.num_levels - 1):
                up_edges = self.mesh_up_features[level_index].shape[0]
                down_edges = self.mesh_down_features[level_index].shape[0]
                print(f"  {level_index}<->{level_index+1}")
                print(f" - {up_edges} up edges, {down_edges} down edges")
        # Embedders
        # Assume all levels have same static feature dimensionality
        mesh_dim = self.mesh_static_features[0].shape[1]
        mesh_same_dim = self.m2m_features[0].shape[1]
        mesh_up_dim = self.mesh_up_features[0].shape[1]
        mesh_down_dim = self.mesh_down_features[0].shape[1]

        # Separate mesh node embedders for each level
        self.mesh_embedders = nn.ModuleList(
            [
                make_mlp([mesh_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels)
            ]
        )
        self.mesh_same_embedders = nn.ModuleList(
            [
                make_mlp([mesh_same_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels)
            ]
        )
        self.mesh_up_embedders = nn.ModuleList(
            [
                make_mlp([mesh_up_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels - 1)
            ]
        )
        self.mesh_down_embedders = nn.ModuleList(
            [
                make_mlp([mesh_down_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels - 1)
            ]
        )

        # Instantiate GNNs
        # Init GNNs
        self.mesh_init_gnns = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )

        # Read out GNNs
        self.mesh_read_gnns = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                    update_edges=False,
                )
                for edge_index in self.mesh_down_edge_index
            ]
        )

        # ----------------------------------------------------------------------------
        # GraphFM specific parameters
        # Make down GNNs, both for down edges and same level
        self.mesh_down_gnns = nn.ModuleList(
            [self.make_down_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels-1)
        self.mesh_down_same_gnns = nn.ModuleList(
            [self.make_same_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels)

        # Make up GNNs, both for up edges and same level
        self.mesh_up_gnns = nn.ModuleList(
            [self.make_up_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels-1)
        self.mesh_up_same_gnns = nn.ModuleList(
            [self.make_same_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels)

    # ----------------------------------------------------------------------------
    def forward(self, x, noise_level, cond):
        """
        Forward pass through the model
        x: (B, num_grid_nodes, feature_dim)
        noise_level: (B, num_grid_nodes, feature_dim)
        cond: (B, num_grid_nodes, feature_dim)
        """
        # Mapping.
        emb = self.map_noise(noise_level).unsqueeze(1).expand(x.shape[0], 1, -1)
        prediction = self.predict_step(x, emb, cond)

        return prediction

    # BaseGraphModel methods
    def predict_step(self, x, emb, cond):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        x: (B, num_grid_nodes, feature_dim)
        emb: (B, num_grid_nodes, feature_dim)
        cond: (B, num_grid_nodes, feature_dim)
        """
        batch_size = x.shape[0]

        # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
        grid_features = torch.cat(
            (
                x,
                cond,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )

        # Embed all features
        grid_emb = self.grid_embedder(grid_features, emb)  # (B, num_grid_nodes, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features, emb)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features, emb)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes(emb)

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded, emb
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.encoding_grid_mlp(
            grid_emb, emb
        )  # (B, num_grid_nodes, d_h)

        # Run processor step
        mesh_rep = self.process_step(mesh_rep, emb)

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded, emb
        )  # (B, num_grid_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(grid_rep, emb)  # (B, num_grid_nodes, d_grid_out)

        return net_output

    # ----------------------------------------------------------------------------
    # BaseHiGraphModel methods
    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        num_mesh_nodes = sum(
            node_feat.shape[0] for node_feat in self.mesh_static_features
        )
        num_mesh_nodes_ignore = num_mesh_nodes - self.mesh_static_features[0].shape[0]
        return num_mesh_nodes, num_mesh_nodes_ignore

    def embedd_mesh_nodes(self, emb):
        """
        Embed static mesh features
        This embeds only bottom level, rest is done at beginning of
        processing step
        Returns tensor of shape (num_mesh_nodes[0], d_h)
        """
        return self.mesh_embedders[0](self.mesh_static_features[0], emb)

    def process_step(self, mesh_rep, emb):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        batch_size = mesh_rep.shape[0]

        # EMBED REMAINING MESH NODES (levels >= 1) -
        # Create list of mesh node representations for each level,
        # each of size (B, num_mesh_nodes[l], d_h)
        mesh_rep_levels = [mesh_rep] + [
            self.expand_to_batch(embedder(node_static_features, emb), batch_size)
            for embedder, node_static_features in zip(
                list(self.mesh_embedders)[1:],
                list(self.mesh_static_features)[1:],
            )
        ]

        # - EMBED EDGES -
        # Embed edges, expand with batch dimension
        mesh_same_rep = [
            self.expand_to_batch(embedder(edge_feat, emb), batch_size)
            for embedder, edge_feat in zip(self.mesh_same_embedders, self.m2m_features)
        ]
        mesh_up_rep = [
            self.expand_to_batch(embedder(edge_feat, emb), batch_size)
            for embedder, edge_feat in zip(
                self.mesh_up_embedders, self.mesh_up_features
            )
        ]
        mesh_down_rep = [
            self.expand_to_batch(embedder(edge_feat, emb), batch_size)
            for embedder, edge_feat in zip(
                self.mesh_down_embedders, self.mesh_down_features
            )
        ]

        # - MESH INIT. -
        # Let level_l go from 1 to L
        for level_l, gnn in enumerate(self.mesh_init_gnns, start=1):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l - 1
            ]  # (B, num_mesh_nodes[l-1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_up_rep[level_l - 1]

            # Apply GNN
            new_node_rep, new_edge_rep = gnn(send_node_rep, rec_node_rep, edge_rep, emb)

            # Update node and edge vectors in lists
            mesh_rep_levels[level_l] = new_node_rep  # (B, num_mesh_nodes[l], d_h)
            mesh_up_rep[level_l - 1] = new_edge_rep  # (B, M_up[l-1], d_h)

        # - PROCESSOR -
        mesh_rep_levels, _, _, mesh_down_rep = self.hi_processor_step(
            mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep, emb
        )

        # - MESH READ OUT. -
        # Let level_l go from L-1 to 0
        for level_l, gnn in zip(
            range(self.num_levels - 2, -1, -1), reversed(self.mesh_read_gnns)
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l + 1
            ]  # (B, num_mesh_nodes[l+1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_down_rep[level_l]

            # Apply GNN
            new_node_rep = gnn(send_node_rep, rec_node_rep, edge_rep, emb)

            # Update node and edge vectors in lists
            mesh_rep_levels[level_l] = new_node_rep  # (B, num_mesh_nodes[l], d_h)

        # Return only bottom level representation
        return mesh_rep_levels[0]  # (B, num_mesh_nodes[0], d_h)

    # ----------------------------------------------------------------------------
    # GraphFM specific methods
    def make_same_gnns(self, args):
        """
        Make intra-level GNNs.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                )
                for edge_index in self.m2m_edge_index
            ]
        )

    def make_up_gnns(self, args):
        """
        Make GNNs for processing steps up through the hierarchy.
        """
        gnn_class = PropagationNet if args.vertical_propnets else InteractionNet
        return nn.ModuleList(
            [
                gnn_class(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )

    def make_down_gnns(self, args):
        """
        Make GNNs for processing steps down through the hierarchy.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                )
                for edge_index in self.mesh_down_edge_index
            ]
        )

    def mesh_down_step(
        self,
        mesh_rep_levels,
        mesh_same_rep,
        mesh_down_rep,
        down_gnns,
        same_gnns,
        emb,
    ):
        """
        Run down-part of vertical processing, sequentially alternating between
        processing using down edges and same-level edges.
        """
        # Run same level processing on level L
        mesh_rep_levels[-1], mesh_same_rep[-1] = same_gnns[-1](
            mesh_rep_levels[-1], mesh_rep_levels[-1], mesh_same_rep[-1], emb
        )

        # Let level_l go from L-1 to 0
        for level_l, down_gnn, same_gnn in zip(
            range(self.num_levels - 2, -1, -1),
            reversed(down_gnns),
            reversed(same_gnns[:-1]),
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[level_l + 1]  # (B, N_mesh[l+1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            down_edge_rep = mesh_down_rep[level_l]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply down GNN
            new_node_rep, mesh_down_rep[level_l] = down_gnn(
                send_node_rep, rec_node_rep, down_edge_rep, emb
            )

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep, emb
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_down_rep

    def mesh_up_step(
        self,
        mesh_rep_levels,
        mesh_same_rep,
        mesh_up_rep,
        up_gnns,
        same_gnns,
        emb,
    ):
        """
        Run up-part of vertical processing, sequentially alternating between
        processing using up edges and same-level edges.
        """

        # Run same level processing on level 0
        mesh_rep_levels[0], mesh_same_rep[0] = same_gnns[0](
            mesh_rep_levels[0], mesh_rep_levels[0], mesh_same_rep[0], emb
        )

        # Let level_l go from 1 to L
        for level_l, (up_gnn, same_gnn) in enumerate(
            zip(up_gnns, same_gnns[1:]), start=1
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[level_l - 1]  # (B, N_mesh[l-1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            up_edge_rep = mesh_up_rep[level_l - 1]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply up GNN
            new_node_rep, mesh_up_rep[level_l - 1] = up_gnn(
                send_node_rep, rec_node_rep, up_edge_rep, emb
            )
            # (B, N_mesh[l], d_h) and (B, M_up[l-1], d_h)

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep, emb
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_up_rep

    def hi_processor_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep, emb
    ):
        """
        Internal processor step of hierarchical graph models.
        Between mesh init and read out.

        Each input is list with representations, each with shape

        mesh_rep_levels: (B, N_mesh[l], d_h)
        mesh_same_rep: (B, M_same[l], d_h)
        mesh_up_rep: (B, M_up[l -> l+1], d_h)
        mesh_down_rep: (B, M_down[l <- l+1], d_h)

        Returns same lists
        """
        for down_gnns, down_same_gnns, up_gnns, up_same_gnns in zip(
            self.mesh_down_gnns,
            self.mesh_down_same_gnns,
            self.mesh_up_gnns,
            self.mesh_up_same_gnns,
        ):
            # Down
            mesh_rep_levels, mesh_same_rep, mesh_down_rep = self.mesh_down_step(
                mesh_rep_levels,
                mesh_same_rep,
                mesh_down_rep,
                down_gnns,
                down_same_gnns,
                emb,
            )

            # Up
            mesh_rep_levels, mesh_same_rep, mesh_up_rep = self.mesh_up_step(
                mesh_rep_levels,
                mesh_same_rep,
                mesh_up_rep,
                up_gnns,
                up_same_gnns,
                emb,
            )

        # Note: We return all, even though only down edges really are used later
        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class NoiseLevelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=16):
        super(NoiseLevelMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, emb):
        x = silu(self.fc1(emb))
        x = silu(self.fc2(x))
        x = self.fc3(x)
        return x  # Output noise-level encoding (batch_size, output_dim)


# TODO: Check if this is correct
class NoiseEmbedding(nn.Module):
    def __init__(self, num_frequencies=32, base_period=16):
        super(NoiseEmbedding, self).__init__()
        self.fourier_transform = FourierEmbedding(
            num_channels=num_frequencies, scale=base_period
        )
        self.mlp = NoiseLevelMLP(input_dim=num_frequencies, output_dim=16)

    def forward(self, log_noise_levels):
        emb = self.fourier_transform(log_noise_levels)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape).contiguous()
        )  # swap sin/cos
        noise_level_encoding = self.mlp(emb)
        return noise_level_encoding


class ConditionalLayerNorm(nn.Module):
    def __init__(self, normalized_shape, noise_level_dim=16):
        super(ConditionalLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.scale_layer = nn.Linear(noise_level_dim, normalized_shape)
        self.offset_layer = nn.Linear(noise_level_dim, normalized_shape)

    def forward(self, x, noise_level_encoding):
        scale = self.scale_layer(noise_level_encoding)  # (batch_size, normalized_shape)
        offset = self.offset_layer(
            noise_level_encoding
        )  # (batch_size, normalized_shape)
        return self.layer_norm(x) * scale + offset


class MLP(nn.Module):
    def __init__(self, blueprint, layer_norm, noise_level_dim=16):
        super(MLP, self).__init__()
        hidden_layers = len(blueprint) - 2
        assert hidden_layers >= 0, "Invalid MLP blueprint"

        layers = []
        for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
            layers.append(nn.Linear(dim1, dim2))
            if layer_i != hidden_layers:
                layers.append(nn.SiLU())  # Swish activation

        self.mlp_layers = nn.Sequential(*layers)

        # Optionally add layer norm to output
        if layer_norm:
            self.layer_norm = ConditionalLayerNorm(
                blueprint[-1], noise_level_dim
            )  # TODO: Check if this is correct
        else:
            self.layer_norm = None

    def forward(self, x, emb):
        x = self.mlp_layers(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x, emb)

        return x


def make_mlp(blueprint, layer_norm=True, noise_level_dim=16):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)
    """

    return MLP(blueprint, layer_norm, noise_level_dim)


class InteractionNet(pyg.nn.MessagePassing):
    """
    Implementation of a generic Interaction Network,
    from Battaglia et al. (2016)
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(
        self,
        edge_index,
        input_dim,
        update_edges=True,
        hidden_layers=1,
        hidden_dim=None,
        edge_chunk_sizes=None,
        aggr_chunk_sizes=None,
        aggr="sum",
    ):
        """
        Create a new InteractionNet

        edge_index: (2,M), Edges in pyg format
        input_dim: Dimensionality of input representations,
            for both nodes and edges
        update_edges: If new edge representations should be computed
            and returned
        hidden_layers: Number of hidden layers in MLPs
        hidden_dim: Dimensionality of hidden layers, if None then same
            as input_dim
        edge_chunk_sizes: List of chunks sizes to split edge representation
            into and use separate MLPs for (None = no chunking, same MLP)
        aggr_chunk_sizes: List of chunks sizes to split aggregated node
            representation into and use separate MLPs for
            (None = no chunking, same MLP)
        aggr: Message aggregation method (sum/mean)
        """
        assert aggr in ("sum", "mean"), f"Unknown aggregation method: {aggr}"
        super().__init__(aggr=aggr)

        if hidden_dim is None:
            # Default to input dim if not explicitly given
            hidden_dim = input_dim

        # Make both sender and receiver indices of edge_index start at 0
        edge_index = edge_index - edge_index.min(dim=1, keepdim=True)[0]
        # Store number of receiver nodes according to edge_index
        self.num_rec = edge_index[1].max() + 1
        edge_index[0] = edge_index[0] + self.num_rec  # Make sender indices after rec
        self.register_buffer("edge_index", edge_index, persistent=False)

        # Create MLPs
        edge_mlp_recipe = [3 * input_dim] + [hidden_dim] * (hidden_layers + 1)
        aggr_mlp_recipe = [2 * input_dim] + [hidden_dim] * (hidden_layers + 1)

        if edge_chunk_sizes is None:
            self.edge_mlp = make_mlp(edge_mlp_recipe)
        else:
            self.edge_mlp = SplitMLPs(
                [make_mlp(edge_mlp_recipe) for _ in edge_chunk_sizes],
                edge_chunk_sizes,
            )

        if aggr_chunk_sizes is None:
            self.aggr_mlp = make_mlp(aggr_mlp_recipe)
        else:
            self.aggr_mlp = SplitMLPs(
                [make_mlp(aggr_mlp_recipe) for _ in aggr_chunk_sizes],
                aggr_chunk_sizes,
            )

        self.update_edges = update_edges

    def forward(self, send_rep, rec_rep, edge_rep, emb=0):
        """
        Apply interaction network to update the representations of receiver
        nodes, and optionally the edge representations.

        send_rep: (N_send, d_h), vector representations of sender nodes
        rec_rep: (N_rec, d_h), vector representations of receiver nodes
        edge_rep: (M, d_h), vector representations of edges used

        Returns:
        rec_rep: (N_rec, d_h), updated vector representations of receiver nodes
        (optionally) edge_rep: (M, d_h), updated vector representations
            of edges
        """
        # Always concatenate to [rec_nodes, send_nodes] for propagation,
        # but only aggregate to rec_nodes
        node_reps = torch.cat((rec_rep, send_rep), dim=-2)
        edge_rep_aggr, edge_diff = self.propagate(
            self.edge_index, x=node_reps, edge_attr=edge_rep, emb=emb
        )
        rec_diff = self.aggr_mlp(torch.cat((rec_rep, edge_rep_aggr), dim=-1), emb)

        # Residual connections
        rec_rep = rec_rep + rec_diff

        if self.update_edges:
            edge_rep = edge_rep + edge_diff
            return rec_rep, edge_rep

        return rec_rep

    def message(self, x_j, x_i, edge_attr, emb):
        """
        Compute messages from node j to node i.
        """
        return self.edge_mlp(torch.cat((edge_attr, x_j, x_i), dim=-1), emb)

    # pylint: disable-next=signature-differs
    def aggregate(self, inputs, index, ptr, dim_size):
        """
        Overridden aggregation function to:
        * return both aggregated and original messages,
        * only aggregate to number of receiver nodes.
        """
        aggr = super().aggregate(inputs, index, ptr, self.num_rec)
        return aggr, inputs


class PropagationNet(InteractionNet):
    """
    Alternative version of InteractionNet that incentivices the propagation of
    information from sender nodes to receivers.
    """

    def __init__(
        self,
        edge_index,
        input_dim,
        update_edges=True,
        hidden_layers=1,
        hidden_dim=None,
        edge_chunk_sizes=None,
        aggr_chunk_sizes=None,
        aggr="sum",
    ):
        # Use mean aggregation in propagation version to avoid instability
        super().__init__(
            edge_index,
            input_dim,
            update_edges=update_edges,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            edge_chunk_sizes=edge_chunk_sizes,
            aggr_chunk_sizes=aggr_chunk_sizes,
            aggr="mean",
        )

    def forward(self, send_rep, rec_rep, edge_rep, emb=0):
        """
        Apply propagation network to update the representations of receiver
        nodes, and optionally the edge representations.

        send_rep: (N_send, d_h), vector representations of sender nodes
        rec_rep: (N_rec, d_h), vector representations of receiver nodes
        edge_rep: (M, d_h), vector representations of edges used

        Returns:
        rec_rep: (N_rec, d_h), updated vector representations of receiver nodes
        (optionally) edge_rep: (M, d_h), updated vector representations
            of edges
        """
        # Always concatenate to [rec_nodes, send_nodes] for propagation,
        # but only aggregate to rec_nodes
        node_reps = torch.cat((rec_rep, send_rep), dim=-2)
        edge_rep_aggr, edge_diff = self.propagate(
            self.edge_index, x=node_reps, edge_attr=edge_rep, emb=emb
        )
        rec_diff = self.aggr_mlp(torch.cat((rec_rep, edge_rep_aggr), dim=-1), emb)

        # Residual connections
        rec_rep = edge_rep_aggr + rec_diff  # residual is to aggregation

        if self.update_edges:
            edge_rep = edge_rep + edge_diff
            return rec_rep, edge_rep

        return rec_rep

    def message(self, x_j, x_i, edge_attr, emb=0):
        """
        Compute messages from node j to node i.
        """
        # Residual connection is to sender node, propagating information to edge
        return x_j + self.edge_mlp(torch.cat((edge_attr, x_j, x_i), dim=-1), emb)


class SplitMLPs(nn.Module):
    """
    Module that feeds chunks of input through different MLPs.
    Split up input along dim -2 using given chunk sizes and feeds
    each chunk through separate MLPs.
    """

    def __init__(self, mlps, chunk_sizes):
        super().__init__()
        assert len(mlps) == len(
            chunk_sizes
        ), "Number of MLPs must match the number of chunks"

        self.mlps = nn.ModuleList(mlps)
        self.chunk_sizes = chunk_sizes

    def forward(self, x):
        """
        Chunk up input and feed through MLPs

        x: (..., N, d), where N = sum(chunk_sizes)

        Returns:
        joined_output: (..., N, d), concatenated results from the MLPs
        """
        chunks = torch.split(x, self.chunk_sizes, dim=-2)
        chunk_outputs = [
            mlp(chunk_input) for mlp, chunk_input in zip(self.mlps, chunks)
        ]
        return torch.cat(chunk_outputs, dim=-2)
