# Standard library
import os
from argparse import ArgumentParser

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import networkx
import numpy as np
import scipy.spatial
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx

# Local
from .config import load_config_and_datastore
from .datastore.base import BaseRegularGridDatastore


def plot_graph(graph, title=None, graph_dir_path=None):
    fig, axis = plt.subplots(figsize=(12, 6), dpi=200)  # W,H
    edge_index = graph.edge_index
    pos = graph.pos

    # Fix for re-indexed edge indices only containing mesh nodes at
    # higher levels in hierarchy
    edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)
    # TODO: indicate direction of directed edges

    # Move all to cpu and numpy, compute (in)-degrees
    degrees = pyg.utils.degree(edge_index[1], num_nodes=pos.shape[0]).cpu().numpy()
    edge_index = edge_index.cpu().numpy()
    pos = pos.cpu().numpy()

    # Plot edges
    from_pos = pos[edge_index[0]]  # (M/2, 2)
    to_pos = pos[edge_index[1]]  # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    axis.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=0.4, colors="black", zorder=1
        )
    )

    # Plot nodes
    node_scatter = axis.scatter(
        pos[:, 0],
        pos[:, 1],
        c=degrees,
        s=3,
        marker="o",
        zorder=2,
        cmap="viridis",
        clim=None,
    )

    plt.colorbar(node_scatter, aspect=50)

    if title is not None:
        axis.set_title(title)

    if graph_dir_path is not None:
        plt.savefig(os.path.join(graph_dir_path, f"{title}.png"))


def sort_nodes_internally(nx_graph):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting.
    # This function fixes this.
    H = networkx.DiGraph()
    H.add_nodes_from(sorted(nx_graph.nodes(data=True)))
    H.add_edges_from(nx_graph.edges(data=True))
    return H


def save_edges(graph, name, base_path):
    torch.save(graph.edge_index, os.path.join(base_path, f"{name}_edge_index.pt"))
    edge_features = torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
        torch.float32
    )  # Save as float32
    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))


def save_edges_list(graphs, name, base_path):
    torch.save(
        [graph.edge_index for graph in graphs],
        os.path.join(base_path, f"{name}_edge_index.pt"),
    )
    edge_features = [
        torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(torch.float32)
        for graph in graphs
    ]  # Save as float32
    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))


def from_networkx_with_start_index(nx_graph, start_index):
    pyg_graph = from_networkx(nx_graph)
    pyg_graph.edge_index += start_index
    return pyg_graph


def mk_2d_graph(xy, nx, ny, earth_mask):
    xm, xM = np.amin(xy[0][0, :]), np.amax(xy[0][0, :])
    ym, yM = np.amin(xy[1][:, 0]), np.amax(xy[1][:, 0])

    # avoid nodes on border
    dx = (xM - xm) / nx
    dy = (yM - ym) / ny
    lx = np.linspace(xm + dx / 2, xM - dx / 2, nx, dtype=np.float32)
    ly = np.linspace(ym + dy / 2, yM - dy / 2, ny, dtype=np.float32)

    mg = np.meshgrid(lx, ly)
    g = networkx.grid_2d_graph(len(ly), len(lx))

    # kdtree for nearest neighbor search of earth nodes
    earth_points = np.argwhere(earth_mask.T).astype(np.float32)
    earth_kdtree = scipy.spatial.KDTree(earth_points)

    # add nodes excluding earth
    for node in list(g.nodes):
        node_pos = np.array([mg[0][node], mg[1][node]], dtype=np.float32)
        dist, _ = earth_kdtree.query(node_pos, k=1)
        if dist < np.sqrt(0.5):
            g.remove_node(node)
        else:
            g.nodes[node]["pos"] = node_pos

    # add diagonal edges if both nodes exist
    for x in range(nx - 1):
        for y in range(ny - 1):
            if g.has_node((x, y)) and g.has_node((x + 1, y + 1)):
                g.add_edge((x, y), (x + 1, y + 1))
            if g.has_node((x + 1, y)) and g.has_node((x, y + 1)):
                g.add_edge((x + 1, y), (x, y + 1))

    # turn into directed graph
    dg = networkx.DiGraph(g)

    # add node data
    for u, v in g.edges():
        d = np.sqrt(np.sum((g.nodes[u]["pos"] - g.nodes[v]["pos"]) ** 2))
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = g.nodes[u]["pos"] - g.nodes[v]["pos"]
        dg.add_edge(v, u)
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = g.nodes[v]["pos"] - g.nodes[u]["pos"]

    # add self edge if needed
    for v, degree in list(dg.degree()):
        if degree <= 1:
            dg.add_edge(v, v, len=0, vdiff=np.array([0, 0]))

    return dg


def prepend_node_index(graph, new_index):
    # Relabel node indices in graph, insert (graph_level, i, j)
    ijk = [tuple((new_index,) + x) for x in graph.nodes]
    to_mapping = dict(zip(graph.nodes, ijk))
    return networkx.relabel_nodes(graph, to_mapping, copy=True)


def create_graph(
    graph_dir_path: str,
    xy: np.ndarray,
    earth_mask: np.ndarray,
    n_max_levels: int,
    hierarchical: bool,
    create_plot: bool,
):
    """
    Create graph components from `xy` grid coordinates and store in
    `graph_dir_path`.

    Creates the following files for all graphs:
    - g2m_edge_index.pt  [2, N_g2m_edges]
    - g2m_features.pt    [N_g2m_edges, d_features]
    - m2g_edge_index.pt  [2, N_m2m_edges]
    - m2g_features.pt    [N_m2m_edges, d_features]
    - m2m_edge_index.pt  list of [2, N_m2m_edges_level], length==n_levels
    - m2m_features.pt    list of [N_m2m_edges_level, d_features],
                         length==n_levels
    - mesh_features.pt   list of [N_mesh_nodes_level, d_mesh_static],
                         length==n_levels

    where
      d_features:
            number of features per edge (currently d_features==3, for
            edge-length, x and y)
      N_g2m_edges:
            number of edges in the graph from grid-to-mesh
      N_m2g_edges:
            number of edges in the graph from mesh-to-grid
      N_m2m_edges_level:
            number of edges in the graph from mesh-to-mesh at a given level
            (list index corresponds to the level)
      d_mesh_static:
            number of static features per mesh node (currently
            d_mesh_static==2, for x and y)
      N_mesh_nodes_level:
            number of nodes in the mesh at a given level

    And in addition for hierarchical graphs:
    - mesh_up_edge_index.pt
        list of [2, N_mesh_updown_edges_level], length==n_levels-1
    - mesh_up_features.pt
        list of [N_mesh_updown_edges_level, d_features], length==n_levels-1
    - mesh_down_edge_index.pt
        list of [2, N_mesh_updown_edges_level], length==n_levels-1
    - mesh_down_features.pt
        list of [N_mesh_updown_edges_level, d_features], length==n_levels-1

    where N_mesh_updown_edges_level is the number of edges in the graph from
    mesh-to-mesh between two consecutive levels (list index corresponds index
    of lower level)


    Parameters
    ----------
    graph_dir_path : str
        Path to store the graph components.
    xy : np.ndarray
        Grid coordinates, expected to be of shape (Nx, Ny, 2).
    n_max_levels : int
        Limit multi-scale mesh to given number of levels, from bottom up
        (default: None (no limit)).
    hierarchical : bool
        Generate hierarchical mesh graph (default: False).
    create_plot : bool
        If graphs should be plotted during generation (default: False).

    Returns
    -------
    None

    """
    os.makedirs(graph_dir_path, exist_ok=True)

    print(f"Writing graph components to {graph_dir_path}")

    grid_xy = torch.tensor(xy)
    pos_max = torch.max(torch.abs(grid_xy))

    #
    # Mesh
    #

    # graph geometry
    nx = 3  # number of children = nx**2
    nlev = int(np.log(max(xy.shape)) / np.log(nx))
    nleaf = nx**nlev  # leaves at the bottom = nleaf**2

    mesh_levels = nlev - 1
    if n_max_levels:
        # Limit the levels in mesh graph
        mesh_levels = min(mesh_levels, n_max_levels)

    print(f"nlev: {nlev}, nleaf: {nleaf}, mesh_levels: {mesh_levels}")

    # multi resolution tree levels
    G = []
    for lev in range(1, mesh_levels + 1):
        n = int(nleaf / (nx**lev))
        g = mk_2d_graph(xy, n, n, earth_mask)
        if create_plot:
            plot_graph(from_networkx(g), f"Mesh graph, level {lev}", graph_dir_path)
            plt.show()

        G.append(g)

    if hierarchical:
        # Relabel nodes of each level with level index first
        G = [prepend_node_index(graph, level_i) for level_i, graph in enumerate(G)]

        num_nodes_level = np.array([len(g_level.nodes) for g_level in G])
        # First node index in each level in the hierarchical graph
        first_index_level = np.concatenate(
            (np.zeros(1, dtype=int), np.cumsum(num_nodes_level[:-1]))
        )

        # Create inter-level mesh edges
        up_graphs = []
        down_graphs = []
        for from_level, to_level, G_from, G_to, start_index in zip(
            range(1, mesh_levels),
            range(0, mesh_levels - 1),
            G[1:],
            G[:-1],
            first_index_level[: mesh_levels - 1],
        ):
            # start out from graph at from level
            G_down = G_from.copy()
            G_down.clear_edges()
            G_down = networkx.DiGraph(G_down)

            # Add nodes of to level
            G_down.add_nodes_from(G_to.nodes(data=True))

            # build kd tree for mesh point pos
            # order in vm should be same as in vm_xy
            v_to_list = list(G_to.nodes)
            v_from_list = list(G_from.nodes)
            v_from_xy = np.array([xy for _, xy in G_from.nodes.data("pos")])
            kdt_m = scipy.spatial.KDTree(v_from_xy)

            # add edges from mesh to grid
            for v in v_to_list:
                # find 1(?) nearest neighbours (index to vm_xy)
                neigh_idx = kdt_m.query(G_down.nodes[v]["pos"], 1)[1]
                u = v_from_list[neigh_idx]

                # add edge from mesh to grid
                G_down.add_edge(u, v)
                d = np.sqrt(
                    np.sum((G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]) ** 2)
                )
                G_down.edges[u, v]["len"] = d
                G_down.edges[u, v]["vdiff"] = (
                    G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]
                )

            # relabel nodes to integers (sorted)
            G_down_int = networkx.convert_node_labels_to_integers(
                G_down, first_label=start_index, ordering="sorted"
            )  # Issue with sorting here
            G_down_int = sort_nodes_internally(G_down_int)
            pyg_down = from_networkx_with_start_index(G_down_int, start_index)

            # Create up graph, invert downwards edges
            up_edges = torch.stack(
                (pyg_down.edge_index[1], pyg_down.edge_index[0]), dim=0
            )
            pyg_up = pyg_down.clone()
            pyg_up.edge_index = up_edges

            up_graphs.append(pyg_up)
            down_graphs.append(pyg_down)

            if create_plot:
                plot_graph(
                    pyg_down,
                    f"Down graph, {from_level} -> {to_level}",
                    graph_dir_path,
                )
                plt.show()

                plot_graph(
                    pyg_down,
                    f"Up graph, {to_level} -> {from_level}",
                    graph_dir_path,
                )
                plt.show()

        # Save up and down edges
        save_edges_list(up_graphs, "mesh_up", graph_dir_path)
        save_edges_list(down_graphs, "mesh_down", graph_dir_path)

        # Extract intra-level edges for m2m
        m2m_graphs = [
            from_networkx_with_start_index(
                networkx.convert_node_labels_to_integers(
                    level_graph, first_label=start_index, ordering="sorted"
                ),
                start_index,
            )
            for level_graph, start_index in zip(G, first_index_level)
        ]

        mesh_pos = [graph.pos.to(torch.float32) for graph in m2m_graphs]

        # For use in g2m and m2g
        G_bottom_mesh = G[0]

        joint_mesh_graph = networkx.union_all([graph for graph in G])
        all_mesh_nodes = joint_mesh_graph.nodes(data=True)

    else:
        # Combine all levels into one multiscale graph
        G_tot = G[0].copy()

        # Iteratively merge levels into the graph
        for lev in range(1, len(G)):
            G_fine = G[lev]

            # Get nodes and their positions from current graph
            coarse_nodes = list(G_tot.nodes)
            coarse_pos = np.array([G_tot.nodes[n]["pos"] for n in coarse_nodes])

            # Get nodes and their positions from the new graph
            fine_nodes = list(G_fine.nodes)
            fine_pos = np.array([G_fine.nodes[n]["pos"] for n in fine_nodes])

            # Build a KDTree on the node positions
            kdtree = scipy.spatial.KDTree(coarse_pos)

            # For each node in the new graph,
            # find the index of the nearest node in the current graph
            _, parent_indices = kdtree.query(fine_pos)

            # Create a mapping from each fine node
            # to its corresponding coarse node parent
            relabel_mapping = {
                fine_node: coarse_nodes[parent_idx]
                for fine_node, parent_idx in zip(fine_nodes, parent_indices)
            }

            # Relabel the graph nodes
            G_fine_relabeled = networkx.relabel_nodes(
                G_fine, relabel_mapping, copy=True
            )

            # Compose the relabeled graph, merging nodes and edges
            G_tot = networkx.compose(G_tot, G_fine_relabeled)

        # Relabel mesh nodes to start with 0
        G_tot = prepend_node_index(G_tot, 0)

        # relabel nodes to integers (sorted)
        G_int = networkx.convert_node_labels_to_integers(
            G_tot, first_label=0, ordering="sorted"
        )

        # Graph to use in g2m and m2g
        G_bottom_mesh = G_tot
        all_mesh_nodes = G_tot.nodes(data=True)

        # export the nx graph to PyTorch geometric
        pyg_m2m = from_networkx(G_int)
        m2m_graphs = [pyg_m2m]
        mesh_pos = [pyg_m2m.pos.to(torch.float32)]

        if create_plot:
            plot_graph(pyg_m2m, "Mesh-to-mesh", graph_dir_path)
            plt.show()

    # Save m2m edges
    save_edges_list(m2m_graphs, "m2m", graph_dir_path)

    # Divide mesh node pos by max coordinate of grid cell
    mesh_pos = [pos / pos_max for pos in mesh_pos]

    # Save mesh positions
    torch.save(
        mesh_pos, os.path.join(graph_dir_path, "mesh_features.pt")
    )  # mesh pos, in float32

    #
    # Grid2Mesh
    #

    # radius within which grid nodes are associated with a mesh node
    # (in terms of mesh distance)
    DM_SCALE = 0.67

    # mesh nodes on lowest level
    vm = G_bottom_mesh.nodes
    vm_xy = np.array([xy for _, xy in vm.data("pos")])

    # find consecutive nodes on the same row
    vm_pos = {key: pos for key, pos in vm.data("pos")}
    sorted_keys = sorted(vm_pos.keys(), key=lambda k: (k[0], k[1], k[2]))
    key1, key2 = None, None
    for i in range(len(sorted_keys) - 1):
        k1, k2 = sorted_keys[i], sorted_keys[i + 1]
        if k1[0] == k2[0] and k1[1] == k2[1] and k1[2] + 1 == k2[2]:
            if np.array_equal(vm_pos[k1][1], vm_pos[k2][1]):
                key1, key2 = k1, k2
                break

    # distance between mesh nodes
    dm = np.sqrt(np.sum((vm.data("pos")[key1] - vm.data("pos")[key2]) ** 2))

    # grid nodes
    Ny, Nx = xy.shape[1:]

    G_grid = networkx.grid_2d_graph(Ny, Nx)
    G_grid.clear_edges()

    # vg features (only pos introduced here)
    nodes_to_remove = []
    for node in G_grid.nodes:
        # Remove the node from the graph if it is a earth node
        if earth_mask[node[0], node[1]]:
            nodes_to_remove.append(node)
        else:
            # pos is in feature but here explicit for convenience
            G_grid.nodes[node]["pos"] = np.array([xy[0][node], xy[1][node]])

    for node in nodes_to_remove:
        G_grid.remove_node(node)

    # add 1000 to node key to separate grid nodes (1000,i,j) from mesh nodes
    # (i,j) and impose sorting order such that vm are the first nodes
    G_grid = prepend_node_index(G_grid, 1000)

    # build kd tree for grid point pos
    # order in vg_list should be same as in vg_xy
    vg_list = list(G_grid.nodes)
    vg_xy = np.array([[xy[0][node[1:]], xy[1][node[1:]]] for node in vg_list])
    kdt_g = scipy.spatial.KDTree(vg_xy)

    # now add (all) mesh nodes, include features (pos)
    G_grid.add_nodes_from(all_mesh_nodes)

    # Re-create graph with sorted node indices
    # Need to do sorting of nodes this way for indices to map correctly to pyg
    G_g2m = networkx.Graph()
    G_g2m.add_nodes_from(sorted(G_grid.nodes(data=True)))

    # turn into directed graph
    G_g2m = networkx.DiGraph(G_g2m)

    # add edges
    for v in vm:
        # find neighbours (index to vg_xy)
        neigh_idxs = kdt_g.query_ball_point(vm[v]["pos"], dm * DM_SCALE)
        for i in neigh_idxs:
            u = vg_list[i]
            # add edge from grid to mesh
            G_g2m.add_edge(u, v)
            d = np.sqrt(np.sum((G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]) ** 2))
            G_g2m.edges[u, v]["len"] = d
            G_g2m.edges[u, v]["vdiff"] = G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]

    pyg_g2m = from_networkx(G_g2m)

    if create_plot:
        pyg_g2m_reversed = pyg_g2m.clone()
        pyg_g2m_reversed.edge_index = pyg_g2m.edge_index[[1, 0]]
        plot_graph(pyg_g2m_reversed, "Grid-to-mesh", graph_dir_path)
        plt.show()

    #
    # Mesh2Grid
    #

    # start out from Grid2Mesh and then replace edges
    G_m2g = G_g2m.copy()
    G_m2g.clear_edges()

    # build kd tree for mesh point pos
    # order in vm should be same as in vm_xy
    vm_list = list(vm)
    kdt_m = scipy.spatial.KDTree(vm_xy)

    # add edges from mesh to grid
    for v in vg_list:
        # find 4 nearest neighbours (index to vm_xy)
        neigh_idxs = kdt_m.query(G_m2g.nodes[v]["pos"], 4)[1]
        for i in neigh_idxs:
            u = vm_list[i]
            # add edge from mesh to grid
            G_m2g.add_edge(u, v)
            d = np.sqrt(np.sum((G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]) ** 2))
            G_m2g.edges[u, v]["len"] = d
            G_m2g.edges[u, v]["vdiff"] = G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]

    # relabel nodes to integers (sorted)
    G_m2g_int = networkx.convert_node_labels_to_integers(
        G_m2g, first_label=0, ordering="sorted"
    )
    pyg_m2g = from_networkx(G_m2g_int)

    if create_plot:
        plot_graph(pyg_m2g, "Mesh-to-grid", graph_dir_path)
        plt.show()

    # Save g2m and m2g everything
    # g2m
    save_edges(pyg_g2m, "g2m", graph_dir_path)
    # m2g
    save_edges(pyg_m2g, "m2g", graph_dir_path)


def create_graph_from_datastore(
    datastore: BaseRegularGridDatastore,
    output_root_path: str,
    n_max_levels: int = None,
    hierarchical: bool = False,
    create_plot: bool = False,
):
    if isinstance(datastore, BaseRegularGridDatastore):
        earth_mask = datastore.get_mask(stacked=False, invert=True)
        y_idx, x_idx = np.indices(earth_mask.shape)
        xy = np.stack([x_idx, y_idx], axis=0)
    else:
        raise NotImplementedError(
            "Only graph creation for BaseRegularGridDatastore is supported"
        )

    create_graph(
        graph_dir_path=output_root_path,
        xy=xy,
        earth_mask=earth_mask,
        n_max_levels=n_max_levels,
        hierarchical=hierarchical,
        create_plot=create_plot,
    )


def cli(input_args=None):
    parser = ArgumentParser(description="Graph generation arguments")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to neural-lam configuration file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="multiscale",
        help="Name to save graph as (default: multiscale)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If graphs should be plotted during generation " "(default: False)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Limit multi-scale mesh to given number of levels, "
        "from bottom up (default: None (no limit))",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Generate hierarchical mesh graph (default: False)",
    )
    args = parser.parse_args(input_args)

    assert args.config_path is not None, "Specify your config with --config_path"

    # Load neural-lam configuration and datastore to use
    _, datastore = load_config_and_datastore(config_path=args.config_path)

    create_graph_from_datastore(
        datastore=datastore,
        output_root_path=os.path.join(datastore.root_path, "graph", args.name),
        n_max_levels=args.levels,
        hierarchical=args.hierarchical,
        create_plot=args.plot,
    )


if __name__ == "__main__":
    cli()
