import trimesh
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import networkx as nx
from matplotlib.path import Path

def slice(INPUT_FILE, OUTPUT_FOLDER, NUM_SLICES=100, RESOLUTION = (1024,1024), SCAN_AXIS='Y'):
    mesh = trimesh.load(INPUT_FILE)
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    bounds = mesh.bounds
    if SCAN_AXIS.upper() == 'X':
        levels = np.linspace(bounds[0][0] + 0.001, bounds[1][0] - 0.001, NUM_SLICES)
        plane_normal, view_indices = [1, 0, 0], [1, 2]
        lims = (bounds[0][1], bounds[1][1], bounds[0][2], bounds[1][2])
    elif SCAN_AXIS.upper() == 'Y':
        levels = np.linspace(bounds[0][1] + 0.001, bounds[1][1] - 0.001, NUM_SLICES)
        plane_normal, view_indices = [0, 1, 0], [0, 2]
        lims = (bounds[0][0], bounds[1][0], bounds[0][2], bounds[1][2])
    else:
        levels = np.linspace(bounds[0][2] + 0.001, bounds[1][2] - 0.001, NUM_SLICES)
        plane_normal, view_indices = [0, 0, 1], [0, 1]
        lims = (bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1])

    fig_size = (RESOLUTION[0] / 100, RESOLUTION[1] / 100)

    for i, val in tqdm(enumerate(levels), total=NUM_SLICES):
        origin = [0, 0, 0]
        origin['XYZ'.find(SCAN_AXIS.upper())] = val
        lines = trimesh.intersections.mesh_plane(mesh, plane_normal=plane_normal, plane_origin=origin)
        
        fig, ax = plt.subplots(figsize=fig_size, dpi=100)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        if len(lines) > 0:
            segments = lines[:, :, view_indices]
            G = nx.Graph()
            for seg in segments:
                p1, p2 = tuple(np.round(seg[0], 4)), tuple(np.round(seg[1], 4))
                G.add_edge(p1, p2)
            
            # Extract all loops as Matplotlib Paths
            paths = []
            for component in nx.connected_components(G):
                if len(component) > 2:
                    subgraph = G.subgraph(component)
                    ordered = list(nx.dfs_preorder_nodes(subgraph, list(component)[0]))
                    paths.append(Path(np.array(ordered + [ordered[0]]))) # Ensure closed loop

            # Sort paths by area to handle nested holes
            # We use the bounding box area as a quick proxy for sorting
            paths.sort(key=lambda p: (p.get_extents().width * p.get_extents().height), reverse=True)

            # Check if path is a "hole"
            for current_path in paths:
                is_hole = False
                # If this path is inside any other path, it's a hole
                for other_path in paths:
                    if current_path == other_path: continue
                    # Check if the first point of current_path is inside other_path
                    if other_path.contains_point(current_path.vertices[0]):
                        is_hole = not is_hole # Toggle (handles nested islands)

                color = 'black' if is_hole else 'white'
                # Fill
                patch = plt.Polygon(current_path.vertices, facecolor=color, edgecolor=color, linewidth=0.5)
                ax.add_patch(patch)

        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])
        ax.set_aspect('equal')
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"slice_{i:03d}.png"), facecolor='black')
        plt.close(fig)