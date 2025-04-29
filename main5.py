
import argparse
import os
import nibabel as nib
import numpy as np
from skimage import measure
from skimage.morphology import skeletonize
import open3d as o3d
import networkx as nx
import plotly.graph_objects as go


def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.header


def downsample_volume(vol, factor):
    if factor <= 1:
        return vol
    shape = np.array(vol.shape) // factor * factor
    vol_cropped = vol[:shape[0], :shape[1], :shape[2]]
    return vol_cropped.reshape(
        shape[0]//factor, factor,
        shape[1]//factor, factor,
        shape[2]//factor, factor
    ).mean(axis=(1,3,5))


def crop_roi(vol, roi):
    xmin, xmax, ymin, ymax, zmin, zmax = map(int, roi.split(','))
    return vol[xmin:xmax, ymin:ymax, zmin:zmax]


def extract_mesh(vol, threshold):
    verts, faces, _, _ = measure.marching_cubes(vol, level=threshold)
    return verts, faces


def simplify_mesh(verts, faces, max_triangles):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces)
    )
    mesh.compute_vertex_normals()
    simplified = mesh.simplify_quadric_decimation(max_triangles)
    return np.asarray(simplified.vertices), np.asarray(simplified.triangles)


def compute_3d_volume(vol, header, threshold=0.5):
    binary = vol > threshold
    dx, dy, dz = header.get_zooms()
    voxel_vol = dx * dy * dz
    return binary.sum() * voxel_vol


def compute_2d_extent(vol, header, threshold=0.5):
    binary = vol > threshold
    coords = np.column_stack(np.nonzero(binary))
    if coords.size == 0:
        return 0.0, 0.0
    i_min, j_min, _ = coords.min(axis=0)
    i_max, j_max, _ = coords.max(axis=0)
    dx, dy, _ = header.get_zooms()
    width = (i_max - i_min + 1) * dx
    height = (j_max - j_min + 1) * dy
    return width, height


def compute_centerline_length(vol, header, threshold=0.5):
    binary = vol > threshold
    skel = skeletonize(binary)
    spacing = np.array(header.get_zooms())
    total_length = skel.sum() * spacing.mean()
    return total_length


def compute_geodesic_endpoints_length(vol, header, threshold=0.5):
    binary = vol > threshold
    skel = skeletonize(binary)
    coords = set(zip(*np.nonzero(skel)))
    # Build adjacency list for skeleton voxels
    spacing = np.array(header.get_zooms())
    deltas = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    G = nx.Graph()
    # Add edges between 6-connected skeleton voxels
    for node in coords:
        for d in deltas:
            neigh = (node[0]+d[0], node[1]+d[1], node[2]+d[2])
            if neigh in coords:
                dist = np.linalg.norm(np.array(d) * spacing)
                G.add_edge(node, neigh, weight=dist)
    # Find endpoints: degree == 1
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    if len(endpoints) < 2:
        return None, None, None
    # Compute pairwise geodesic distances, pick max
    max_len = 0.0
    start, end = None, None
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            u, v = endpoints[i], endpoints[j]
            try:
                length = nx.dijkstra_path_length(G, u, v, weight='weight')
                if length > max_len:
                    max_len = length
                    start, end = u, v
            except nx.NetworkXNoPath:
                continue
    return start, end, max_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('body_nifti')
    parser.add_argument('mask_nifti')
    parser.add_argument('output_dir')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--max_triangles', type=int, default=None)
    parser.add_argument('--opacity_body', type=float, default=0.5)
    parser.add_argument('--opacity_mask', type=float, default=0.8)
    parser.add_argument('--flatshading', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vol_body, hdr_body = load_nifti(args.body_nifti)
    vol_mask, hdr_mask = load_nifti(args.mask_nifti)
    if args.downsample > 1:
        vol_body = downsample_volume(vol_body, args.downsample)
        vol_mask = downsample_volume(vol_mask, args.downsample)
    if args.roi:
        vol_body = crop_roi(vol_body, args.roi)
        vol_mask = crop_roi(vol_mask, args.roi)

    # Extract and simplify meshes
    vb, fb = extract_mesh(vol_body, args.threshold)
    vm, fm = extract_mesh(vol_mask, args.threshold)
    if args.max_triangles:
        vb, fb = simplify_mesh(vb, fb, args.max_triangles)
        vm, fm = simplify_mesh(vm, fm, args.max_triangles)
    mb = go.Mesh3d(x=vb[:,0], y=vb[:,1], z=vb[:,2],
                   i=fb[:,0], j=fb[:,1], k=fb[:,2],
                   color='green', opacity=args.opacity_body,
                   flatshading=args.flatshading)
    mm = go.Mesh3d(x=vm[:,0], y=vm[:,1], z=vm[:,2],
                   i=fm[:,0], j=fm[:,1], k=fm[:,2],
                   color='red', opacity=args.opacity_mask,
                   flatshading=args.flatshading)
    fig = go.Figure(data=[mb, mm])
    fig.update_layout(autosize=False, width=800, height=600,
                      scene=dict(aspectmode='data',
                                 xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)),
                      margin=dict(l=0,r=0,t=0,b=0))
    html_path = os.path.join(
        args.output_dir,
        os.path.splitext(os.path.basename(args.mask_nifti))[0] + '_overlay.html'
    )
    fig.write_html(html_path, include_plotlyjs='cdn')

    # Compute metrics
    vol3d_body = compute_3d_volume(vol_body, hdr_body, args.threshold)
    vol3d_mask = compute_3d_volume(vol_mask, hdr_mask, args.threshold)
    w_body, h_body = compute_2d_extent(vol_body, hdr_body, args.threshold)
    w_mask, h_mask = compute_2d_extent(vol_mask, hdr_mask, args.threshold)
    skeleton_len = compute_centerline_length(vol_mask, hdr_mask, args.threshold)
    start, end, geo_len = compute_geodesic_endpoints_length(vol_mask, hdr_mask, args.threshold)

    # Save metrics
    txt_path = os.path.join(
        args.output_dir,
        os.path.splitext(os.path.basename(args.mask_nifti))[0] + '_metrics.txt'
    )
    with open(txt_path, 'w') as f:
        f.write(f"Body 3D Volume (mm^3): {vol3d_body:.2f}\n")
        f.write(f"Mask 3D Volume (mm^3): {vol3d_mask:.2f}\n")
        f.write(f"Body 2D Width (mm): {w_body:.2f}, Height: {h_body:.2f}\n")
        f.write(f"Mask 2D Width (mm): {w_mask:.2f}, Height: {h_mask:.2f}\n")
        f.write(f"Skeleton Total Length (mm): {skeleton_len:.2f}\n")
        if geo_len is not None:
            f.write(f"Geodesic Endpoints Length (mm): {geo_len:.2f}\n")
            f.write(f"Start Voxel Index: {start}, End Voxel Index: {end}\n")
    print(f"Saved metrics → {txt_path}")

if __name__ == '__main__':
    main()
