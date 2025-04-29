#!/usr/bin/env python3
"""
Overlay a single organ mask on a full-body segmentation and compute 3D volumes and 2D extents.
Generates an interactive 3D HTML viewer and saves measurements to a TXT file in the output folder.

Usage:
  python nifti_to_interactive_viewer.py \
    <body.nii.gz> <mask.nii.gz> <output_dir> \
    [--threshold 0.5] [--downsample 1] [--roi xmin,xmax,ymin,ymax,zmin,zmax] \
    [--max_triangles 50000] [--opacity_body 0.5] [--opacity_mask 0.8] [--flatshading]
"""
import argparse
import os
import nibabel as nib
import numpy as np
from skimage import measure
import open3d as o3d
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
    xmin,xmax,ymin,ymax,zmin,zmax = map(int, roi.split(','))
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
    # Binary mask and voxel volume
    binary = vol > threshold
    zooms = header.get_zooms()
    voxel_vol = zooms[0] * zooms[1] * zooms[2]
    return binary.sum() * voxel_vol

def compute_2d_extent(vol, header, threshold=0.5):
    # Compute bounding box in x-y plane
    binary = vol > threshold
    coords = np.column_stack(np.nonzero(binary))  # (i,j,k)
    if coords.size == 0:
        return 0.0, 0.0
    i_min, j_min, _ = coords.min(axis=0)
    i_max, j_max, _ = coords.max(axis=0)
    dx, dy, _ = header.get_zooms()
    width  = (i_max - i_min + 1) * dx
    height = (j_max - j_min + 1) * dy
    return width, height

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('body_nifti', help='Body segmentation NIfTI')
    parser.add_argument('mask_nifti', help='Single organ mask NIfTI')
    parser.add_argument('output_dir', help='Output folder for HTML and TXT')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--max_triangles', type=int, default=None)
    parser.add_argument('--opacity_body', type=float, default=0.5)
    parser.add_argument('--opacity_mask', type=float, default=0.8)
    parser.add_argument('--flatshading', action='store_true')
    args = parser.parse_args()

    # ensure output_dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # load and preprocess
    vol_body, hdr_body = load_nifti(args.body_nifti)
    vol_mask, hdr_mask = load_nifti(args.mask_nifti)
    if args.downsample > 1:
        vol_body = downsample_volume(vol_body, args.downsample)
        vol_mask = downsample_volume(vol_mask, args.downsample)
    if args.roi:
        vol_body = crop_roi(vol_body, args.roi)
        vol_mask = crop_roi(vol_mask, args.roi)

    # extract meshes
    vb, fb = extract_mesh(vol_body, args.threshold)
    vm, fm = extract_mesh(vol_mask, args.threshold)
    if args.max_triangles:
        vb, fb = simplify_mesh(vb, fb, args.max_triangles)
        vm, fm = simplify_mesh(vm, fm, args.max_triangles)

    # build plotly meshes
    mb = go.Mesh3d(
        x=vb[:,0], y=vb[:,1], z=vb[:,2],
        i=fb[:,0], j=fb[:,1], k=fb[:,2],
        color='green', opacity=args.opacity_body, flatshading=args.flatshading
    )
    mm = go.Mesh3d(
        x=vm[:,0], y=vm[:,1], z=vm[:,2],
        i=fm[:,0], j=fm[:,1], k=fm[:,2],
        color='red', opacity=args.opacity_mask, flatshading=args.flatshading
    )

    # save HTML
    html_path = os.path.join(
        args.output_dir,
        os.path.splitext(os.path.basename(args.mask_nifti))[0] + '_overlay.html'
    )
    fig = go.Figure(data=[mb, mm])
    fig.update_layout(scene=dict(aspectmode='data',
                       xaxis=dict(visible=False),
                       yaxis=dict(visible=False),
                       zaxis=dict(visible=False)),
                      margin=dict(l=0,r=0,t=0,b=0))
    fig.write_html(html_path)
   

    # compute metrics
    vol3d_body = compute_3d_volume(vol_body, hdr_body, args.threshold)
    vol3d_mask = compute_3d_volume(vol_mask, hdr_mask, args.threshold)
    w_body, h_body = compute_2d_extent(vol_body, hdr_body, args.threshold)
    w_mask, h_mask = compute_2d_extent(vol_mask, hdr_mask, args.threshold)

    # write metrics
    txt_path = os.path.join(
        args.output_dir,
        os.path.splitext(os.path.basename(args.mask_nifti))[0] + '_metrics.txt'
    )
    with open(txt_path, 'w') as f:
        f.write(f"Body 3D Volume (ml): {vol3d_body:.2f}\n")
        f.write(f"Mask 3D Volume (ml): {vol3d_mask:.2f}\n")
        f.write(f"Body 2D Width (mm): {w_body:.2f}\n")
        f.write(f"Body 2D Height (mm): {h_body:.2f}\n")
        f.write(f"Mask 2D Width (mm): {w_mask:.2f}\n")
        f.write(f"Mask 2D Height (mm): {h_mask:.2f}\n")
    print(f"Saved metrics → {txt_path}")

if __name__ == '__main__':
    main()