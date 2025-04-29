#!/usr/bin/env python3
"""
nifti_to_interactive_viewer.py

두 개의 NIfTI(.nii/.nii.gz) 세그멘테이션 마스크(예: 전체 신체 CT와 심장 영역)를 오버레이하여
인체 전체(초록)와 심장(빨강)을 동시에 보여주는 인터랙티브 3D HTML 뷰어를 생성합니다.

Usage:
  pip install nibabel scikit-image plotly open3d numpy
  python nifti_to_interactive_viewer.py \
    <body.nii.gz> <heart.nii.gz> <output.html> \
    [--threshold 0.5] [--downsample 2] [--roi xmin,xmax,ymin,ymax,zmin,zmax] \
    [--max_triangles 50000] [--opacity_body 0.5] [--opacity_heart 0.8] [--flatshading]
"""
import argparse
import nibabel as nib
import numpy as np
from skimage import measure
import open3d as o3d
import plotly.graph_objects as go

def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata()

def downsample_volume(vol, factor):
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

def build_figure(verts_body, faces_body, verts_heart, faces_heart,
                 opacity_body, opacity_heart, flatshading):
    # Body mesh (green)
    xb, yb, zb = verts_body.T
    ib, jb, kb = faces_body.T
    mesh_body = go.Mesh3d(
        x=xb, y=yb, z=zb,
        i=ib, j=jb, k=kb,
        color='green',
        opacity=opacity_body,
        flatshading=flatshading
    )
    # Heart mesh (red)
    xh, yh, zh = verts_heart.T
    ih, jh, kh = faces_heart.T
    mesh_heart = go.Mesh3d(
        x=xh, y=yh, z=zh,
        i=ih, j=jh, k=kh,
        color='red',
        opacity=opacity_heart,
        flatshading=flatshading
    )
    fig = go.Figure(data=[mesh_body, mesh_heart])
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            zaxis=dict(showgrid=False, visible=False)
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

def main():
    parser = argparse.ArgumentParser(
        description='Overlay two NIfTI segmentations in interactive 3D HTML'
    )
    parser.add_argument('body_nifti', help='Body segmentation NIfTI path')
    parser.add_argument('heart_nifti', help='Heart segmentation NIfTI path')
    parser.add_argument('output_html', help='Output HTML file path')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Iso-surface level for both volumes')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Uniform downsample factor')
    parser.add_argument('--roi', type=str, default=None,
                        help='ROI crop as xmin,xmax,ymin,ymax,zmin,zmax')
    parser.add_argument('--max_triangles', type=int, default=None,
                        help='Max triangles for mesh simplification')
    parser.add_argument('--opacity_body', type=float, default=0.5,
                        help='Opacity for body mesh')
    parser.add_argument('--opacity_heart', type=float, default=0.8,
                        help='Opacity for heart mesh')
    parser.add_argument('--flatshading', action='store_true',
                        help='Enable flat shading for both meshes')
    args = parser.parse_args()

    # Load volumes
    vol_body = load_nifti(args.body_nifti)
    vol_heart = load_nifti(args.heart_nifti)
    # Preprocess
    if args.downsample > 1:
        vol_body = downsample_volume(vol_body, args.downsample)
        vol_heart = downsample_volume(vol_heart, args.downsample)
    if args.roi:
        vol_body = crop_roi(vol_body, args.roi)
        vol_heart = crop_roi(vol_heart, args.roi)
    # Extract meshes
    vb, fb = extract_mesh(vol_body, args.threshold)
    vh, fh = extract_mesh(vol_heart, args.threshold)
    # Simplify if requested
    if args.max_triangles:
        vb, fb = simplify_mesh(vb, fb, args.max_triangles)
        vh, fh = simplify_mesh(vh, fh, args.max_triangles)
    # Build and save
    fig = build_figure(vb, fb, vh, fh,
                       args.opacity_body, args.opacity_heart,
                       args.flatshading)
    fig.write_html(args.output_html)
    print(f"Saved overlay viewer to {args.output_html}")

if __name__ == '__main__':
    main()
