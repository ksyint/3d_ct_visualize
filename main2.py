#!/usr/bin/env python3
"""
nifti_to_interactive_viewer.py

여러 NIfTI(.nii/.nii.gz) 세그멘테이션 마스크(예: 전체 신체 CT + 복수의 장기 마스크)를
초록색 본체(body) 위에 각기 다른 색으로 오버레이한 인터랙티브 3D HTML 뷰어를 생성하고,
동일 경로에 각 마스크의 라벨을 담은 TXT 파일을 기록합니다.

Usage:
  python nifti_to_interactive_viewer.py \
    <body.nii.gz> <output.html> <mask1.nii.gz> [<mask2.nii.gz> ...] \
    [--threshold 0.5] [--downsample 2] [--roi xmin,xmax,ymin,ymax,zmin,zmax] \
    [--max_triangles 50000] [--opacity_body 0.5] [--opacity_masks 0.8] [--flatshading]
"""
import argparse
import os
import nibabel as nib
import numpy as np
from skimage import measure
import open3d as o3d
import plotly.graph_objects as go

def load_nifti(path):
    return nib.load(path).get_fdata()

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('body_nifti', help='Body segmentation NIfTI')
    parser.add_argument('output_html', help='Output HTML path')
    parser.add_argument('mask_niftis', nargs='+', help='One or more mask NIfTIs')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--max_triangles', type=int, default=None)
    parser.add_argument('--opacity_body', type=float, default=0.5)
    parser.add_argument('--opacity_masks', type=float, default=0.8)
    parser.add_argument('--flatshading', action='store_true')
    args = parser.parse_args()

    # 1) Load & preprocess body
    vol_body = load_nifti(args.body_nifti)
    if args.downsample > 1:
        vol_body = downsample_volume(vol_body, args.downsample)
    if args.roi:
        vol_body = crop_roi(vol_body, args.roi)

    vb, fb = extract_mesh(vol_body, args.threshold)
    if args.max_triangles:
        vb, fb = simplify_mesh(vb, fb, args.max_triangles)

    # 2) Prepare body Mesh3d (green)
    xb, yb, zb = vb.T
    ib, jb, kb = fb.T
    mesh_body = go.Mesh3d(
        x=xb, y=yb, z=zb,
        i=ib, j=jb, k=kb,
        color='green',
        opacity=args.opacity_body,
        flatshading=args.flatshading
    )

    # 3) Load & build each mask mesh with distinct colors
    COLORS = ['blue','red','yellow','purple','orange','cyan','magenta','grey','brown','pink']
    mask_meshes = []
    labels = []
    for idx, mask_path in enumerate(args.mask_niftis):
        vol_mask = load_nifti(mask_path)
        if args.downsample > 1:
            vol_mask = downsample_volume(vol_mask, args.downsample)
        if args.roi:
            vol_mask = crop_roi(vol_mask, args.roi)

        vm, fm = extract_mesh(vol_mask, args.threshold)
        if args.max_triangles:
            vm, fm = simplify_mesh(vm, fm, args.max_triangles)

        x, y, z = vm.T
        i, j, k = fm.T
        color = COLORS[idx % len(COLORS)]
        mask_meshes.append(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=args.opacity_masks,
            flatshading=args.flatshading
        ))

        # derive label from filename (strip .nii or .nii.gz)
        base = os.path.basename(mask_path)
        label = base[:-7] if base.endswith('.nii.gz') else base[:-4]
        label+=" ---> "
        label+=COLORS[idx % len(COLORS)]
        labels.append(label)

    # 4) Compose figure and save
    fig = go.Figure(data=[mesh_body] + mask_meshes)
    fig.update_layout(
        scene=dict(aspectmode='data',
                   xaxis=dict(showgrid=False, visible=False),
                   yaxis=dict(showgrid=False, visible=False),
                   zaxis=dict(showgrid=False, visible=False)),
        margin=dict(l=0,r=0,t=0,b=0)
    )
    fig.write_html(args.output_html)
    print(f"Saved HTML → {args.output_html}")

    # 5) Write labels to TXT alongside HTML
    txt_path = os.path.splitext(args.output_html)[0] + '.txt'
    with open(txt_path, 'w') as f:
        for label in labels:
            f.write(label + '\n')
    print(f"Saved labels → {txt_path}")

if __name__ == '__main__':
    main()
