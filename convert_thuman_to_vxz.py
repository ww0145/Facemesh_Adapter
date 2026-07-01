#!/usr/bin/env python
"""
Convert THuman2.1 OBJ meshes to O-Voxel VXZ files for TRELLIS.2 SC-VAE.

Place this script at the TRELLIS.2 repository root:

    TRELLIS.2/convert_thuman_to_vxz.py

Example:
    python convert_thuman_to_vxz.py \
        --thuman_dir ~/TRELLIS/model \
        --resolution 1024 \
        --num_samples 1 \
        --out_dir data/thuman_vxz
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
import trimesh
import o_voxel
from pathlib import Path
from tqdm import tqdm


def load_obj_mesh(obj_path):
    """Load an OBJ mesh and normalize vertices into the [-0.5, 0.5] cube."""
    mesh = trimesh.load(obj_path, process=False, force='mesh')
    vertices = torch.from_numpy(np.array(mesh.vertices)).float()
    faces = torch.from_numpy(np.array(mesh.faces)).long()

    v_min = vertices.min(dim=0)[0]
    v_max = vertices.max(dim=0)[0]
    center = (v_min + v_max) / 2
    scale = 0.99999 / (v_max - v_min).max()
    vertices = (vertices - center) * scale

    assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), \
        f"vertices out of range: [{vertices.min():.4f}, {vertices.max():.4f}]"

    return vertices, faces


def mesh_to_vxz(vertices, faces, resolution, output_path):
    """Convert normalized mesh geometry to a VXZ dual-grid file."""
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices,
        faces=faces,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False,
    )

    dual_vertices = dual_vertices * resolution - voxel_indices
    dual_vertices = torch.clamp(dual_vertices, 0, 1)
    dual_vertices = (dual_vertices * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o_voxel.io.write_vxz(
        output_path,
        voxel_indices,
        {'vertices': dual_vertices, 'intersected': intersected},
    )

    return len(dual_vertices)


def find_thuman_objs(thuman_dir, num_samples=None):
    """Find OBJ files under common THuman2.1 directory layouts."""
    thuman_dir = Path(thuman_dir)
    obj_files = []

    for d in sorted(thuman_dir.iterdir()):
        if d.is_dir():
            obj = d / f"{d.name}.obj"
            if obj.exists():
                obj_files.append(obj)
            else:
                objs = list(d.glob("*.obj"))
                if objs:
                    obj_files.append(objs[0])
        elif d.suffix == '.obj':
            obj_files.append(d)

    if num_samples:
        obj_files = obj_files[:num_samples]
    return obj_files


def main():
    parser = argparse.ArgumentParser(description="Convert THuman OBJ meshes to O-Voxel VXZ files")
    parser.add_argument("--thuman_dir", type=str, default=None,
                        help="THuman2.1 root directory")
    parser.add_argument("--obj_files", type=str, nargs='+', default=None,
                        help="Explicit OBJ file paths")
    parser.add_argument("--resolution", type=int, default=512,
                        help="O-Voxel grid resolution")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Maximum number of samples to convert")
    parser.add_argument("--out_dir", type=str, default="data/thuman_vxz",
                        help="Output directory")
    args = parser.parse_args()

    if args.obj_files:
        obj_files = [Path(f) for f in args.obj_files]
    elif args.thuman_dir:
        obj_files = find_thuman_objs(args.thuman_dir, args.num_samples)
    else:
        parser.error("Please specify --thuman_dir or --obj_files")

    if not obj_files:
        print("ERROR: no OBJ files found")
        return

    out_dir = Path(args.out_dir) / f"dual_grid_{args.resolution}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(obj_files)} samples to O-Voxel {args.resolution}")
    print(f"Output: {out_dir}\n")

    for obj_path in tqdm(obj_files, desc="Converting"):
        sample_id = obj_path.stem
        vxz_path = out_dir / f"{sample_id}.vxz"

        if vxz_path.exists():
            print(f"  {sample_id}: exists, skip")
            continue

        try:
            vertices, faces = load_obj_mesh(str(obj_path))
            n_voxels = mesh_to_vxz(vertices, faces, args.resolution, str(vxz_path))
            print(f"  {sample_id}: {vertices.shape[0]} verts -> {n_voxels} voxels")
        except Exception as e:
            print(f"  {sample_id}: ERROR - {e}")
            continue

    print(f"\nDone. VXZ files saved to: {out_dir}/")
    print("\nNext step:")
    print("  CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/data/preprocess_thuman.py \\")
    print("      --thuman_dir ~/TRELLIS/model \\")
    print(f"      --vxz_dir {out_dir} \\")
    print("      --out_dir data/adapter_train_hq_1024 \\")
    print(f"      --resolution {args.resolution} \\")
    print("      --save_debug_images --hq_condition --overwrite")


if __name__ == "__main__":
    main()
