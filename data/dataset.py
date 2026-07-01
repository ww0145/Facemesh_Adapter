"""
Dataset for FaceMesh-Adapter v2 training.

Loads preprocessed .pt files from preprocess_thuman.py.

Each .pt file contains:
    - shape_coords: [N, 3] uint8 sparse coordinates
    - shape_feats: [N, 32] float32 SC-VAE encoded shape latent
    - cond: [N_tok, 1024] DINOv3 full-body condition tokens
    - neg_cond: [N_tok, 1024] negative condition (zeros)
    - id_emb: [1024] head identity embedding
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class ShapeAdapterDataset(Dataset):
    """
    Dataset for Shape flow model adapter training.

    Returns shape GT latent as SparseTensor-ready format,
    plus full-body cond and head identity embedding.
    """

    def __init__(self, data_dir: str, split: str = 'train', val_ratio: float = 0.1):
        all_files = sorted(Path(data_dir).glob("*.pt"))
        assert len(all_files) > 0, f"No .pt files in {data_dir}"

        # Simple train/val split
        n_val = max(1, int(len(all_files) * val_ratio))
        if split == 'val':
            self.files = all_files[-n_val:]
        else:
            self.files = all_files[:-n_val] if n_val < len(all_files) else all_files

        # Load one sample to check format
        d = torch.load(self.files[0], map_location='cpu', weights_only=False)
        self.latent_dim = d['shape_feats'].shape[1]  # 32

        print(f"[ShapeAdapterDataset] {split}: {len(self.files)} samples")
        print(f"  latent_dim: {self.latent_dim}")
        print(f"  cond shape: {d['cond'].shape}")
        print(f"  id_emb shape: {d['id_emb'].shape}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location='cpu', weights_only=False)

        # Reconstruct coords with batch dim 0
        coords_np = d['shape_coords']  # [N, 3] uint8
        coords = torch.from_numpy(coords_np).int()
        batch_col = torch.zeros(coords.shape[0], 1, dtype=torch.int32)
        coords = torch.cat([batch_col, coords], dim=1)  # [N, 4]

        feats = torch.from_numpy(d['shape_feats']).float()  # [N, 32]

        return {
            'coords': coords,                    # [N, 4] int (batch_idx, x, y, z)
            'shape_feats': feats,                 # [N, 32] GT shape latent
            'cond': d['cond'],                    # [N_tok, 1024] full-body DINOv3
            'neg_cond': d['neg_cond'],            # [N_tok, 1024]
            'id_emb': d['id_emb'],                # [1024] head identity
            'sample_id': d.get('sample_id', ''),
        }


class TextureAdapterDataset(Dataset):
    """
    Dataset for Texture flow model adapter training.

    Placeholder — requires PBR voxel data from voxelize_pbr.py
    and tex SC-VAE encoding. Implement after shape adapter works.

    Expected format matches ShapeAdapterDataset but with:
        - tex_coords, tex_feats: texture GT latent
        - shape_feats: for concat_cond (shape conditions texture)
    """

    def __init__(self, data_dir: str, **kwargs):
        raise NotImplementedError(
            "TextureAdapterDataset: implement after shape adapter training works. "
            "Requires PBR voxel data (voxelize_pbr.py) + tex SC-VAE encoding."
        )


def collate_sparse(batch):
    """
    Custom collate for variable-length sparse data.

    Adjusts batch indices in coords and stacks conditions.
    """
    all_coords = []
    all_feats = []
    all_cond = []
    all_neg_cond = []
    all_id_emb = []

    for i, sample in enumerate(batch):
        coords = sample['coords'].clone()
        coords[:, 0] = i  # set batch index
        all_coords.append(coords)
        all_feats.append(sample['shape_feats'])
        all_cond.append(sample['cond'])
        all_neg_cond.append(sample['neg_cond'])
        all_id_emb.append(sample['id_emb'])

    return {
        'coords': torch.cat(all_coords, dim=0),       # [N_total, 4]
        'shape_feats': torch.cat(all_feats, dim=0),    # [N_total, 32]
        'cond': torch.stack(all_cond, dim=0),           # [B, N_tok, 1024]
        'neg_cond': torch.stack(all_neg_cond, dim=0),
        'id_emb': torch.stack(all_id_emb, dim=0),      # [B, 1024]
    }