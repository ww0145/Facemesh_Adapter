# FaceMesh-Adapter v2 for TRELLIS.2

FaceMesh-Adapter v2 is an experimental identity adapter for the TRELLIS.2
image-to-3D pipeline. The goal is to preserve face identity when generating a
full-body 3D human/avatar from decoupled visual conditions.

The current implementation focuses on the **shape SLat stage**. Sparse
structure (SS) is kept frozen. The shape flow model is wrapped by an
IP-Adapter-style module that injects head identity features through sparse
cross-attention layers.

## Current Status

- Shape adapter training and inference are implemented.
- SS stage is frozen and can optionally use GT sparse coords for overfit/debug.
- Base condition is a head-masked full-body image encoded by TRELLIS.2 DINOv3.
- Adapter condition is a cropped head image encoded by TRELLIS.2 DINOv3 and
  mean-pooled into `id_emb`.
- Texture adapter is still a placeholder.
- Debug tools are included for latent-space and mesh-space error analysis.

Recent 1024 overfit/debug setting:

```text
Shape normalization loaded: mean [-3.19, 3.79], std [4.55, 6.24]
Trainable: 770.8M, Frozen: 1292.3M
Dataset sample:
  latent_dim: 32
  cond shape: [1029, 1024]
  id_emb shape: [1024]
```

The trainable parameter count depends on `num_id_tokens`, `ip_interval`, and the
chosen flow model (`shape_slat_flow_model_512` or `shape_slat_flow_model_1024`).
For example, 1024 shape overfit with 16 identity tokens and IP layers at every
block has about 770.8M trainable parameters.

## Architecture Overview

```text
THuman2.1 mesh
  -> O-Voxel / VXZ
  -> TRELLIS.2 shape encoder
  -> shape_coords + shape_feats              (shape target)

Full-body render
  -> head masking
  -> TRELLIS.2 DINOv3 cond tokens            (base model condition)

Head crop render
  -> TRELLIS.2 DINOv3 tokens
  -> mean pool
  -> id_emb                                  (adapter condition)

Frozen SS stage:
  cond -> sparse coords

Shape stage:
  SLat flow model + IP-Adapter(id_emb)
  flow matching loss against GT shape latent
```

In overfit/debug mode, the SS stage can be bypassed with GT coords from the
preprocessed `.pt` file. This isolates shape adapter behavior from sparse
structure sampling.

## Important Design Choices

1. **SS is frozen by default**

   Sparse structure mainly controls body occupancy/silhouette. The current
   experiments focus on shape latent feature alignment, so SS is not trained.

2. **Two-condition design**

   The base model receives a head-masked full-body condition. The adapter
   receives a head-only identity condition. This avoids leaking facial identity
   through the base condition.

3. **Mean-pooled head identity**

   The baseline uses mean-pooled DINOv3 head tokens as `id_emb`. Full head-token
   conditioning was tested experimentally but is not part of the current
   baseline.

4. **Shape latent normalization**

   Training normalizes raw `shape_feats` with `pipeline.shape_slat_normalization`.
   Debug scripts are provided to check whether a preprocessed `.pt` matches the
   expected normalization distribution.

5. **1024 experiments are heavier**

   1024 sparse coords have many more latent tokens than 512. Adapter capacity,
   IP scale, and optimization become more sensitive.

## Quick Start

Run commands from the TRELLIS.2 repository root.

### 1. Convert THuman Meshes to VXZ

The conversion script lives at the TRELLIS.2 root in the current setup.

```bash
python convert_thuman_to_vxz.py \
  --thuman_dir ~/TRELLIS/model \
  --resolution 1024 \
  --num_samples 1 \
  --out_dir data/thuman_vxz
```

This creates:

```text
data/thuman_vxz/dual_grid_1024/0000.vxz
```

### 2. Preprocess Training Data

```bash
CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/data/preprocess_thuman.py \
  --thuman_dir ~/TRELLIS/model \
  --vxz_dir data/thuman_vxz/dual_grid_1024 \
  --out_dir data/adapter_train_hq_1024 \
  --resolution 1024 \
  --save_debug_images \
  --num_samples 1 \
  --hq_condition \
  --overwrite
```

This saves:

```text
data/adapter_train_hq_1024/0000.pt
data/adapter_train_hq_1024/debug_images/0000_fullbody_hq.png
data/adapter_train_hq_1024/debug_images/0000_masked_hq.png
data/adapter_train_hq_1024/debug_images/0000_head_hq.png
```

### 3. Train Shape Adapter

```bash
CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/scripts/train.py \
  --config facemesh_adapter_v2/configs/overfit_1sample.yaml \
  --model.flow_model_key shape_slat_flow_model_1024 \
  --data.train_dir data/adapter_train_hq_1024 \
  --output.dir runs/overfit_test_1024
```

For clean one-sample overfit debugging, use:

```yaml
train:
  cfg_drop_cond: 0.0
  cfg_drop_id: 0.0
  cfg_drop_both: 0.0
```

### 4. Inference With GT Coords/Conditions

```bash
CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/scripts/infer.py \
  --config facemesh_adapter_v2/configs/overfit_1sample.yaml \
  --adapter_ckpt runs/overfit_test_1024/shape_xxxxxx/adapter_final.pt \
  --gt_pt data/adapter_train_hq_1024/0000.pt \
  --use_gt_cond \
  --debug_latent_error \
  --debug_mesh_error \
  --mesh_error_samples 20000 \
  --mesh_unit_scale_cm 170 \
  --pipeline_type 1024 \
  --shape_steps 50 \
  --out_dir results/overfit_1024_debug
```

`--use_gt_cond` uses stored `cond`, `neg_cond`, and `id_emb` from the `.pt` file.
`--gt_pt` also provides GT sparse coords when debugging overfit behavior.

### 5. End-to-End 1024 Overfit Script

```bash
bash facemesh_adapter_v2/scripts/run_overfit_1024.sh
```

Common overrides:

```bash
GPU=2 \
THUMAN_DIR=~/TRELLIS/model \
SAMPLE_ID=0000 \
bash facemesh_adapter_v2/scripts/run_overfit_1024.sh
```

If the VXZ converter is not at the TRELLIS.2 root:

```bash
CONVERT_SCRIPT=/path/to/convert_thuman_to_vxz.py \
bash facemesh_adapter_v2/scripts/run_overfit_1024.sh
```

## Debugging Utilities

### Check Shape Normalization

```bash
CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/scripts/check_shape_norm.py \
  --pt data/adapter_train_hq_1024/0000.pt \
  --pipeline_model microsoft/TRELLIS.2-4B \
  --out_json results/check_shape_norm_1024.json
```

A healthy `.pt` should have normalized feature statistics roughly near:

```text
mean ~= 0
std  ~= 1
```

### Check Adapter/Base Forward Parity

```bash
CUDA_VISIBLE_DEVICES=2 python facemesh_adapter_v2/scripts/check_adapter_parity.py \
  --config facemesh_adapter_v2/configs/overfit_1sample.yaml \
  --pt data/adapter_train_hq_1024/0000.pt \
  --flow_model_key shape_slat_flow_model_1024 \
  --out_json results/check_adapter_parity_1024.json
```

This compares base model output to adapter-wrapper output with `ip_scale=0`. The
expected result is near-zero error. A large error means the adapter wrapper does
not faithfully reproduce the base flow model forward pass.

### Latent Error Report

`infer.py --debug_latent_error` writes:

```text
latent_error_seed42.json
```

Important fields:

- `raw_gt_space.mse`: main latent-space metric when sampled feats are in raw GT space.
- `raw_gt_space.topk`: worst sparse tokens and channels.
- `raw_gt_space.spatial_error_summary`: coarse spatial distribution of token error.

The `normalized_gt_space` block is only useful when both prediction and target
are compared in the same normalized space. In current raw-output debugging,
`raw_gt_space` is the primary metric.

### Mesh Error Report

`infer.py --debug_mesh_error` decodes the GT latent and compares sampled surface
points between predicted and GT-reconstruction meshes. It writes:

```text
mesh_error_seed42.json
```

Metrics include full mesh, top-Y head region, and approximate central-face region
nearest-point distances.

## Project Structure

```text
facemesh_adapter_v2/
  configs/
    default.yaml
    overfit_1sample.yaml

  data/
    dataset.py
    preprocess_thuman.py

  libs/
    adapters/
      base.py
      cross_attention.py
      ip_adapter_slat.py
      ip_adapter_ss.py
    encoders/
      projections.py
    utils/
      head_extract.py
      sampling.py

  scripts/
    train.py
    infer.py
    validate.py
    run_overfit_1024.sh
    check_shape_norm.py
    check_adapter_parity.py

  trainer/
    base_trainer.py
    shape_trainer.py
    texture_trainer.py
```

## Git Notes

You can submit only selected files to GitHub. Check changes first:

```bash
git status
```

Stage only the files you want:

```bash
git add facemesh_adapter_v2/README.md
git add facemesh_adapter_v2/scripts/train.py
git add facemesh_adapter_v2/scripts/infer.py
```

Or interactively stage parts of a file:

```bash
git add -p facemesh_adapter_v2/scripts/infer.py
```

Commit only staged files:

```bash
git commit -m "Update FaceMesh adapter training and debug tools"
```

Verify the staged diff before committing:

```bash
git diff --cached
```
