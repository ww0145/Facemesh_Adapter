#!/bin/bash
# =============================================================
# FaceMesh-Adapter v2 - Full Pipeline Example
# =============================================================
# Run from TRELLIS.2 root: bash facemesh_adapter_v2/examples/full_pipeline.sh
# =============================================================

set -e
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com

TRELLIS2_ROOT=$(pwd)
ADAPTER_ROOT="$TRELLIS2_ROOT/facemesh_adapter_v2"
THUMAN_DIR="$HOME/TRELLIS/model"

echo "=========================================="
echo "  FaceMesh-Adapter v2 Full Pipeline"
echo "=========================================="

# Step 1: Convert THuman → O-Voxel
echo "[1/4] Converting THuman → O-Voxel..."
python convert_thuman_to_vxz.py \
    --thuman_dir "$THUMAN_DIR" \
    --resolution 512 \
    --num_samples 10 \
    --out_dir data/thuman_vxz

# Step 2: Preprocess (render + encode → .pt)
echo "[2/4] Preprocessing THuman samples..."
python "$ADAPTER_ROOT/data/preprocess_thuman.py" \
    --thuman_dir "$THUMAN_DIR" \
    --vxz_dir data/thuman_vxz/dual_grid_512 \
    --out_dir data/adapter_train \
    --num_samples 10

# Step 3: Train shape adapter (overfit test)
echo "[3/4] Training shape adapter (overfit)..."
python "$ADAPTER_ROOT/scripts/train.py" \
    --config "$ADAPTER_ROOT/configs/overfit_1sample.yaml" \
    --data.train_dir data/adapter_train

# Step 4: Inference
echo "[4/4] Running inference..."
LATEST_RUN=$(ls -td runs/shape_* 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
    python "$ADAPTER_ROOT/scripts/infer.py" \
        --config "$ADAPTER_ROOT/configs/default.yaml" \
        --adapter_ckpt "$LATEST_RUN/adapter_final.pt" \
        --head_image woman.jpg \
        --out_dir results/infer_test
else
    echo "No training run found, skipping inference"
fi

echo ""
echo "Done! Check results/ for outputs."
