#!/usr/bin/env bash
set -euo pipefail

# Run from TRELLIS2.2 root:
#   bash facemesh_adapter_v2/scripts/run_overfit_1024.sh
#
# Useful overrides:
#   GPU=2 THUMAN_DIR=~/TRELLIS/model SAMPLE_ID=0000 bash facemesh_adapter_v2/scripts/run_overfit_1024.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-2}"
SAMPLE_ID="${SAMPLE_ID:-0000}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"

THUMAN_DIR="${THUMAN_DIR:-${HOME}/TRELLIS/model}"
VXZ_ROOT="${VXZ_ROOT:-data/thuman_vxz}"
VXZ_DIR="${VXZ_ROOT}/dual_grid_1024"
TRAIN_DIR="${TRAIN_DIR:-data/adapter_train_hq_1024}"
RUN_ROOT="${RUN_ROOT:-runs/overfit_test_1024}"
RESULT_ROOT="${RESULT_ROOT:-results/overfit_1024}"

CONFIG="${CONFIG:-facemesh_adapter_v2/configs/overfit_1sample.yaml}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-convert_thuman_to_vxz.py}"
PREPROCESS_SCRIPT="facemesh_adapter_v2/data/preprocess_thuman.py"
TRAIN_SCRIPT="facemesh_adapter_v2/scripts/train.py"
INFER_SCRIPT="facemesh_adapter_v2/scripts/infer.py"

SHAPE_STEPS="${SHAPE_STEPS:-50}"
MESH_ERROR_SAMPLES="${MESH_ERROR_SAMPLES:-20000}"
MESH_UNIT_SCALE_CM="${MESH_UNIT_SCALE_CM:-170}"

echo "Root: ${ROOT_DIR}"
echo "GPU: ${GPU}"
echo "Sample: ${SAMPLE_ID}"
echo "THuman dir: ${THUMAN_DIR}"
echo "VXZ dir: ${VXZ_DIR}"
echo "Train dir: ${TRAIN_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Result root: ${RESULT_ROOT}"

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
  echo "ERROR: convert script not found: ${CONVERT_SCRIPT}"
  echo "Set CONVERT_SCRIPT=/path/to/convert_thuman_to_vxz.py if needed."
  exit 1
fi

echo
echo "[1/5] Convert THuman OBJ to 1024 VXZ"
CUDA_VISIBLE_DEVICES="${GPU}" python "${CONVERT_SCRIPT}" \
  --thuman_dir "${THUMAN_DIR}" \
  --resolution 1024 \
  --num_samples "${NUM_SAMPLES}" \
  --out_dir "${VXZ_ROOT}"

echo
echo "[2/5] Preprocess 1024 latent + HQ conditions"
CUDA_VISIBLE_DEVICES="${GPU}" python "${PREPROCESS_SCRIPT}" \
  --thuman_dir "${THUMAN_DIR}" \
  --vxz_dir "${VXZ_DIR}" \
  --out_dir "${TRAIN_DIR}" \
  --resolution 1024 \
  --save_debug_images \
  --num_samples "${NUM_SAMPLES}" \
  --hq_condition \
  --overwrite

echo
echo "[3/5] Train 1024 shape adapter"
mkdir -p "${RUN_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" python "${TRAIN_SCRIPT}" \
  --config "${CONFIG}" \
  --model.flow_model_key shape_slat_flow_model_1024 \
  --data.train_dir "${TRAIN_DIR}" \
  --output.dir "${RUN_ROOT}"

echo
echo "[4/5] Find trained adapter checkpoint"
LATEST_RUN="$(find "${RUN_ROOT}" -maxdepth 1 -type d -name 'shape_*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "${LATEST_RUN}" ]]; then
  echo "ERROR: no shape_* run directory found under ${RUN_ROOT}"
  exit 1
fi

ADAPTER_CKPT="${LATEST_RUN}/adapter_final.pt"
if [[ ! -f "${ADAPTER_CKPT}" ]]; then
  echo "ERROR: adapter checkpoint not found: ${ADAPTER_CKPT}"
  exit 1
fi
echo "Using adapter: ${ADAPTER_CKPT}"

FULLBODY_IMAGE="${TRAIN_DIR}/debug_images/${SAMPLE_ID}_fullbody_hq.png"
HEAD_IMAGE="${TRAIN_DIR}/debug_images/${SAMPLE_ID}_head_hq.png"
GT_PT="${TRAIN_DIR}/${SAMPLE_ID}.pt"
OUT_DIR="${RESULT_ROOT}/$(basename "${LATEST_RUN}")"

if [[ ! -f "${FULLBODY_IMAGE}" ]]; then
  echo "ERROR: fullbody image not found: ${FULLBODY_IMAGE}"
  exit 1
fi
if [[ ! -f "${HEAD_IMAGE}" ]]; then
  echo "ERROR: head image not found: ${HEAD_IMAGE}"
  exit 1
fi
if [[ ! -f "${GT_PT}" ]]; then
  echo "ERROR: GT pt not found: ${GT_PT}"
  exit 1
fi

echo
echo "[5/5] Infer with trained 1024 adapter"
CUDA_VISIBLE_DEVICES="${GPU}" python "${INFER_SCRIPT}" \
  --config "${CONFIG}" \
  --adapter_ckpt "${ADAPTER_CKPT}" \
  --fullbody_image "${FULLBODY_IMAGE}" \
  --head_image "${HEAD_IMAGE}" \
  --out_dir "${OUT_DIR}" \
  --gt_pt "${GT_PT}" \
  --use_gt_cond \
  --debug_latent_error \
  --debug_mesh_error \
  --mesh_error_samples "${MESH_ERROR_SAMPLES}" \
  --mesh_unit_scale_cm "${MESH_UNIT_SCALE_CM}" \
  --pipeline_type 1024 \
  --shape_steps "${SHAPE_STEPS}"

echo
echo "Done."
echo "Run dir: ${LATEST_RUN}"
echo "Adapter: ${ADAPTER_CKPT}"
echo "Results: ${OUT_DIR}"
