#!/bin/bash
#SBATCH --job-name=semif_direct
# H100 rejected for this account: gpu-H100 AllowQos=faculty,officer,h100,
# and p.yaswant's only QOS is btech (sbatch: Invalid qos specification,
# including an explicit --qos=h100). Fallback is gpu-A100. Never gpu-P100.
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/userhome/btech/p.yaswant/BTP/job_results/%x_%j.out
#SBATCH --error=/userhome/btech/p.yaswant/BTP/job_results/%x_%j.err

# Official SemIf direct scoring. Examples verify goes to $TMPDIR only.
# Full JevBench / OpenDecisions / FastDecisions rows go to semif__*.jsonl.
set -euo pipefail
cd "$TMPDIR" || exit 1
mkdir -p results

BTP=/userhome/btech/p.yaswant/BTP
SIF=/userhome/btech/p.yaswant/containers/vllm-openai.sif
VENV="$BTP/semif-venv"
MODEL="$BTP/models/Qwen3.5-4B"
REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
COMMIT=23cf1f39fc9534fe81437200959b6dfc7106e45a
RAW="$BTP/Decision_Models/eval/outputs/raw"
ART="$BTP/job_results/semif_direct_${SLURM_JOB_ID}"
WORK="$TMPDIR/results"

echo "=== $(date -Is) job=${SLURM_JOB_ID:-local} host=$(hostname) partition=${SLURM_JOB_PARTITION:-unknown} ==="
nvidia-smi -L || true
mkdir -p "$ART" "$RAW" "$WORK" "$TMPDIR/cache"

# One visible GPU. Do not override a SLURM-assigned id.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$TMPDIR/cache/hf"
export XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$HF_HOME"

# Pinned Torch 2.10.0+cu128 lives in the venv. The job prepends that wheel's
# CUDA 12.8 library dirs so they win over the image's CUDA 13 tree.
apptainer exec --nv \
  --bind "${BTP}:/btp:ro" \
  --bind "${TMPDIR}:${TMPDIR}" \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
  --env "HF_HUB_OFFLINE=1" \
  --env "TRANSFORMERS_OFFLINE=1" \
  --env "HF_HOME=${HF_HOME}" \
  --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
  --env "PATH=${VENV}/bin:/usr/local/bin:/usr/bin:/bin" \
  --env "PYTHONPATH=/btp/Decision_Models" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  bash -s -- "$MODEL" "$REV" "$COMMIT" "$RAW" "$WORK" "$ART" <<'EOS'
set -euo pipefail
cd "$TMPDIR" || exit 1
mkdir -p results
MODEL="$1"
REV="$2"
COMMIT="$3"
RAW="$4"
WORK="$5"
ART="$6"

echo "=== import check ==="
# Prefer the pinned wheel's CUDA 12.8 libs over the image CUDA 13 tree.
NVIDIA_LIBS="$(python - <<'PY'
import pathlib
try:
    import nvidia
except ImportError:
    raise SystemExit(0)
roots = [pathlib.Path(path) for path in getattr(nvidia, "__path__", [])]
libs = []
for root in roots:
    libs.extend(path for path in sorted(root.glob("*/lib")) if path.is_dir())
print(":".join(str(path) for path in libs))
PY
)"
if [[ -n "$NVIDIA_LIBS" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
python - <<'PY'
import torch, transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_build", torch.version.cuda)
print("cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("need exactly one visible CUDA device")
print("device", torch.cuda.get_device_name(0))
PY

echo "=== semif-score --help ==="
semif-score --help >/dev/null

echo "=== examples verify (not a BTP output) ==="
semif-score --mode direct \
  --model "$MODEL" \
  --revision "$REV" \
  --input /btp/SemIf/examples/decisions.jsonl \
  --output "$WORK/examples-direct.jsonl"
python - "$WORK/examples-direct.jsonl" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 3:
    raise SystemExit(f"examples: expected 3 rows, got {len(rows)}")
for row in rows:
    probs = row.get("probabilities")
    ids = row.get("option_ids")
    if not isinstance(probs, list) or not isinstance(ids, list) or len(probs) != len(ids) or len(ids) < 2:
        raise SystemExit(f"examples: bad row {row.get('id')}")
print("examples ok", [row["id"] for row in rows])
PY
cp "$WORK/examples-direct.jsonl" "$ART/examples-direct.jsonl"

echo "=== full direct infer ==="
python /btp/cluster/semif_direct.py \
  --datasets jevbench opendecisions fastdecisions \
  --model "$MODEL" \
  --revision "$REV" \
  --semif-commit "$COMMIT" \
  --work "$WORK" \
  --raw-dest "$RAW" \
  --semif-score semif-score \
  --summary-json "$WORK/summary.json"
cp "$WORK/summary.json" "$ART/summary.json"
shopt -s nullglob
raws=( "$WORK"/semif_raw_*.jsonl )
if ((${#raws[@]})); then
  cp "${raws[@]}" "$ART/"
fi
shopt -u nullglob
EOS

python3 - "$RAW" "$ART" "${SLURM_JOB_ID}" "${SLURM_JOB_PARTITION:-unknown}" "$SIF" "$COMMIT" "$MODEL" "$REV" <<'PY'
import json, sys
from pathlib import Path
raw = Path(sys.argv[1])
art = Path(sys.argv[2])
job, partition, sif, commit, model, rev = sys.argv[3:9]
summary = {}
summary_path = art / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
names = {
    "jevbench": 335,
    "opendecisions": 2352,
    "fastdecisions": 255,
}
lines = []
unsupported = 0
unsupported_by = {}
other_errors = 0
for dataset, expected in names.items():
    path = raw / f"semif__{dataset}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    ids = [row["item_id"] for row in rows]
    if len(rows) != expected or len(set(ids)) != expected:
        raise SystemExit(f"{path.name}: expected {expected} unique items, got {len(rows)} rows / {len(set(ids))} ids")
    bad = [row["item_id"] for row in rows if row.get("model") != "semif" or "questions" not in row]
    if bad:
        raise SystemExit(f"{path.name}: schema mismatch {bad[:3]}")
    for row in rows:
        for question in row["questions"].values():
            err = question.get("error") or ""
            if err.startswith("semif_unsupported_option_count:"):
                unsupported += 1
                key = err.split(":", 1)[1]
                unsupported_by[key] = unsupported_by.get(key, 0) + 1
            elif err:
                other_errors += 1
    lines.append(f"- {path.name}: {len(rows)}")
text = "\n".join([
    f"# semif_direct job {job}",
    "",
    f"- partition: {partition}",
    "- H100: rejected before submit (AllowQos=faculty,officer,h100; account QOS is btech only). Fallback gpu-A100.",
    "- smoke: no (full JB/OD/FD; examples verify only under $TMPDIR)",
    f"- SIF: {sif}",
    "- SIF tag: docker://vllm/vllm-openai:v0.29.0",
    "- runtime: reused vLLM SIF with a user venv (torch==2.10.0+cu128, transformers==5.17.0). Image itself shipped torch 2.13.0+cu130 and transformers 5.16.1, which do not match SemIf's pins.",
    f"- install: {Path.home()}/BTP/semif-venv  (pip install torch from the cu128 index, then pinned requirements, then pip install -e SemIf --no-deps)",
    f"- SemIf commit: {commit}",
    f"- model: {model}",
    f"- revision: {rev}",
    "- mode: semif-score --mode direct",
    "- questions: one SemIf row per question, id item_id::question.name; merged with eval.records.build_row",
    f"- unsupported option counts (not sent to SemIf): {unsupported} {unsupported_by}",
    f"- other question errors: {other_errors}",
    "- line counts:",
    *lines,
])
(art / "SUMMARY.md").write_text(text + "\n", encoding="utf-8")
print(text)
PY

echo "=== done job ${SLURM_JOB_ID} ==="
