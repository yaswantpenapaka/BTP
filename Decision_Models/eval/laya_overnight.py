"""Overnight Laya pipeline: OD infer (base) -> FT jevbench -> infer -> FT fastdecisions -> infer -> analyze."""
import json, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "eval" / "outputs" / "laya_overnight_status.md"
CK = ROOT / "checkpoints"
DATA = CK / "laya_data"
EPOCHS = 1  # reduced from default 4: CPU measured ~3 s/item inference
log = []

def status(step, blocker=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    log.append(f"- {now} IST: {step}")
    STATUS.write_text("# Laya overnight status\n\nUpdated: %s IST\n\nCurrent: %s\n\n## Log\n%s\n\n## Blockers\n%s\n\n"
        "## Setup\n- torch 2.14.1+cpu, laya 0.4.15, transformers in venv\n- base: checkpoints/laya-base (English root only used)\n"
        "- train data: checkpoints/laya_data/*.jsonl (gold-only questions, same option order as eval adapters)\n"
        "- FT config: laya.train.finetune defaults (RLCD, micro_batch 8, grad_accum 8, enc lr 2.5e-5, head lr 1e-4) except epochs=%d; eval_data=validation split; CPU\n"
        % (now, step, "\n".join(log), blocker or "none", EPOCHS), encoding="utf-8")

def sh(args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    if r.returncode:
        raise RuntimeError(f"{args} exit {r.returncode}")

def ft(ds):
    from laya.train import TrainConfig, finetune
    out = CK / f"laya-base-ft-{ds}"
    t0 = time.time()
    cfg = TrainConfig(epochs=EPOCHS, eval_data=str(DATA / f"{ds}_validation.jsonl"))
    summ = finetune(str(DATA / f"{ds}_train.jsonl"), str(CK / "laya-base"), str(out), cfg, device="cpu")
    wall = (time.time() - t0) / 3600
    (out / "TRAIN_README.md").write_text(
        f"# laya-base-ft-{ds}\n\nStart: convaiinnovations/laya (English base)\nTrain: {ds} train split; eval/calibration: validation split (+ laya internal calib holdout of train)\n"
        f"Device: CPU (laptop)\nEpochs: {EPOCHS}; other knobs = laya TrainConfig defaults\nWall time: {wall:.2f} h\n\n```json\n{json.dumps(summ, default=str, indent=1)[:20000]}\n```\n", encoding="utf-8")
    return wall

steps = [
    ("infer untuned base on OpenDecisions (2352)", lambda: sh(["-m", "eval.laya", "run", "--ckpt", "checkpoints/laya-base", "--tag", "base", "--datasets", "opendecisions"])),
    ("FALLBACK: untuned base on JevBench+FD test -> eval/outputs/laya_base_fallback", lambda: sh(["-m", "eval.laya", "run", "--ckpt", "checkpoints/laya-base", "--tag", "base", "--datasets", "jevbench", "fastdecisions", "--out", "eval/outputs/laya_base_fallback"])),
    ("fine-tune base on JevBench train", lambda: ft("jevbench")),
    ("infer laya-base-ft-jevbench on JevBench test", lambda: sh(["-m", "eval.laya", "run", "--ckpt", "checkpoints/laya-base-ft-jevbench", "--tag", "base-ft-jevbench", "--datasets", "jevbench"])),
    ("fine-tune base on FastDecisions train", lambda: ft("fastdecisions")),
    ("infer laya-base-ft-fastdecisions on FastDecisions test", lambda: sh(["-m", "eval.laya", "run", "--ckpt", "checkpoints/laya-base-ft-fastdecisions", "--tag", "base-ft-fastdecisions", "--datasets", "fastdecisions"])),
    ("analyze (3x3 matrices)", lambda: sh(["-m", "eval.analyze"])),
]
for name, fn in steps:
    status("STARTED " + name)
    try:
        fn()
        status("DONE " + name)
    except Exception:
        status("FAILED " + name, traceback.format_exc()[-3000:])
        continue
status("PIPELINE FINISHED")


