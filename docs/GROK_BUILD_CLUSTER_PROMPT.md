# Grok Build — CSE Cluster Resume Prompt (BTP / AnyJev L0)

**Read this file end-to-end before doing anything. Obey it strictly. Do not invent methods, prompts, formulas, or APIs.**

If something is missing from this file or from disk, write `~/BTP/job_results/BLOCKED.md` with the exact gap and stop. Do **not** guess.

---

## 0. Plain-English picture (for Yaswant and for you)

| Piece | What it is | Do you “activate venv”? |
|-------|------------|-------------------------|
| `~/BTP/venv-hf` or `~/hf-venv` | Tiny Python env used **only** on the login node for `hf download` / `hf version` | Only when downloading models from Hugging Face on login. **Not** for GPU inference. |
| `~/BTP/AnyJev/` (git clone) | **Yes — required.** Official library source: [nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev). This is how L0 is defined. | Clone on login. Install the package **into the Apptainer image / job environment**, not into system Python. |
| `~/containers/*.sif` | Apptainer image with CUDA + **vLLM** (+ `anyjev` installed inside or bind-mounted editable) | N/A — use `apptainer exec --nv …` inside SLURM. |
| `~/BTP/models/Qwen3-4B` | Local weights (already downloaded) | N/A |
| `~/BTP/Datasets` + `~/BTP/Decision_Models` | Frozen eval sets + our eval harness | N/A |

**Yaswant does not need to activate `venv-hf` for the inference job.** That venv is only for Hub downloads on login.

**You (Grok Build) must have the AnyJev code on the server** (`git clone` into `~/BTP/AnyJev`) and **import** it (`from anyjev import Decider, Question`). Do **not** reimplement L0.

---

## 1. Role and stop rules

You are on **clusterlogin** as `p.yaswant`. Prepare a bug-free **SLURM + Apptainer + vLLM + official AnyJev Decider(L0)** pipeline, then submit jobs.

**Never:**
- Invent AnyJev L0 math, prompts, or backends
- Use `gpu-P100` (vLLM unsupported)
- Run heavy GPU work on the login node
- Use `pip install --break-system-packages` or `sudo`
- Delete/modify `~/BTP/Datasets/**` or existing `eval/outputs/raw/jev__*`, `luna__*`, `laya__opendecisions.jsonl`, or `checkpoints/laya_data/**`
- Call a hand-rolled readout “AnyJev L0”
- Expand scope to Laya FT, Tacit, SemIf, OpenJev, or full metrics unless this file says so

**Always:**
- Check whether a tool/lib/file exists **before** installing/downloading; reuse if present
- Prefer `gpu-A100` (`#SBATCH --partition=gpu-A100`, `#SBATCH --gres=gpu:1`) — either A100 node/GPU is fine
- Start every job body with `cd "$TMPDIR" || exit 1` and `mkdir -p results`
- Stage human-facing artifacts under `~/BTP/job_results/<run>_<JOBID>/`
- Smoke (`--limit 2`) before full run

---

## 2. Authoritative sources for AnyJev L0 (use these only)

1. Git repo: `https://github.com/nokia-applied-research/AnyJev`
2. After clone, read in order:
   - `~/BTP/AnyJev/README.md`
   - `~/BTP/AnyJev/docs/method_v3.md` (if present)
   - Backend modules under `~/BTP/AnyJev/` for the real import paths (`HFBackend`, `VLLMBackend` or whatever the repo actually exports — **verify with code**, do not assume names from memory)
3. Optional paper (reference only): https://arxiv.org/html/2610.00831

**L0 meaning (summary only — implementation = library call):**
training-free typed decision from an open causal LM: restrict next-token distribution to option labels; average over cyclic option rotations; correct with a label prior from unlabelled inputs; **no text generation**, **no training**.

If the installed package API differs from snippets below, **follow the installed package** and note the exact versions/imports in `SUMMARY.md`.

---

## 3. Fixed paths on this cluster

```text
~/BTP/
  Datasets/
    JevBench/org/test/test.jsonl          # 335  ← infer
    JevBench/org/train/train.jsonl        # FT later only
    JevBench/org/validation/validation.jsonl
    OpenDecisions/raw/opendecisions.jsonl # 2352 ← infer all
    FastDecisions/test/test.jsonl         # 255  ← infer
    FastDecisions/train|validation/...
  Decision_Models/
    eval/datasets.py   # BTP = parents[2]; PATHS point at Datasets above
    eval/models.py     # Jev/Luna HTTP plugins today
    eval/run.py        # writes eval/outputs/raw/{model}__{dataset}.jsonl
    eval/records.py    # build_row schema — match this
    eval/outputs/raw/  # jev__*, luna__*, laya__opendecisions.jsonl already here
    checkpoints/laya_data/  # keep; not used for AnyJev L0
    checkpoints/laya-base/  # optional; not required for this job
  models/Qwen3-4B/     # already complete — do not re-download unless corrupt
  AnyJev/              # CREATE via git clone if missing
  cluster/             # CREATE — SLURM scripts live here
  job_results/         # CREATE — logs + SUMMARY per run
~/containers/          # CREATE — store vllm*.sif here
~/BTP/venv-hf or ~/hf-venv  # Hub CLI only
```

Verify datasets once:

```bash
wc -l ~/BTP/Datasets/JevBench/org/test/test.jsonl \
      ~/BTP/Datasets/OpenDecisions/raw/opendecisions.jsonl \
      ~/BTP/Datasets/FastDecisions/test/test.jsonl
# must be 335, 2352, 255
```

---

## 4. Check-before-install checklist (run first; log to `~/BTP/job_results/_setup_log.md`)

```bash
hostname; whoami; date
df -h ~
du -sh ~/BTP ~/BTP/models ~/containers ~/.apptainer 2>/dev/null
command -v sbatch sinfo apptainer git python3
sinfo -o "%P %A %G %a"
sacctmgr show user $USER withassoc
ls -la ~/BTP/models/Qwen3-4B | head
test -d ~/BTP/AnyJev && echo "AnyJev present" || echo "AnyJev MISSING"
ls ~/containers/*.sif 2>/dev/null || echo "no sif"
```

Quota: soft ~50GB / hard ~200GB. If free space is insufficient for a ~10GB SIF, stop and report.

---

## 5. Exact setup steps (in order)

### 5.1 Directories

```bash
mkdir -p ~/BTP/job_results ~/BTP/cluster ~/containers
```

### 5.2 Clone AnyJev (required on server)

```bash
if [ ! -d ~/BTP/AnyJev/.git ]; then
  git clone https://github.com/nokia-applied-research/AnyJev.git ~/BTP/AnyJev
fi
cd ~/BTP/AnyJev && git rev-parse --short HEAD && git status -sb
```

Record the commit hash in `SUMMARY.md`.

### 5.3 Hub venv (only if you need `hf`)

Do **not** use this venv to run vLLM/AnyJev inference.

```bash
# reuse whichever exists
#   ~/BTP/venv-hf  OR  ~/hf-venv
# only if hf missing:
#   python3 -m venv ~/BTP/venv-hf && source ~/BTP/venv-hf/bin/activate
#   pip install -U "huggingface_hub[cli]"
```

Qwen3-4B is already at `~/BTP/models/Qwen3-4B`. Re-download only if shards/index are incomplete.

### 5.4 vLLM Apptainer image (required)

No `.sif` is present (old ones were deleted). Obtain a **pinned** image:

```bash
# Prefer a version tag, not floating latest. After checking Docker Hub / vLLM docs
# for a current stable x86_64 tag that includes CUDA for A100, e.g.:
apptainer build ~/containers/vllm-openai.sif docker://vllm/vllm-openai:v0.11.0
```

If that tag fails, try another **pinned** `vllm/vllm-openai:vX.Y.Z` (or `…-cu129` if drivers require it). Record the exact tag.

If login cannot pull/build, write `BLOCKED.md` with the error and the laptop command:

```bash
# on a machine with Apptainer/Docker + network, then:
scp vllm-openai.sif p.yaswant@172.16.112.202:~/containers/
```

Install **anyjev into the environment that will run inside the container**, e.g. one of:

- `apptainer exec` + `pip install -e /btp/AnyJev` into a writable overlay / user site you bind, **or**
- bake into a custom `.def` derived from the vLLM image (only if pull-only image cannot import anyjev)

**Verify imports inside a short GPU job** (not on login CPU-only assumption for CUDA):

```text
import anyjev, vllm, torch
print(anyjev.__version__ if hasattr(anyjev,'__version__') else 'anyjev-ok', vllm.__version__, torch.cuda.is_available())
```

### 5.5 Wire eval plugin (minimal; no API invention)

Use existing harness patterns in `~/BTP/Decision_Models/eval/`:

- Load items via `eval.datasets.load_items` / PATHS
- For each `Item`, map each shared `Question` to AnyJev’s `Question.choice` / yes-no API **as documented in the cloned repo**
- Call `Decider(backend, level="L0")` (or repo default if it is L0)
- Convert `distribution` → `Answer` / `build_row`
- Model slug: **`anyjev_l0`**
- Output files:
  - `~/BTP/Decision_Models/eval/outputs/raw/anyjev_l0__jevbench.jsonl`
  - `~/BTP/Decision_Models/eval/outputs/raw/anyjev_l0__opendecisions.jsonl`
  - `~/BTP/Decision_Models/eval/outputs/raw/anyjev_l0__fastdecisions.jsonl`

Backend preference (check real class names in the clone):

1. Official **vLLM** backend pointed at local weights `~/BTP/models/Qwen3-4B` (or serve vLLM and attach)
2. Else `HFBackend` on the same path **only if** vLLM backend cannot be made to work — still on A100 via Apptainer

**Multi-question items (esp. Fast Decisions):** follow AnyJev’s API for multiple questions in one `decide(...)` call if the library supports it; otherwise one `decide` per question head. Document which you used. Do not invent a third scheme.

**Prompt templates:** use **only** what AnyJev’s Decider/backends render. Do not hand-write a Qwen chat template for L0 unless the library requires passing raw strings and the README shows how.

---

## 6. SLURM script requirements

Create `~/BTP/cluster/submit_anyjev_l0_a100.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=anyjev_l0
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/userhome/btech/p.yaswant/BTP/job_results/%x_%j.out
#SBATCH --error=/userhome/btech/p.yaswant/BTP/job_results/%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
mkdir -p results

# copy or bind: SIF, Qwen3-4B, Decision_Models, Datasets, AnyJev
# apptainer exec --nv ... run smoke or full
# on success: copy jsonl → ~/BTP/Decision_Models/eval/outputs/raw/
#             copy logs  → ~/BTP/job_results/anyjev_l0_${SLURM_JOB_ID}/
```

Fallback partition **only** if `sbatch` rejects A100 association: `gpu-H100`. Never P100.

### Phased submit

1. **Smoke:** env `SMOKE=1` / `--limit 2` on all three datasets (or one dataset first). Wait until job leaves queue. Validate JSONL lines parse and have `item_id`, `probs`, `chosen`.
2. **Full:** only after smoke succeeds. Full = all items on all three datasets. Resumable by `item_id` if possible (mirror `eval.run`).

Monitor:

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

---

## 7. Cluster hard rules (manuals)

- Login: light work only
- Job body: `cd "$TMPDIR" || exit 1`; prefer writing under `$TMPDIR/results/`
- Also keep `~/BTP/job_results/...` copies for Yaswant
- Max 2 submitted jobs; 1 running at a time
- `apptainer exec --nv`
- Internet on login only — everything needed at runtime must be on disk before `sbatch`

---

## 8. Done criteria

1. `~/BTP/AnyJev` cloned; commit recorded
2. `~/containers/vllm-openai.sif` (or pinned name) exists; tag recorded
3. `anyjev` + `vllm` import succeeds in a GPU job
4. Smoke job `CD` with sample `anyjev_l0__*.jsonl`
5. Full job submitted (pending on busy A100 is OK)
6. `~/BTP/job_results/.../SUMMARY.md` lists: job ids, partition, SIF tag, AnyJev commit, model path, import paths used, line counts, errors

If blocked: only `BLOCKED.md` + exact next human action — no hallucinated workaround presented as success.

---

## 9. What Yaswant should do (outside Grok Build)

1. Ensure this file is on the cluster: `~/BTP/GROK_BUILD_CLUSTER_PROMPT.md`
2. Point Grok Build at that file and say **resume**
3. Do **not** activate `venv-hf` unless Grok Build asks you to run `hf download` on login
4. When asked for laptop-side SIF build/scp, run the command from `BLOCKED.md` if login cannot pull

---

## 10. Start now

1. Checklist (§4) → `_setup_log.md`
2. Clone AnyJev if missing (§5.2)
3. Build/obtain vLLM SIF (§5.4)
4. Install/import `anyjev` in the job/container env; verify against README
5. Implement thin eval adapter calling `Decider` L0 only
6. Submit A100 smoke → verify → full

**End of prompt.**
