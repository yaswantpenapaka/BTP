> **Note (actual run):** this is the original build prompt. gpu-H100 was rejected (Invalid qos for btech); the SemIf job (5539) actually ran on **gpu-A100**. The script name `submit_semif_direct_h100.sh` is legacy.

# Grok Build — CSE Cluster Prompt (BTP / SemIf direct)

**Read this file end-to-end before doing anything. Obey it strictly. Do not invent methods, prompts, formulas, or APIs.**

If something is missing from this file or from disk, write `~/BTP/job_results/BLOCKED.md` with the exact gap and stop. Do **not** guess.

**Scope locked by Yaswant:** full inference on all three frozen test sets → `semif__*.jsonl`. **No smoke job.** GPU runtime = **auto-decide** (§5.4).

**Partition locked by Yaswant:** use **`gpu-H100`** (`#SBATCH --partition=gpu-H100`, `#SBATCH --gres=gpu:1`). Fall back to `gpu-A100` only if H100 is rejected. Never `gpu-P100`.

---

## 0. Plain-English picture

| Piece | What it is | Notes |
|-------|------------|--------|
| `~/BTP/SemIf/` | **Required** git clone of [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) (formerly OpenJev). Official `semif-score` CLI. | Do **not** reimplement logit readout. |
| `~/BTP/models/Qwen3.5-4B/` | Local weights for pinned Hub rev (download if missing). | ~8–9 GB. Not the same as `Qwen3-4B` used by AnyJev. |
| Apptainer `.sif` | CUDA + PyTorch env that can `import torch` / run `semif-score` | **Auto-decide:** probe existing `~/containers/vllm-openai.sif` first; if SemIf cannot install/run cleanly there, pull/build a separate CUDA Torch SIF. |
| `~/BTP/venv-hf` or `~/hf-venv` | Login-node Hub CLI only | Not for GPU scoring. |
| `~/BTP/Datasets` + `~/BTP/Decision_Models` | Frozen eval sets + harness | Adapter maps Items → SemIf JSONL → `build_row` schema. |

**Yaswant does not activate `venv-hf` for the GPU job.** That venv is only for `hf download` on login.

---

## 1. Role and stop rules

You are on **clusterlogin** as `p.yaswant`. Prepare a bug-free **SLURM + Apptainer + official SemIf `semif-score --mode direct`** pipeline on pinned **Qwen3.5-4B**, then submit the **full** H100 job (no smoke).

**Never:**
- Invent SemIf prompts, answer-token IDs, softmax rules, or a hand-rolled “direct” scorer
- Use `gpu-P100`
- Run heavy GPU work on the login node
- Use `pip install --break-system-packages` or `sudo`
- Delete/modify `~/BTP/Datasets/**` or existing raw outputs (`jev__*`, `luna__*`, `anyjev_l0__*`, `laya__*`)
- Call a custom readout “SemIf”
- Expand scope to AnyJev, Laya FT, OpenJev 27B, reranker/serial/shared modes, calibration, or full metrics unless this file says so
- Submit a smoke/`--limit` job (scope is full-only)

**Always:**
- Check whether a tool/lib/file exists **before** installing/downloading; reuse if present
- Prefer `gpu-H100` (`#SBATCH --partition=gpu-H100`, `#SBATCH --gres=gpu:1`) — Yaswant confirmed H100 is allowed for this account; A100 is fallback only
- Start every job body with `cd "$TMPDIR" || exit 1` and `mkdir -p results`
- Stage human-facing artifacts under `~/BTP/job_results/<run>_<JOBID>/`
- Follow the **cloned** SemIf README / `docs/REPRODUCE.md` / `docs/METHOD.md` if anything here conflicts with the installed package

---

## 2. Authoritative sources for SemIf (use these only)

1. Git repo: `https://github.com/TheoLeeCJ/SemIf`
2. After clone, read in order:
   - `~/BTP/SemIf/README.md`
   - `~/BTP/SemIf/docs/REPRODUCE.md`
   - `~/BTP/SemIf/docs/METHOD.md`
   - `~/BTP/SemIf/pyproject.toml` (pinned deps; entry point `semif-score`)
   - `~/BTP/SemIf/examples/decisions.jsonl` (input schema)
3. Hub model: `Qwen/Qwen3.5-4B` at revision **`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`** (also listed in repo manifests — verify after clone)

**Method summary (implementation = library CLI, not your code):**
SemIf’s **direct** mode prompts the frozen 4B model with state + criterion + **2–16** described options, one forward pass, softmax only over fixed uppercase answer-token logits. **No answer token is generated.**

If the installed CLI flags differ from snippets below, **follow the installed package** and record exact flags/versions in `SUMMARY.md`.

---

## 3. Fixed paths on this cluster

```text
~/BTP/
  Datasets/
    JevBench/org/test/test.jsonl          # 335  ← infer
    OpenDecisions/raw/opendecisions.jsonl # 2352 ← infer all
    FastDecisions/test/test.jsonl         # 255  ← infer
  Decision_Models/
    eval/datasets.py   # load_items / Item / Question
    eval/records.py    # build_row schema — MATCH THIS
    eval/outputs/raw/  # write semif__*.jsonl here; leave jev/luna/anyjev_l0 alone
  models/
    Qwen3-4B/          # AnyJev — do NOT reuse for SemIf
    Qwen3.5-4B/        # CREATE via hf download at pinned revision if missing
  SemIf/               # CREATE via git clone if missing
  cluster/             # CREATE — SLURM scripts
  job_results/         # CREATE — logs + SUMMARY
~/containers/          # vllm-openai.sif may already exist; may add pytorch*.sif
~/BTP/venv-hf or ~/hf-venv
```

Verify datasets once:

```bash
wc -l ~/BTP/Datasets/JevBench/org/test/test.jsonl \
      ~/BTP/Datasets/OpenDecisions/raw/opendecisions.jsonl \
      ~/BTP/Datasets/FastDecisions/test/test.jsonl
# must be 335, 2352, 255
```

---

## 4. Check-before-install checklist (run first; log to `~/BTP/job_results/_setup_log_semif.md`)

```bash
hostname; whoami; date
df -h ~
du -sh ~/BTP ~/BTP/models ~/containers ~/.apptainer 2>/dev/null
command -v sbatch sinfo apptainer git python3
sinfo -o "%P %A %G %a"
sacctmgr show user $USER withassoc
test -d ~/BTP/models/Qwen3.5-4B && ls ~/BTP/models/Qwen3.5-4B | head || echo "Qwen3.5-4B MISSING"
test -d ~/BTP/SemIf/.git && (cd ~/BTP/SemIf && git rev-parse --short HEAD) || echo "SemIf MISSING"
ls -la ~/containers/*.sif 2>/dev/null || echo "no sif"
```

Quota: soft ~50GB / hard ~200GB. Weights ~9GB + possible new SIF. If free space is insufficient, stop and report in `BLOCKED.md`.

---

## 5. Exact setup steps (in order)

### 5.1 Directories

```bash
mkdir -p ~/BTP/job_results ~/BTP/cluster ~/containers ~/BTP/models
mkdir -p ~/BTP/Decision_Models/eval/outputs/raw
```

### 5.2 Clone SemIf (required)

```bash
if [ ! -d ~/BTP/SemIf/.git ]; then
  git clone https://github.com/TheoLeeCJ/SemIf.git ~/BTP/SemIf
fi
cd ~/BTP/SemIf && git rev-parse HEAD && git status -sb
```

Record the **full** commit hash in `SUMMARY.md`. Prefer whatever `master`/`main` is at clone time unless Yaswant pins a commit later — always record it.

### 5.3 Download pinned Qwen3.5-4B (Hub venv on login only)

Reuse `~/BTP/venv-hf` or `~/hf-venv`. Create only if `hf` missing:

```bash
# python3 -m venv ~/BTP/venv-hf && source ~/BTP/venv-hf/bin/activate
# pip install -U "huggingface_hub[cli]"
```

```bash
export HF_HOME="${HF_HOME:-$HOME/BTP/hf_home}"
mkdir -p "$HF_HOME"
# If ~/BTP/models/Qwen3.5-4B already has a complete snapshot for this revision, skip.
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir ~/BTP/models/Qwen3.5-4B
```

Verify `config.json` / weight shards exist. Do **not** use `~/BTP/models/Qwen3-4B` (AnyJev).

### 5.4 GPU runtime — **auto-decide** (record choice in SUMMARY)

SemIf’s `pyproject.toml` pins **Torch + Transformers** (e.g. `torch==2.10.0`, `transformers==5.17.0` — **read the cloned file** and obey those pins). Default backend is Torch/CUDA, **not** vLLM.

**Decision procedure (do in order; log every probe):**

1. If `~/containers/vllm-openai.sif` (or similarly named vLLM SIF) exists, probe inside it (short interactive/`apptainer exec`, CPU OK for import probe):

   ```bash
   apptainer exec ~/containers/vllm-openai.sif python -c \
     "import torch,transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
   ```

2. Attempt to install SemIf **into a writable overlay / bind-mounted user site** used only for this job (never system Python), then:

   ```bash
   # conceptual — use overlay or PIP_TARGET / PYTHONPATH you create under ~/BTP/
   pip install -e ~/BTP/SemIf[test]
   which semif-score
   semif-score --help
   ```

3. **Reuse the vLLM SIF** only if: install succeeds, `semif-score --help` works, and a tiny GPU job can import `torch` with `cuda` available and run SemIf’s owned `examples/decisions.jsonl` (3 rows) to a throwaway path under `$TMPDIR` (this is a **runtime verify**, not a BTP smoke / not written as `semif__*`).

4. **Otherwise** pull/build a **separate** pinned CUDA PyTorch Apptainer image under `~/containers/` (name it clearly, e.g. `pytorch-cuda-semif.sif`). Prefer a **pinned** Docker tag compatible with H100 drivers (check what the cluster manuals / existing vLLM image used). Then install SemIf into that image’s writable overlay / bind env the same way. Record the exact Docker/SIF tag.

5. If login cannot `apptainer pull`/`build`, write `BLOCKED.md` with the error and the laptop-side build + `scp` command to `p.yaswant@172.16.112.202:~/containers/`.

6. Do **not** use llama.cpp / MLX / MPS backends on the cluster. Do **not** use `--mode serial|shared|reranker` for the BTP deliverable — **`--mode direct` only**.

### 5.5 Thin adapter: BTP Items → SemIf JSONL → eval `build_row` schema

Use `~/BTP/Decision_Models/eval/datasets.py` (`load_items`) so every model sees the same Items.

**SemIf input row** (from README / `examples/decisions.jsonl`):

```json
{
  "id": "<unique>",
  "state": "<string | object | array>",
  "question": "<criterion / instructions text>",
  "options": [{"id": "<option id>", "description": "<text>"}]
}
```

**Mapping rules:**

| BTP field | SemIf field |
|-----------|-------------|
| For each `Question` on an `Item` | one SemIf input row |
| `id` | `{item_id}::{question.name}` (stable; used to merge) |
| `state` | the Item’s `state` (dict OK — SemIf allows object/array) |
| `question` | `Question.instructions` |
| `options` | `[{"id": oid, "description": desc}, ...]` in **dataset option order** |

**Option-count rule (METHOD.md):** SemIf direct supports **2–16** options. If a question has fewer than 2 or more than 16 options, **do not** call SemIf for that question; emit an eval row/question with `choice=None`, `probs=None`, and a clear `error` like `semif_unsupported_option_count:N` (same pattern as AnyJev’s FD letter-readout errors). Count these in `SUMMARY.md`.

**Multi-question items (Fast Decisions):** one SemIf call per question head; merge by `item_id` into one eval JSONL row with nested `chosen` / `probs` / `questions` matching `eval.records.build_row`.

**After `semif-score`:** parse SemIf’s output probabilities (read actual keys from a verify run / README — typically per-option scores). Map option `id` → float. Build:

- `choice` = argmax over option ids in dataset order (or SemIf’s reported decision id if present and valid)
- `probs` = normalised dict over dataset option ids (use `normalise` from `eval.records` if you call into the harness)
- Preserve `gold` from the Item
- Model slug: **`semif`**
- `model_version`: SemIf git commit + Hub revision short hash
- `model_reported`: local model path or Hub id + revision

**Output files (required):**

```text
~/BTP/Decision_Models/eval/outputs/raw/semif__jevbench.jsonl
~/BTP/Decision_Models/eval/outputs/raw/semif__opendecisions.jsonl
~/BTP/Decision_Models/eval/outputs/raw/semif__fastdecisions.jsonl
```

Schema must match existing `jev__*` / `luna__*` / `anyjev_l0__*` rows enough for `eval.analyze` (schema 1: `item_id`, `chosen`, `probs`, `gold`, `questions`, …). Prefer calling `build_row` with a thin plugin wrapper; if you write rows by hand, mirror `records.py` fields exactly.

**CLI shape** (adjust only if `--help` differs; still `--mode direct`):

```bash
CUDA_VISIBLE_DEVICES=0 semif-score --mode direct \
  --model ~/BTP/models/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input /path/to/semif_inputs_<dataset>.jsonl \
  --output /path/to/semif_raw_<dataset>.jsonl
```

If local-dir + `--revision` is rejected by the installed CLI, follow SemIf docs: pass Hub id + revision, with `HF_HOME` / local cache pointed at the already-downloaded snapshot — **do not invent flags**. SemIf refuses an existing `--output` path; write to `$TMPDIR` then copy.

Prefer resumability: skip `item_id`s already present in the destination `semif__*.jsonl` (last row wins, same as `read_rows`).

---

## 6. SLURM script requirements

Create `~/BTP/cluster/submit_semif_direct_h100.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=semif_direct
#SBATCH --partition=gpu-H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/userhome/btech/p.yaswant/BTP/job_results/%x_%j.out
#SBATCH --error=/userhome/btech/p.yaswant/BTP/job_results/%x_%j.err

set -euo pipefail
cd "$TMPDIR" || exit 1
mkdir -p results

# bind/copy: chosen SIF, Qwen3.5-4B, SemIf install, Decision_Models, Datasets
# apptainer exec --nv … run FULL infer for jevbench, opendecisions, fastdecisions
# on success: copy semif__*.jsonl → ~/BTP/Decision_Models/eval/outputs/raw/
#             copy logs + SUMMARY → ~/BTP/job_results/semif_direct_${SLURM_JOB_ID}/
```

Fallback partition **only** if `sbatch` rejects H100 association: `gpu-A100`. Never P100.

### Submit policy (no smoke)

1. Complete setup + runtime verify on owned `examples/decisions.jsonl` only (not BTP outputs).
2. Submit **one** full job covering all three datasets (or three sequential full datasets in one script). Waiting in queue is OK.
3. Do **not** submit a BTP `--limit` / smoke job.

Monitor:

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Cluster caps: max **2** submitted jobs; **1** running at a time. Do not stack conflicting GPU jobs.

---

## 7. Cluster hard rules (manuals)

- Login: light work only (clone, hf download, write scripts)
- Job body: `cd "$TMPDIR" || exit 1`; prefer `$TMPDIR/results/`
- Also keep `~/BTP/job_results/...` copies for Yaswant
- `apptainer exec --nv`
- Internet on login only — everything needed at runtime must be on disk before `sbatch`

---

## 8. Done criteria

1. `~/BTP/SemIf` cloned; commit recorded
2. `~/BTP/models/Qwen3.5-4B` complete at revision `851bf6e…`
3. Runtime choice recorded: **reused vLLM SIF** or **new Torch SIF** (tag/path) + why auto-decide picked it
4. `semif-score --mode direct` verified on SemIf’s `examples/decisions.jsonl` inside a GPU job
5. Full job submitted (pending on busy H100 is OK) **or** completed with:
   - `semif__jevbench.jsonl` (335 items; some questions may carry option-count errors)
   - `semif__opendecisions.jsonl` (2352)
   - `semif__fastdecisions.jsonl` (255)
6. `~/BTP/job_results/semif_direct_<JOBID>/SUMMARY.md` lists: job id(s), partition, SIF path/tag, SemIf commit, model path + revision, install method, line counts per file, unsupported-option-count tallies, errors

If blocked: only `BLOCKED.md` + exact next human action — no hallucinated workaround presented as success.

---

## 9. What Yaswant should do (outside Grok Build)

1. Put this file on the cluster as `~/BTP/GROK_BUILD_SEMIF_PROMPT.md`
2. Point Grok Build at that file and say **resume**
3. Do **not** activate `venv-hf` unless asked to run `hf download` on login
4. If `BLOCKED.md` asks for a laptop SIF build/scp, run that command

---

## 10. Start now

1. Checklist (§4) → `_setup_log_semif.md`
2. Clone SemIf (§5.2)
3. Download Qwen3.5-4B at pinned revision (§5.3)
4. Auto-decide runtime (§5.4); record decision
5. Implement thin adapter (§5.5); model slug `semif`
6. Submit H100 **full** job — **no smoke**

**End of prompt.**
