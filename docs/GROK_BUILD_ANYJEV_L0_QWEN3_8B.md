# Grok Build — CSE Cluster Prompt (BTP / AnyJev L0 on Qwen3-8B)

**Read this file end-to-end before doing anything. Obey it strictly. Do not invent methods, prompts, formulas, or APIs.**

If something is missing from this file or from disk, write `~/BTP/job_results/BLOCKED.md` with the exact gap and stop. Do **not** guess.

---

## 0. Goal (this run only)

Run **official AnyJev `Decider` at L0** (training-free) on **`Qwen/Qwen3-8B`**, same three frozen test sets as the completed Qwen3-4B L0 run.

| Item | Value |
|------|--------|
| Hub id | `Qwen/Qwen3-8B` |
| Local dir | `~/BTP/models/Qwen3-8B` (~16.4 GB safetensors) |
| Model slug (outputs) | **`anyjev_l0_qwen3_8b`** |
| Partition | **`gpu-A100` only** (`#SBATCH --gres=gpu:1`) |
| Do **not** use | `gpu-H100` (btech QoS rejects it), `gpu-P100` |

**Already done — reuse, do not redo unless corrupt:**

- `~/BTP/AnyJev` (commit used for job 5473 was `35311c5` — record whatever `git rev-parse HEAD` shows now)
- `~/containers/vllm-openai.sif` (v0.29.0 used before — record tag/version)
- Adapter / cluster scripts under `~/BTP/cluster/` from the Qwen3-4B L0 run (prefer edit + new submit script over rewriting from scratch)
- Datasets: JB org test 335, OD full 2352, FD test 255
- Existing raw files **`anyjev_l0__*.jsonl`** (Qwen3-4B) — **never overwrite or delete**

**Outputs to write (new slug only):**

```text
~/BTP/Decision_Models/eval/outputs/raw/anyjev_l0_qwen3_8b__jevbench.jsonl
~/BTP/Decision_Models/eval/outputs/raw/anyjev_l0_qwen3_8b__opendecisions.jsonl
~/BTP/Decision_Models/eval/outputs/raw/anyjev_l0_qwen3_8b__fastdecisions.jsonl
```

Schema must match existing `anyjev_l0__*` / `jev__*` / `luna__*` rows (`item_id`, `questions` with `choice`/`probs`/`confidence`, `chosen`, `gold`, `correct`, `error`, … via `eval.records.build_row` or the same adapter path used for Qwen3-4B).

---

## 1. Never / Always

**Never:**

- Invent AnyJev L0 math, prompts, or backends — call the library
- Use `gpu-P100` or `gpu-H100` for this account
- Run heavy GPU work on the login node
- `pip install --break-system-packages` or `sudo`
- Delete/modify `~/BTP/Datasets/**` or existing raw `jev__*`, `luna__*`, `anyjev_l0__*`, `semif__*`, `laya__*`
- Call a hand-rolled letter readout “AnyJev L0”
- Expand to Tacit, SemIf, OpenJev, Laya FT, or metrics Excel in this job

**Always:**

- Check existence before install/download; reuse if present
- `cd "$TMPDIR" || exit 1` at job start; write under `$TMPDIR/results/`
- Stage copies under `~/BTP/job_results/anyjev_l0_qwen3_8b_<JOBID>/`
- Smoke (`--limit 2` or env `SMOKE=1`) before full
- Resume by `item_id` if a full job is interrupted
- Log setup to `~/BTP/job_results/_setup_log_anyjev_l0_qwen3_8b.md`

---

## 2. Authoritative L0 sources (use these only)

1. `https://github.com/nokia-applied-research/AnyJev` (already cloned at `~/BTP/AnyJev`)
2. Read in the clone: `README.md`, backend modules (`VLLMBackend` / `HFBackend` — **verify import paths in code**), `docs/` as needed
3. Prefer the **same Decider + VLLMBackend path** that succeeded for Qwen3-4B (job 5473). Point the backend at `~/BTP/models/Qwen3-8B` instead of `Qwen3-4B`.

L0 summary (implementation = library): typed choice from next-token distribution over option labels; cyclic rotations + label-prior correction; **no generation**, **no training**.

If API differs from memory, follow the installed package and record versions in `SUMMARY.md`.

---

## 3. Disk / download (login node only)

Quota soft ~50GB / hard ~200GB. Qwen3-8B is ~16.4 GB.

```bash
hostname; whoami; date
df -h ~
du -sh ~/BTP ~/BTP/models ~/containers ~/.apptainer 2>/dev/null
ls -la ~/BTP/models/Qwen3-4B | head
ls -la ~/BTP/models/Qwen3-8B 2>/dev/null | head || echo "Qwen3-8B MISSING"
test -d ~/BTP/AnyJev && (cd ~/BTP/AnyJev && git rev-parse HEAD)
ls ~/containers/*.sif
```

If free space cannot hold ~16.4 GB plus job scratch copies, write `BLOCKED.md` with `df -h ~` and stop.

Download only if missing/incomplete (use existing Hub venv, not system pip):

```bash
# source ~/BTP/venv-hf or ~/hf-venv
hf download Qwen/Qwen3-8B --local-dir ~/BTP/models/Qwen3-8B
# verify all safetensor shards + model.safetensors.index.json present; record revision in SUMMARY.md
```

Do **not** re-download Qwen3-4B or Qwen3.5-4B.

---

## 4. Adapter / submit scripts

Reuse the working Qwen3-4B AnyJev cluster driver under `~/BTP/cluster/` (whatever submitted jobs 5469/5473). Changes required:

1. Model path → `~/BTP/models/Qwen3-8B`
2. Output slug → `anyjev_l0_qwen3_8b`
3. New submit script: `~/BTP/cluster/submit_anyjev_l0_qwen3_8b_a100.sh`
4. Job name e.g. `anyjev_q38b`
5. `#SBATCH --partition=gpu-A100` and `#SBATCH --gres=gpu:1` only
6. Time: start with `--time=12:00:00` (8B is slower than 4B; 5473 was ~10 min on 4B — 8B may need several×; raise if smoke timing suggests it)
7. Mem: keep ≥64G

```bash
#!/bin/bash
#SBATCH --job-name=anyjev_q38b
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/userhome/btech/p.yaswant/BTP/job_results/%x_%j.out
#SBATCH --error=/userhome/btech/p.yaswant/BTP/job_results/%x_%j.err
```

Bind/copy: SIF, `Qwen3-8B`, `AnyJev`, `Decision_Models`, `Datasets`. Prefer `apptainer exec --nv` with the same pattern as 5473.

**Multi-question Fast Decisions:** same handling as Qwen3-4B run. Items with >26 options may error (letter readout limit) — keep the row with `error` set; do not drop from JSONL. Expect ~14 `support_intent_*` failures (same as Qwen3-4B).

**U+2028 / JSON:** if you write intermediate JSONL for tools, prefer `ensure_ascii=True` or line-iterator reads (not `read_text().splitlines()`) so Unicode line separators cannot break parsing. The BTP raw writer should match the existing successful schema.

---

## 5. Phased submit

1. **Smoke:** `--limit 2` (or `SMOKE=1`) on all three datasets. Wait for job to finish. Validate JSONL parses; fields `item_id`, `chosen`/`questions`, `probs` present on non-error rows.
2. **Full:** only after smoke OK. All 335 + 2352 + 255. Resumable by `item_id`.
3. On success, copy jsonl → `~/BTP/Decision_Models/eval/outputs/raw/` and write `SUMMARY.md` under `~/BTP/job_results/anyjev_l0_qwen3_8b_<JOBID>/` with: job ids, partition, node, elapsed, exit code, SIF/vLLM version, AnyJev commit, Hub revision of Qwen3-8B, line counts, error counts per dataset.

Monitor:

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode,NodeList
```

Max 2 submitted / 1 running. Pending on busy A100 is OK.

---

## 6. Done criteria

1. `~/BTP/models/Qwen3-8B` complete; revision recorded
2. Smoke exit 0 with sample `anyjev_l0_qwen3_8b__*.jsonl`
3. Full job exit 0 (or still running/pending — report status)
4. Line counts: JB 335, OD 2352, FD 255 (including error rows)
5. Existing `anyjev_l0__*` (4B) untouched
6. `SUMMARY.md` + setup log written

If blocked: only `BLOCKED.md` + exact next human action.

---

## 7. Start now

1. Checklist + disk (§3) → setup log
2. Download Qwen3-8B if needed
3. Point existing AnyJev L0 adapter at new model path + slug
4. Submit A100 smoke → verify → full

**End of prompt.**
