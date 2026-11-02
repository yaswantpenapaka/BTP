import subprocess, sys, time, os, json, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
LOG = ROOT/"eval/outputs/laya_resume2_log.md"; CK = ROOT/"checkpoints"; DATA = CK/"laya_data"
PY = str(ROOT/".venv/Scripts/python.exe"); env = dict(os.environ, PYTHONIOENCODING="utf-8")
def log(m):
    with open(LOG,"a",encoding="utf-8") as f: f.write(f"- {time.strftime('%H:%M:%S IST')} {m}\n")
def ok(d): return d.exists() and (any(d.rglob("*.safetensors")) or any(d.rglob("*.bin")) or any(d.rglob("*.pt")))
def pid_alive(p):
    r = subprocess.run(["tasklist","/FI",f"PID eq {p}"],capture_output=True,text=True); return str(p) in r.stdout
def run(args, name):
    log(f"START {name}: {' '.join(args)}")
    with open(ROOT/f"eval/outputs/resume2_{name}.log","w",encoding="utf-8") as f:
        rc = subprocess.run([PY]+args, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"END {name} rc={rc}"); return rc
FT = ("from laya.train import TrainConfig, finetune; import json; "
      "cfg=TrainConfig(epochs=1, eval_data=r'{v}'); s=finetune(r'{t}', r'{b}', r'{o}', cfg, device='cpu'); "
      "open(r'{o}\\TRAIN_README.md','w',encoding='utf-8').write('Device CPU, epochs 1\\n'+json.dumps(s,default=str,indent=1)[:20000])")
def ft(ds):
    o = CK/f"laya-base-ft-{ds}"
    code = FT.format(v=DATA/f"{ds}_validation.jsonl", t=DATA/f"{ds}_train.jsonl", b=CK/"laya-base", o=o)
    for attempt in range(2):
        run(["-c", code], f"ft_{ds}_{attempt}")
        if ok(o): return True
    return False
def infer(ds):
    return run(["-m","eval.laya","run","--ckpt",str((CK/f"laya-base-ft-{ds}").resolve()),"--tag",f"base-ft-{ds}","--datasets",ds], f"infer_{ds}")
log("resume2 started; waiting for PID 2964 (JB FT)")
while pid_alive(2964): time.sleep(60)
jb = CK/"laya-base-ft-jevbench"
log(f"PID 2964 exited; JB ckpt usable={ok(jb)}")
if not ok(jb) and not ft("jevbench"): log("FAILED JB FT")
if ok(jb): infer("jevbench")
if ft("fastdecisions"): infer("fastdecisions")
else: log("FAILED FD FT")
run(["-m","eval.analyze"],"analyze")
log("resume2 DONE")
