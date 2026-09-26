"""Emit a k8s Job spec for hp_family generation or evaluation.
usage: mkjob.py <name> gen  <arm> <rounds> <root>
       mkjob.py <name> eval <prefix> <only_epoch> <root>
Each run gets its own AB_GPT_NNGPT_DIR so arms never clobber each other."""
import json, sys
name, mode, a, b, root = sys.argv[1:6]
gpus = int(sys.argv[6]) if len(sys.argv) > 6 else 1
# Shadow ab.nn from the mounted local checkout (/a/nnd), not the image: the
# image ships a pre-train_stat ab.nn that queries a `loader` table this DB does
# not have. stat/ (3.1G of JSONs) is excluded; the package dirs must be writable
# because nn-dataset materialises metric/nn/transform code into them at runtime.
SHADOW = (f"rm -rf {root}/abnn && mkdir -p {root}/abnn/ab && "
          f"tar -C /a/nnd -cf - ab/nn --exclude='ab/nn/stat' | tar -C {root}/abnn -xf - && "
          f"mkdir -p {root}/abnn/ab/nn/stat {root}/abnn/ab/nn/metric "
          f"{root}/abnn/ab/nn/transform && export PYTHONPATH={root}/abnn && ")
if mode == 'gen':
    cmd = f"python3 -u -m ab.gpt.act.alter.hp_family --arm {a} --rounds {b}"
elif mode == 'both':
    # a = "<arm>:<prefix>", b = rounds. Generate every round, then evaluate all
    # of them (Eval with no --only_epoch walks the whole epoch root).
    arm, prefix = a.split(':')
    # Generation is single-model: pin it to one GPU so device_map='auto' does not
    # shard the 7B across all of them. Eval then uses every visible GPU.
    gen_pin = "CUDA_VISIBLE_DEVICES=0 " if gpus > 1 else ""
    cmd = (f"{gen_pin}python3 -u -m ab.gpt.act.alter.hp_family --arm {arm} --rounds {b} && "
           f"python3 -u -m ab.gpt.act.eval.Eval --nn_train_epochs 3 --nn_name_prefix {prefix}")
elif mode == 'evalall':
    # every round under the root; already-evaluated models are skipped
    cmd = f"python3 -u -m ab.gpt.act.eval.Eval --nn_train_epochs 3 --nn_name_prefix {a}"
else:
    cmd = f"python3 -u -m ab.gpt.act.eval.Eval --nn_train_epochs 3 --only_epoch {b} --nn_name_prefix {a}"
# nn-dataset materialises metric/transform/loader code into a cwd-relative
# ab/nn/, which then shadows the real ab.nn package for any later analysis run
# from the repo root. Remove it before starting and on every exit path.
args = (f"cd /a/mm && rm -rf /a/mm/ab/nn && trap 'rm -rf /a/mm/ab/nn' EXIT && "
        f"mkdir -p {root} /a/mm/.hf && "
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && " + SHADOW +
        f"{cmd} 2>&1 | tee -a {root}/{name}.log")
json.dump({
 "apiVersion":"batch/v1","kind":"Job","metadata":{"name":name},
 "spec":{"parallelism":1,"backoffLimit":0,"ttlSecondsAfterFinished":86400,
  "template":{"spec":{"restartPolicy":"Never","containers":[{
    "name":"ml","image":"ws.ab/shared/ai-linux-nngpt:v1","imagePullPolicy":"IfNotPresent",
    "command":["/bin/bash","-c"],"args":[args],
    "env":[{"name":"HF_HOME","value":"/a/mm/.hf"},{"name":"PYTHONUNBUFFERED","value":"1"},
           {"name":"AB_GPT_NNGPT_DIR","value":root},
           {"name":"NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS","value":"1" if gpus > 1 else "0"}],
    "securityContext":{"allowPrivilegeEscalation":False,"runAsUser":1062,"runAsGroup":1376},
    "volumeMounts":[{"mountPath":"/dev/shm","name":"shm"},{"mountPath":"/a/mm","name":"v0"},
                    {"mountPath":"/a/mm/data","name":"v1"},
                    {"mountPath":"/a/nnd","name":"v2","readOnly":True}],
    "resources":{"requests":{"nvidia.com/gpu":gpus,"cpu":8,"memory":"24Gi" if gpus==1 else "32Gi"},
                 "limits":{"nvidia.com/gpu":gpus,"cpu":32,"memory":"32Gi" if gpus==1 else "64Gi"}}}],
   "volumes":[{"name":"shm","emptyDir":{"medium":"Memory","sizeLimit":"16Gi"}},
     {"name":"v0","hostPath":{"path":"/shared/ssd/home/b-a-baiju/nngpt/nn-gpt","type":"DirectoryOrCreate"}},
     {"name":"v1","hostPath":{"path":"/shared/local/data/a-group-automl/data","type":"DirectoryOrCreate"}},
     {"name":"v2","hostPath":{"path":"/shared/ssd/home/b-a-baiju/lmur/nn-dataset","type":"Directory"}}]}}}},
 open(f"{name}.json","w"), indent=2)
print(f"{name}.json  [{mode}] {a} -> {root}")
