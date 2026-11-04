#!/bin/sh
# sequential GPU chain: zero-cost diagnostic of the old adapter on stretch renders, then 4 training ablations
cd <PROJECT_ROOT>/src/cascade
PY=python
export HF_HOME='<HF_HOME>'
export PYTHONIOENCODING=utf-8
LOG=<PROJECT_ROOT>/results/lora_v2
mkdir -p $LOG
$PY train_lora_v2.py --tag E0_oldadapter_stretch --render stretch --label argmax --targets llm_vismlp --epochs 0 --resume '<PROJECT_ROOT>\results\lora_spatial\ep2' > $LOG/E0.log 2>&1
$PY train_lora_v2.py --tag A_stretch_smooth --render stretch --label smooth --targets llm_vismlp --n 3000 --epochs 1 > $LOG/A.log 2>&1
$PY train_lora_v2.py --tag B_fixed_smooth   --render fixed   --label smooth --targets llm_vismlp --n 3000 --epochs 1 > $LOG/B.log 2>&1
$PY train_lora_v2.py --tag C_stretch_fullvis --render stretch --label smooth --targets full_vision --n 3000 --epochs 1 --grad_ckpt > $LOG/C.log 2>&1
$PY train_lora_v2.py --tag D_stretch_coord  --render stretch --label smooth --targets llm_vismlp --task coord --n 3000 --epochs 1 > $LOG/D.log 2>&1
echo CHAIN_DONE > $LOG/CHAIN_DONE
