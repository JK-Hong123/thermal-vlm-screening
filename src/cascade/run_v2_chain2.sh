#!/bin/sh
# waits for the first chain, then runs the resolution experiment(s)
cd <PROJECT_ROOT>/src/cascade
PY=python
export HF_HOME='<HF_HOME>'
export PYTHONIOENCODING=utf-8
LOG=<PROJECT_ROOT>/results/lora_v2
until [ -f $LOG/CHAIN_DONE ]; do sleep 60; done
until [ -f <PROJECT_ROOT>/results/lora_v2/V3_READY ]; do sleep 60; done
$PY train_lora_v2.py --tag E_v3_672_tight --render stretch --label smooth --targets llm_vismlp --n 3000 --epochs 1 --img_key img_v3_672 --lock_img thermal_crop_v3_672.png --grad_ckpt > $LOG/E.log 2>&1
echo CHAIN2_DONE > $LOG/CHAIN2_DONE
