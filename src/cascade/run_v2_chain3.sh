#!/bin/sh
cd <PROJECT_ROOT>/src/cascade
PY=python
export HF_HOME='<HF_HOME>'
export PYTHONIOENCODING=utf-8
LOG=<PROJECT_ROOT>/results/lora_v2
until [ -f $LOG/CHAIN2_DONE ]; do sleep 60; done
until grep -qa "jsonl updated" <PROJECT_ROOT>/results/rerender_v3_1008_log.txt; do sleep 60; done
$PY train_lora_v2.py --tag E2_v3_1008_tight --render stretch --label smooth --targets llm_vismlp --n 3000 --epochs 1 --img_key img_v3_1008 --lock_img thermal_crop_v3_1008.png --grad_ckpt --bs 1 --accum 16 > $LOG/E2.log 2>&1
echo CHAIN3_DONE > $LOG/CHAIN3_DONE
