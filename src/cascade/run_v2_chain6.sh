#!/bin/sh
cd <PROJECT_ROOT>/src/cascade
PY=python
export HF_HOME='<HF_HOME>'
export PYTHONIOENCODING=utf-8
LOG=<PROJECT_ROOT>/results/lora_v2
R=<PROJECT_ROOT>/results/eval_v3_big
mkdir -p $R
$PY eval_v3_big.py --n 300 --render_only > $R/build.log 2>&1
$PY eval_v3_big.py --adapter '<PROJECT_ROOT>\results\lora_v2\F_fullvis_coord_672_n6000\ep1' > $R/eval_F.log 2>&1
$PY eval_v3_big.py --adapter none > $R/eval_base.log 2>&1
echo EVAL_DONE > $R/EVAL_DONE
# attribution ablations at 672px / n6000 (which of full-vision vs coord drove F?)
$PY train_lora_v2.py --tag G1_fullvis_cell_672_n6000 --render stretch --label smooth --targets full_vision --task cell  --n 6000 --epochs 1 --img_key img_v3_672 --lock_img thermal_crop_v3_672.png --grad_ckpt --bs 1 --accum 16 > $LOG/G1.log 2>&1
$PY train_lora_v2.py --tag G2_llm_coord_672_n6000     --render stretch --label smooth --targets llm_vismlp  --task coord --n 6000 --epochs 1 --img_key img_v3_672 --lock_img thermal_crop_v3_672.png --grad_ckpt --bs 1 --accum 16 > $LOG/G2.log 2>&1
echo CHAIN6_DONE > $LOG/CHAIN6_DONE
