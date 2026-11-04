#!/bin/sh
# reordered: F (best-recipe) first, then a cheaper E2 (1008px, n=1500) as the third ceiling-ratio data point
cd <PROJECT_ROOT>/src/cascade
PY=python
export HF_HOME='<HF_HOME>'
export PYTHONIOENCODING=utf-8
LOG=<PROJECT_ROOT>/results/lora_v2
$PY rerender_v3.py --size 672 --ctx 1.4 --n_train 6000 > <PROJECT_ROOT>/results/rerender_v3_672_6000_log.txt 2>&1
$PY - <<'PYX'
import json, os
D=r'<PROJECT_ROOT>\results\lora_data_v2'
for name in ('train','val'):
    rows=[json.loads(l) for l in open(os.path.join(D,f'{name}.jsonl'),encoding='utf-8')]
    for r in rows:
        if 'img_v3' in r:
            base=r['img_v3'].replace('_v3_672.png','').replace('_v3_1008.png',''); r['img_v3_672']=base+'_v3_672.png'; r['img_v3_1008']=base+'_v3_1008.png'; r.pop('img_v3',None)
    with open(os.path.join(D,f'{name}.jsonl'),'w',encoding='utf-8') as fo:
        for r in rows: fo.write(json.dumps(r,ensure_ascii=False)+'\n')
print('keys normalised')
PYX
$PY train_lora_v2.py --tag F_fullvis_coord_672_n6000 --render stretch --label smooth --targets full_vision --task coord --n 6000 --epochs 1 --img_key img_v3_672 --lock_img thermal_crop_v3_672.png --grad_ckpt --bs 1 --accum 16 > $LOG/F.log 2>&1
echo CHAIN4_DONE > $LOG/CHAIN4_DONE
rm -rf $LOG/E2_v3_1008_tight
$PY train_lora_v2.py --tag E2_v3_1008_tight --render stretch --label smooth --targets llm_vismlp --n 1500 --epochs 1 --img_key img_v3_1008 --lock_img thermal_crop_v3_1008.png --grad_ckpt --bs 1 --accum 16 > $LOG/E2.log 2>&1
echo CHAIN3_DONE > $LOG/CHAIN3_DONE
