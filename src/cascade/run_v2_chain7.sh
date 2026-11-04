#!/bin/sh
cd <PROJECT_ROOT>/src/cascade
PY=python
export HF_HOME='<HF_HOME>'
export PYTHONIOENCODING=utf-8
LOG=<PROJECT_ROOT>/results/lora_v2
$PY rerender_v3.py --size 672 --ctx 1.4 --n_train 8081 > <PROJECT_ROOT>/results/rerender_v3_672_all_log.txt 2>&1
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
$PY train_lora_v2.py --tag H_final_llm_coord_672_all_2ep --render stretch --label smooth --targets llm_vismlp --task coord --n 8081 --epochs 2 --img_key img_v3_672 --lock_img thermal_crop_v3_672.png --grad_ckpt --bs 1 --accum 16 > $LOG/H.log 2>&1
$PY eval_v3_big.py --adapter '<PROJECT_ROOT>\results\lora_v2\H_final_llm_coord_672_all_2ep\ep2' > <PROJECT_ROOT>/results/eval_v3_big/eval_H.log 2>&1
echo CHAIN7_DONE > $LOG/CHAIN7_DONE
