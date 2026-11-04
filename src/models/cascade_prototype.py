# -*- coding: utf-8 -*-
"""
cascade_prototype.py — Stage1 (high-recall ΔT rule gate) -> Stage2 (VLM false-positive filter).
Purpose: prototype the cascade so it is READY to run when more sessions are acquired.
Key fixes baked in:
  * Input to VLM = RADIOMETRIC temperature rendered at a FIXED absolute scale (consistent color<->temp),
    NOT the auto-ranged camera JPG.
  * VLM judges the BORDERLINE flagged cases (where the ΔT rule is uncertain) using visual/contextual reasoning.
Eval question (once data is large enough): at fixed high recall, does Stage1+VLM improve PRECISION over the rule alone?

This run = SMOKE TEST on a few borderline candidates (small N -> NOT a verdict), to confirm the pipeline end-to-end.

Usage:  set HF_HOME=<HF_HOME>
        python src/models/cascade_prototype.py --n 30
"""
import json, re, os, sys, io, time, random, argparse, collections
import numpy as np, cv2
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',line_buffering=True)

MAN=r"<PROJECT_ROOT>\data\manifests\manifest_dedup.jsonl"
TMP=r"<PROJECT_ROOT>\results\cascade_crops"; os.makedirs(TMP,exist_ok=True)
CLASS_EN={'고압전동기':'high-voltage motor','차단기':'circuit breaker','변압기 접속부':'transformer connection',
          '단로기(DS)':'disconnector switch','신재생 설비':'renewable-energy unit (battery/PV)','접지용 콘덴서':'grounding capacitor'}
LO,HI=10.0,70.0   # fixed absolute temperature scale (deg C) for consistent rendering

def parse_csv(cf):
    try: lines=open(cf,encoding='utf-8').read().splitlines()
    except: return None
    rows=[]
    for l in lines[5:]:
        try: r=[float(p) for p in l.split(';') if p.strip()!='']
        except: continue
        if len(r)>=600: rows.append(r[:640])
    return np.array(rows,np.float32) if len(rows)>=400 else None

def render(T, b, out, ctx=2.2):
    H,W=T.shape
    norm=np.clip((T-LO)/(HI-LO),0,1); img8=(norm*255).astype(np.uint8)
    col=cv2.applyColorMap(img8,cv2.COLORMAP_INFERNO)
    bw,bh=b['width'],b['height']; cx=b['x']+bw/2; cy=b['y']+bh/2; side=max(bw,bh)*ctx
    x=int(max(0,cx-side/2));y=int(max(0,cy-side/2));x2=int(min(W,cx+side/2));y2=int(min(H,cy+side/2))
    cv2.rectangle(col,(b['x'],b['y']),(b['x']+bw,b['y']+bh),(255,255,255),2)
    crop=col[y:y2,x:x2]
    if crop.size==0: return False
    cv2.imwrite(out,cv2.resize(crop,(336,336))); return True

# ---- Stage 1: gate + borderline candidate selection ----
ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=30); ap.add_argument('--band',default='4,10')
ap.add_argument('--manifest',default=MAN); args=ap.parse_args()
lo_b,hi_b=[float(x) for x in args.band.split(',')]
rows=[json.loads(l) for l in open(args.manifest,encoding='utf-8') if json.loads(l)['split']=='val']
random.seed(0); random.shuffle(rows)
cands=[]   # borderline flagged: scene-ΔT in band, mix of danger/normal
for r in rows:
    if len(cands)>=args.n: break
    T=parse_csv(r.get('csv'))
    if T is None: continue
    sm=float(np.median(T)); H,W=T.shape
    for a in r['annotations']:
        st=a.get('status'); b=a.get('bbox')
        if st not in('danger','normal') or not b: continue
        x=max(0,b['x']);y=max(0,b['y']);x2=min(W,x+b['width']);y2=min(H,y+b['height'])
        roi=T[y:y2,x:x2]
        if roi.size<16: continue
        dT=float(roi.max())-sm
        if lo_b<=dT<=hi_b:   # borderline band where the rule is uncertain
            out=os.path.join(TMP,f"c{len(cands)}.png")
            if render(T,b,out):
                cands.append({'equip':CLASS_EN.get(a.get('class'),a.get('class')),'dT':dT,'label':st,'img':out})
                break
print(f"Stage1 flagged {len(cands)} borderline candidates (scene-ΔT in [{lo_b},{hi_b}]°C)")
print(f"  truth in band: danger {sum(c['label']=='danger' for c in cands)}, normal {sum(c['label']=='normal' for c in cands)}")
print(f"  (rule ~can't separate these; this is where the VLM must add value)")

# ---- Stage 2: VLM judges each borderline candidate from the rendered thermal image ----
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
mdl='Qwen/Qwen2.5-VL-3B-Instruct'
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(mdl,torch_dtype=torch.bfloat16,device_map='auto')
proc=AutoProcessor.from_pretrained(mdl)
def judge(img,equip):
    prompt=(f"This is a thermal (infrared) image of a {equip} in an electrical substation/power plant. "
            "Warmer areas are brighter (white/yellow), cooler areas are dark/purple; the colour scale is FIXED. "
            "The component of interest is outlined by the white box. A thermal FAULT shows a localized overheated "
            "hot spot on the component that is clearly hotter than its surroundings and than similar/adjacent parts. "
            "Judge ONLY from the thermal pattern. Is the boxed component in a DANGER (overheating fault) state? "
            "Answer strictly 'DANGER' or 'NORMAL', then a 6-word reason.")
    msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ii,vi=process_vision_info(msgs)
    inp=proc(text=[text],images=ii,videos=vi,padding=True,return_tensors="pt").to(model.device)
    with torch.no_grad(): g=model.generate(**inp,max_new_tokens=40,do_sample=False)
    return proc.batch_decode(g[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()

print("\n=== SMOKE TEST: VLM verdict on borderline candidates ===")
correct=0
for c in cands:
    ans=judge(c['img'],c['equip'])
    pred='danger' if 'DANGER' in ans.upper() else 'normal'
    ok=pred==c['label']; correct+=ok
    print(f"  [{ 'OK ' if ok else 'X  '}] equip={c['equip'][:22]:22s} ΔT={c['dT']:4.1f} truth={c['label']:6s} VLM={pred:6s} | {ans[:60]}")
print(f"\nVLM accuracy on borderline band: {correct}/{len(cands)} = {correct/max(len(cands),1):.2f}")
print("NOTE: SMOKE TEST (small N) — pipeline works end-to-end. Real verdict needs more sessions.")
print("Extensions ready: few-shot exemplars in prompt, give/withhold ΔT, precision@fixed-recall over full val.")
