# -*- coding: utf-8 -*-
"""
cascade_fewshot.py — Stage2 VLM with FEW-SHOT in-context exemplars (per equipment type).
Zero-shot VLM was degenerate (called every borderline case 'danger'). Here we show the VLM a few
CLEAR labeled examples (high-ΔT danger, low-ΔT normal) of the SAME equipment, then ask it to judge
the borderline query. Tests whether in-context adaptation makes it a real discriminator (no training/new data).
"""
import json, re, os, sys, io, time, random, collections
import numpy as np, cv2
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',line_buffering=True)

MAN=r"<PROJECT_ROOT>\data\manifests\manifest_dedup.jsonl"
TMPQ=r"<PROJECT_ROOT>\results\cascade_crops"; os.makedirs(TMPQ,exist_ok=True)
TMPE=r"<PROJECT_ROOT>\results\cascade_exemplars"; os.makedirs(TMPE,exist_ok=True)
CLASS_EN={'고압전동기':'high-voltage motor','차단기':'circuit breaker','변압기 접속부':'transformer connection',
          '단로기(DS)':'disconnector switch','신재생 설비':'renewable-energy unit','접지용 콘덴서':'grounding capacitor'}
LO,HI=10.0,70.0; N_CAND=30; BAND=(4.0,10.0)

def parse_csv(cf):
    try: lines=open(cf,encoding='utf-8').read().splitlines()
    except: return None
    rows=[]
    for l in lines[5:]:
        try: r=[float(p) for p in l.split(';') if p.strip()!='']
        except: continue
        if len(r)>=600: rows.append(r[:640])
    return np.array(rows,np.float32) if len(rows)>=400 else None
def render(T,b,out,ctx=2.2):
    H,W=T.shape; norm=np.clip((T-LO)/(HI-LO),0,1); col=cv2.applyColorMap((norm*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
    bw,bh=b['width'],b['height']; cx=b['x']+bw/2; cy=b['y']+bh/2; s=max(bw,bh)*ctx
    x=int(max(0,cx-s/2));y=int(max(0,cy-s/2));x2=int(min(W,cx+s/2));y2=int(min(H,cy+s/2))
    cv2.rectangle(col,(b['x'],b['y']),(b['x']+bw,b['y']+bh),(255,255,255),2)
    c=col[y:y2,x:x2]
    if c.size==0: return False
    cv2.imwrite(out,cv2.resize(c,(308,308))); return True

rows=[json.loads(l) for l in open(MAN,encoding='utf-8')]
val=[r for r in rows if r['split']=='val']; tr=[r for r in rows if r['split']=='train']
random.seed(0); random.shuffle(val); random.shuffle(tr)

def dT_of(T,b):
    H,W=T.shape; sm=float(np.median(T)); x=max(0,b['x']);y=max(0,b['y']);x2=min(W,x+b['width']);y2=min(H,y+b['height'])
    roi=T[y:y2,x:x2]; return (float(roi.max())-sm) if roi.size>=16 else None

# --- borderline val candidates ---
cands=[]
for r in val:
    if len(cands)>=N_CAND: break
    T=parse_csv(r.get('csv'));
    if T is None: continue
    for a in r['annotations']:
        st=a.get('status'); b=a.get('bbox')
        if st not in('danger','normal') or not b: continue
        d=dT_of(T,b)
        if d is not None and BAND[0]<=d<=BAND[1]:
            o=os.path.join(TMPQ,f"q{len(cands)}.png")
            if render(T,b,o): cands.append({'equip':a.get('class'),'dT':d,'label':st,'img':o}); break
need=set(c['equip'] for c in cands)
print(f"candidates {len(cands)} | equipment needed: {[CLASS_EN[e] for e in need]}")

# --- GLOBAL exemplar pool (rare equipment lack clear high-ΔT danger): clear danger ΔT>=10, clear normal ΔT<=4 ---
G={'danger':[],'normal':[]}
for r in tr:
    if len(G['danger'])>=3 and len(G['normal'])>=3: break
    T=parse_csv(r.get('csv'))
    if T is None: continue
    for a in r['annotations']:
        st=a.get('status'); b=a.get('bbox')
        if st not in('danger','normal') or not b: continue
        d=dT_of(T,b)
        if d is None: continue
        tgt='danger' if (st=='danger' and d>=10) else ('normal' if (st=='normal' and d<=4) else None)
        if tgt and len(G[tgt])<3:
            o=os.path.join(TMPE,f"g_{tgt}_{len(G[tgt])}.png")
            if render(T,b,o): G[tgt].append((o,d))
print(f"  global exemplars: danger {[round(x[1],1) for x in G['danger']]}, normal {[round(x[1],1) for x in G['normal']]}")

# --- VLM few-shot judge ---
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
mdl='Qwen/Qwen2.5-VL-3B-Instruct'
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(mdl,torch_dtype=torch.bfloat16,device_map='auto')
proc=AutoProcessor.from_pretrained(mdl)
def judge(c):
    eq=c['equip']
    content=[{"type":"text","text":
        f"Task: thermal-fault inspection of a {CLASS_EN[eq]} (electrical equipment). Fixed colour scale: brighter=hotter. "
        "A DANGER (fault) component has a clear localized hot spot much hotter than similar/adjacent parts; "
        "a NORMAL component is uniform or only mildly warm. Study these labeled examples:"}]
    for img,_ in G['normal']: content += [{"type":"image","image":img},{"type":"text","text":"-> NORMAL"}]
    for img,_ in G['danger']: content += [{"type":"image","image":img},{"type":"text","text":"-> DANGER"}]
    content += [{"type":"text","text":"Now classify the boxed component in this image:"},
                {"type":"image","image":c['img']},
                {"type":"text","text":"Answer strictly DANGER or NORMAL, then <=6-word reason."}]
    msgs=[{"role":"user","content":content}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ii,vi=process_vision_info(msgs)
    inp=proc(text=[text],images=ii,videos=vi,padding=True,return_tensors="pt").to(model.device)
    with torch.no_grad(): g=model.generate(**inp,max_new_tokens=40,do_sample=False)
    return proc.batch_decode(g[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()

print("\n=== FEW-SHOT VLM on borderline candidates ===")
correct=0; tn=0; nnorm=sum(c['label']=='normal' for c in cands)
for c in cands:
    ans=judge(c); pred='danger' if 'DANGER' in ans.upper() else 'normal'
    ok=pred==c['label']; correct+=ok
    if c['label']=='normal' and pred=='normal': tn+=1
    print(f"  [{'OK ' if ok else 'X  '}] {CLASS_EN[c['equip']][:20]:20s} ΔT={c['dT']:4.1f} truth={c['label']:6s} VLM={pred:6s} | {ans[:50]}")
print(f"\nfew-shot accuracy: {correct}/{len(cands)} = {correct/max(len(cands),1):.2f}   (zero-shot was 0.20 = all-danger)")
print(f"false-positives correctly rejected (true NORMAL called normal): {tn}/{nnorm}   <- the cascade's whole point")
print("NOTE: smoke test, small N. If few-shot rejects some FPs, in-context adaptation works -> scale with data.")
