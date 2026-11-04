# -*- coding: utf-8 -*-
"""
inspect_subtle.py — visualize the subtle-fault cases (test, danger, low scene-ΔT) that all models miss.
Renders fixed-scale thermal (bbox) + paired RGB + numeric context (scene-ΔT, sibling-ΔT). Also stats
on whether these are hotter than SIBLING components (would explain the label via reference beyond scene).
"""
import json, os, re, sys, io, random
import numpy as np, cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
MAN = r"<PROJECT_ROOT>\data\manifests\manifest_e_split.jsonl"
OUTD = r"<USER_HOME>\AppData\Local\Temp\claude\<SESSION_DIR>\dab93585-2101-4579-93da-5b01af9e2801\scratchpad"
LO, HI = 10.0, 70.0
def parse_csv(cf):
    try: lines = open(cf, encoding='utf-8').read().splitlines()
    except: return None
    rows = []
    for l in lines[5:]:
        try: r = [float(p) for p in l.split(';') if p.strip() != '']
        except: continue
        if len(r) >= 600: rows.append(r[:640])
    return np.array(rows, np.float32) if len(rows) >= 400 else None
def imread_u(p):
    try: return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    except: return None
def render_pair(T, rgbimg, b, cap):
    H, W = T.shape
    col = cv2.applyColorMap((np.clip((T-LO)/(HI-LO),0,1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    bw, bh = b['width'], b['height']; cx=b['x']+bw/2; cy=b['y']+bh/2; s=max(bw,bh)*2.4
    x=int(max(0,cx-s/2)); y=int(max(0,cy-s/2)); x2=int(min(W,cx+s/2)); y2=int(min(H,cy+s/2))
    cv2.rectangle(col,(b['x'],b['y']),(b['x']+bw,b['y']+bh),(255,255,255),2)
    tc = cv2.resize(col[y:y2,x:x2],(240,240))
    if rgbimg is not None and rgbimg.shape[0]>=y2 and rgbimg.shape[1]>=x2:
        cv2.rectangle(rgbimg,(b['x'],b['y']),(b['x']+bw,b['y']+bh),(0,255,0),2)
        rc = cv2.resize(rgbimg[y:y2,x:x2],(240,240))
    else: rc = np.zeros((240,240,3),np.uint8); cv2.putText(rc,'no RGB',(70,120),0,0.7,(200,200,200),1)
    pair = np.hstack([tc, rc])
    cv2.rectangle(pair,(0,0),(480,20),(0,0,0),-1); cv2.putText(pair,cap,(3,14),0,0.42,(0,255,255),1)
    return pair

recs = [json.loads(l) for l in open(MAN, encoding='utf-8')]
# only frames that HAVE a danger annotation (skip normal-only frames -> far fewer CSV parses)
test = [r for r in recs if r.get('split_group')=='test' and r.get('csv')
        and any(a.get('status')=='danger' for a in r.get('annotations',[]))]
random.seed(0); random.shuffle(test)
print(f"test danger-frames to scan: {len(test)}")
subtle = []; sib_all = []
panels = []; normals = []
for r in test:
    T = parse_csv(r['csv'])
    if T is None: continue
    sm = float(np.median(T)); H,W=T.shape
    anns=[a for a in r['annotations'] if a.get('bbox') and a.get('status') in ('danger','normal')]
    maxs={}
    for a in anns:
        b=a['bbox']; x=max(0,b['x']);y=max(0,b['y']);x2=min(W,x+b['width']);y2=min(H,y+b['height'])
        roi=T[y:y2,x:x2]; maxs[id(a)]=float(roi.max()) if roi.size else sm
    for a in anns:
        if a['status']!='danger': continue
        b=a['bbox']; x=max(0,b['x']);y=max(0,b['y']);x2=min(W,x+b['width']);y2=min(H,y+b['height'])
        roi=T[y:y2,x:x2]
        if roi.size<16: continue
        dts=float(roi.max())-sm
        if dts<4:  # subtle fault
            sib=[maxs[id(o)] for o in anns if o is not a and o['status'] in('danger','normal') and o.get('class')==a.get('class')]
            sib_ref=float(np.median(sib)) if sib else sm
            dsib=float(roi.max())-sib_ref
            subtle.append((r['facility'],dts,dsib,len(sib))); sib_all.append(dsib)
            if len(panels)<10:
                rgb=imread_u(r.get('rgb'))
                cap=f"{r['facility'][:12]} dTscene={dts:.1f} dTsib={dsib:.1f}(n={len(sib)})"
                panels.append(render_pair(T,rgb,b,cap))
    if len(panels)>=10 and len(subtle)>=40: break

# a few NORMAL of common equipment for reference (matched to subtle equip)
fac_need=set(f for f,_,_,_ in subtle)
for r in test:
    if len(normals)>=4: break
    if r['facility'] not in fac_need: continue
    T=parse_csv(r['csv']);
    if T is None: continue
    sm=float(np.median(T)); H,W=T.shape
    for a in r['annotations']:
        if a.get('status')!='normal' or not a.get('bbox'): continue
        b=a['bbox']; x=max(0,b['x']);y=max(0,b['y']);x2=min(W,x+b['width']);y2=min(H,y+b['height'])
        roi=T[y:y2,x:x2]
        if roi.size<16: continue
        rgb=imread_u(r.get('rgb'))
        normals.append(render_pair(T,rgb,b,f"NORMAL {r['facility'][:12]} dTscene={float(roi.max())-sm:.1f}"))
        break

def montage(ps,fn,cols=2):
    if not ps: return
    rows=[np.hstack(ps[i:i+cols]) if len(ps[i:i+cols])==cols else np.hstack(ps[i:i+cols]+[np.zeros_like(ps[0])]*(cols-len(ps[i:i+cols]))) for i in range(0,len(ps),cols)]
    cv2.imwrite(os.path.join(OUTD,fn), np.vstack(rows)); print("saved",fn)
montage(panels,'subtle_faults.png'); montage(normals,'subtle_normals.png')

sib_all=np.array(sib_all)
print(f"\n=== subtle-fault 통계 (test, danger & dTscene<4): {len(subtle)}건 ===")
print(f"  형제부품 대비 ΔT(dTsib): median {np.median(sib_all):.1f}, >2°C인 비율 {100*(sib_all>2).mean():.0f}%, >0 비율 {100*(sib_all>0).mean():.0f}%")
import collections
print("  설비 분포:", dict(collections.Counter(f for f,_,_,_ in subtle).most_common(6)))
