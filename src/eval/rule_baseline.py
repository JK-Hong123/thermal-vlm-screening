# -*- coding: utf-8 -*-
"""
rule_baseline.py — Phase 1 Experiment 1: how far do RULES alone go? (the ceiling a VLM must beat)
Radiometric features per bbox incl. proper thermography references:
  scene-dT  = bbox_max - scene_median         (already ~0.90)
  sibling-dT= bbox_max - median(other same-class components' max in same image)  <- proper reference
  + per-equipment context, gradient, geometry.
Logistic combo trained on train split, evaluated on val with SESSION-grouped bootstrap 95% CI
(effective N is sessions, not frames). Answers: is detection solved by rules, or is there residual room?
"""
import json, re, os, sys, io, time, collections, random
import numpy as np, cv2
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',line_buffering=True)
import torch, torch.nn as nn

MAN=r"<PROJECT_ROOT>\data\manifests\manifest_dedup.jsonl"
CACHE=r"<PROJECT_ROOT>\results\rulefeats.npz"
CLASSES=['고압전동기','차단기','변압기 접속부','단로기(DS)','신재생 설비','접지용 콘덴서']
N_TRAIN_IMG=4000
random.seed(0); torch.manual_seed(0)

def sess(p):
    b=os.path.basename(p); m=re.match(r'^(.*\(\d+\))\(\d+\)\d+\.jpg$', b); return m.group(1) if m else b
def parse_csv(cf):
    try: lines=open(cf,encoding='utf-8').read().splitlines()
    except: return None
    rows=[]
    for l in lines[5:]:
        try: r=[float(p) for p in l.split(';') if p.strip()!='']
        except: continue
        if len(r)>=600: rows.append(r[:640])
    return np.array(rows,np.float32) if len(rows)>=400 else None
def auroc(s,y):
    s=np.asarray(s);y=np.asarray(y);o=s.argsort();r=np.empty_like(o,float);r[o]=np.arange(1,len(s)+1)
    p=y.sum();n=len(y)-p
    return float('nan') if p==0 or n==0 else float((r[y==1].sum()-p*(p+1)/2)/(p*n))

def bbox_feats(r):
    T=parse_csv(r['csv'])
    if T is None: return []
    H,W=T.shape; sm=float(np.median(T)); sp95=float(np.percentile(T,95))
    anns=[a for a in r['annotations'] if a.get('bbox') and a.get('status') in('danger','normal')]
    info=[]
    for a in anns:
        b=a['bbox']; x=max(0,b['x']);y=max(0,b['y']);x2=min(W,x+b['width']);y2=min(H,y+b['height'])
        roi=T[y:y2,x:x2]
        if roi.size<16: info.append(None); continue
        info.append((a.get('class'),float(roi.max()),float(roi.mean()),roi,(x,y,x2,y2)))
    out=[]
    for i,a in enumerate(anns):
        if info[i] is None: continue
        cls,tmax,tmean,roi,_=info[i]
        sib=[info[j][1] for j in range(len(anns)) if j!=i and info[j] and info[j][0]==cls]
        sib_ref=float(np.median(sib)) if sib else sm
        gx=cv2.Sobel(roi,cv2.CV_32F,1,0,3); gy=cv2.Sobel(roi,cv2.CV_32F,0,1,3); g=np.sqrt(gx*gx+gy*gy)
        feat=[tmax, tmean, tmax-sm, tmean-sm, tmax-sib_ref, tmean-sib_ref,
              float((roi>sp95).mean()), float(g.max()), float(g.mean()),
              float(roi.shape[1]),float(roi.shape[0]),float(roi.shape[0]/max(roi.shape[1],1))]
        feat+=[1.0 if cls==c else 0.0 for c in CLASSES]
        out.append((feat, 1 if a.get('status')=='danger' else 0, sess(r['img']),
                    tmax-sm, tmax-sib_ref))  # keep scene_dT, sibling_dT for single-feature refs
    return out

if os.path.exists(CACHE):
    z=np.load(CACHE,allow_pickle=True)
    Xtr,Ytr,Xva,Yva,Sva,SdT,SibdT=z['Xtr'],z['Ytr'],z['Xva'],z['Yva'],z['Sva'],z['SdT'],z['SibdT']
    print("loaded cache",Xtr.shape,Xva.shape)
else:
    rows=[json.loads(l) for l in open(MAN,encoding='utf-8')]
    tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='val']
    random.shuffle(tr); tr=tr[:N_TRAIN_IMG]
    def build(recs,name):
        X=[];Y=[];S=[];sd=[];sb=[];c=0;t0=time.time()
        for r in recs:
            for feat,lab,ss,scenedt,sibdt in bbox_feats(r):
                X.append(feat);Y.append(lab);S.append(ss);sd.append(scenedt);sb.append(sibdt)
            c+=1
            if c%500==0: print(f"  {name} {c}/{len(recs)} ({time.time()-t0:.0f}s) {len(Y)} bboxes")
        return np.array(X,np.float32),np.array(Y,np.float32),np.array(S),np.array(sd,np.float32),np.array(sb,np.float32)
    print("parsing dedup CSVs...")
    Xtr,Ytr,_,_,_=build(tr,'train'); Xva,Yva,Sva,SdT,SibdT=build(va,'val')
    np.savez(CACHE,Xtr=Xtr,Ytr=Ytr,Xva=Xva,Yva=Yva,Sva=Sva,SdT=SdT,SibdT=SibdT)
print(f"train bboxes {len(Ytr)} (danger {int(Ytr.sum())}), val {len(Yva)} (danger {int(Yva.sum())}), val sessions {len(set(Sva.tolist()))}")

def boot_ci(scores,Y,S,n=1000):
    S=np.asarray(S); uniq=np.unique(S); rng=np.random.default_rng(0); vals=[]
    for _ in range(n):
        pick=rng.choice(uniq,len(uniq),replace=True)
        idx=np.concatenate([np.where(S==s)[0] for s in pick])
        a=auroc(np.asarray(scores)[idx],np.asarray(Y)[idx])
        if a==a: vals.append(a)
    return np.percentile(vals,2.5),np.percentile(vals,97.5)

print("\n=== RULE CEILING (val, session-grouped 95% CI) ===")
for name,sc in [("scene-dT only",SdT),("sibling-dT only",SibdT),("max(scene,sibling)-dT",np.maximum(SdT,SibdT))]:
    a=auroc(sc,Yva); lo,hi=boot_ci(sc,Yva,Sva); print(f"  {name:22s} AUROC {a:.3f}  [{lo:.3f}, {hi:.3f}]")

# logistic combo on all features
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr_n=(Xtr-mu)/sd; Xva_n=(Xva-mu)/sd
net=nn.Sequential(nn.Linear(Xtr.shape[1],32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3); lf=nn.BCEWithLogitsLoss()
Xt=torch.tensor(Xtr_n);Yt=torch.tensor(Ytr);Xv=torch.tensor(Xva_n)
for ep in range(80):
    net.train();perm=torch.randperm(len(Xt))
    for i in range(0,len(perm),256):
        bi=perm[i:i+256];opt.zero_grad();loss=lf(net(Xt[bi]).squeeze(1),Yt[bi]);loss.backward();opt.step()
net.eval()
with torch.no_grad(): sc=torch.sigmoid(net(Xv).squeeze(1)).numpy()
a=auroc(sc,Yva); lo,hi=boot_ci(sc,Yva,Sva)
print(f"  {'logistic/MLP combo':22s} AUROC {a:.3f}  [{lo:.3f}, {hi:.3f}]")
print("\nverdict: if sibling-dT / combo >> scene-dT, better REFERENCE (still a rule) closes the gap.")
print("         residual room for a VLM = whatever these rules still miss.")
