# -*- coding: utf-8 -*-
"""
train_spatial_cnn.py — radiometric 3-ch spatial CNN (+ equipment one-hot) for bbox fault classification.
Session-level train/val(early-stop)/test/gen_test (no frame leakage). Compares to rule(0.744)/RF(0.881).
"""
import numpy as np, sys, io, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import torch, torch.nn as nn

Z = np.load(r"<PROJECT_ROOT>\results\crops_cache.npz", allow_pickle=True)
X, Y, FAC, SES, GRP, DT = Z['X'], Z['Y'].astype(np.float32), Z['FAC'], Z['SES'], Z['GRP'], Z['DT']
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"crops {X.shape} on {dev}; danger frac {Y.mean():.3f}")
rng = np.random.default_rng(0)

# session-level val carve from train
tr_all = np.where(GRP == 'train')[0]
tr_sess = np.unique(SES[tr_all]); rng.shuffle(tr_sess)
val_sess = set(tr_sess[:int(len(tr_sess)*0.15)])
is_val = np.array([s in val_sess for s in SES])
tr = (GRP == 'train') & ~is_val; va = (GRP == 'train') & is_val
te = GRP == 'test'; ge = GRP == 'gen_test'
print(f"train {tr.sum()}, val {va.sum()}, test {te.sum()}, gen {ge.sum()}")

vocab = sorted(set(FAC[tr])); vi = {f: i for i, f in enumerate(vocab)}
def oh(fac):
    O = np.zeros((len(fac), len(vocab)), np.float32)
    for k, f in enumerate(fac):
        if f in vi: O[k, vi[f]] = 1
    return O
E = oh(FAC)

def auroc(s, y):
    s=np.asarray(s,float); y=np.asarray(y); o=s.argsort(); r=np.empty_like(o,float); r[o]=np.arange(1,len(s)+1)
    p=y.sum(); n=len(y)-p
    return float('nan') if p==0 or n==0 else float((r[y==1].sum()-p*(p+1)/2)/(p*n))
def ci(sc, y, ses, n=800):
    ses=np.asarray(ses); uq=np.unique(ses); rr=np.random.default_rng(0); v=[]
    idxby={s:np.where(ses==s)[0] for s in uq}
    for _ in range(n):
        pick=rr.choice(uq,len(uq),replace=True); idx=np.concatenate([idxby[s] for s in pick])
        a=auroc(np.asarray(sc)[idx],np.asarray(y)[idx])
        if a==a: v.append(a)
    return np.percentile(v,2.5), np.percentile(v,97.5)

def conv(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
class Net(nn.Module):
    def __init__(s,ne):
        super().__init__()
        s.c=nn.Sequential(conv(3,32),conv(32,64),conv(64,128),nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        s.f=nn.Sequential(nn.Linear(128+ne,64),nn.ReLU(),nn.Dropout(0.4),nn.Linear(64,1))
    def forward(s,x,e): return s.f(torch.cat([s.c(x),e],1)).squeeze(1)

net=Net(len(vocab)).to(dev)
pw=torch.tensor([(Y[tr]==0).sum()/max((Y[tr]==1).sum(),1)],dtype=torch.float32,device=dev)
opt=torch.optim.Adam(net.parameters(),lr=8e-4,weight_decay=1e-4); lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
def batch(mask,bs=256,shuf=True,aug=False):
    idx=np.where(mask)[0]
    if shuf: rng.shuffle(idx)
    for i in range(0,len(idx),bs):
        bi=idx[i:i+bs]; xb=torch.from_numpy(X[bi].astype(np.float32).transpose(0,3,1,2)/255.).to(dev)
        if aug and np.random.rand()<0.5: xb=torch.flip(xb,[3])
        yield xb, torch.from_numpy(Y[bi]).to(dev), torch.from_numpy(E[bi]).to(dev)
def scores(mask):
    net.eval(); out=[]
    with torch.no_grad():
        for xb,yb,eb in batch(mask,512,shuf=False): out.append(torch.sigmoid(net(xb,eb)).cpu().numpy())
    return np.concatenate(out)

best=0; best_state=None; patience=0
for ep in range(40):
    net.train(); t0=time.time()
    for xb,yb,eb in batch(tr,256,True,aug=True):
        opt.zero_grad(); loss=lossf(net(xb,eb),yb); loss.backward(); opt.step()
    va_auc=auroc(scores(va),Y[va])
    if va_auc>best: best=va_auc; best_state={k:v.clone() for k,v in net.state_dict().items()}; patience=0
    else: patience+=1
    if (ep+1)%3==0: print(f"  ep{ep+1} val AUROC {va_auc:.3f} (best {best:.3f}) {time.time()-t0:.0f}s")
    if patience>=6: print(f"  early stop ep{ep+1}"); break
net.load_state_dict(best_state)

print("\n===== CNN 결과 (best val) =====")
for name,mask in [('WITHIN-TEST',te),('GEN-TEST(zero-shot)',ge)]:
    sc=scores(mask); y=Y[mask]; a=auroc(sc,y); lo,hi=ci(sc,y,SES[mask])
    print(f"  {name:20s} AUROC {a:.3f} [{lo:.3f}, {hi:.3f}]")
# hard cases on test
sc=scores(te); dt=DT[te]; y=Y[te]
hn=(y==0)&(dt>6); sf=(y==1)&(dt<4)
print("\n===== hard cases (within-test) =====")
print(f"  hot-but-normal→normal: CNN 정답률 {100*((sc[hn]<0.5)).mean():.0f}% (n={hn.sum()})  [rule 0%, RF 76%]")
print(f"  subtle-fault→danger:   CNN 정답률 {100*((sc[sf]>=0.5)).mean():.0f}% (n={sf.sum()})  [rule 0%, RF 9%]")
print("\n비교: rule 0.744 / RF 0.881 (within-test)")
torch.save(best_state, r"<PROJECT_ROOT>\results\spatial_cnn.pt")
