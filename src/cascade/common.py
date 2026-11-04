# -*- coding: utf-8 -*-
"""common.py — shared helpers for the cascade (RF gate -> grounded VLM diagnosis)."""
import json, os, re, numpy as np, cv2
MAN = r"<PROJECT_ROOT>\data\manifests\manifest_e_split.jsonl"
FEAT = r"<PROJECT_ROOT>\results\feat_cache.npz"
RF_PATH = r"<PROJECT_ROOT>\results\rf_equip.joblib"
CASES = r"<PROJECT_ROOT>\results\cases"
FEAT_NAMES = ['t_max','t_mean','t_p95','t_std','dT_scene_max','dT_scene_mean','dT_sib_max',
              'frac_hot','grad_max','grad_mean','hot_conc','w','h','aspect','scene_med','scene_p95',
              'hot_frac','n_hot','big_frac','hot_cx','hot_cy','compact','peak_z','skew','kurt','center_edge']
LO, HI = 10.0, 80.0   # fixed absolute colour scale (deg C) for every render

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

def clip_roi(T, b):
    H, W = T.shape; x = max(0, b['x']); y = max(0, b['y']); x2 = min(W, x + b['width']); y2 = min(H, y + b['height'])
    return T[y:y2, x:x2], (x, y, x2, y2)

def frame_feats_idx(T, anns):
    """Identical feature computation to extract_features.frame_feats, but returns per-annotation index + extras."""
    H, W = T.shape; sm = float(np.median(T)); sp95 = float(np.percentile(T, 95))
    info = []
    for a in anns:
        roi, _ = clip_roi(T, a['bbox'])
        info.append((a.get('class'), float(roi.max()) if roi.size else sm, roi) if roi.size >= 16 else None)
    out = []
    for i, a in enumerate(anns):
        if info[i] is None: continue
        cls, tmax, roi = info[i]
        sib = [info[j][1] for j in range(len(anns)) if j != i and info[j] and info[j][0] == cls]
        sib_ref = float(np.median(sib)) if sib else sm
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, 3)
        g = np.sqrt(gx * gx + gy * gy)
        thr = sm + 0.7 * (tmax - sm)
        feat = [tmax, float(roi.mean()), float(np.percentile(roi, 95)), float(roi.std()),
                tmax - sm, float(roi.mean()) - sm, tmax - sib_ref,
                float((roi > sp95).mean()), float(g.max()), float(g.mean()),
                float((roi > thr).mean()), float(roi.shape[1]), float(roi.shape[0]),
                float(roi.shape[0] / max(roi.shape[1], 1)), sm, sp95]
        rv = roi.ravel().astype(np.float64); mu = rv.mean(); sd = rv.std() + 1e-6
        skew = float(((rv-mu)**3).mean()/sd**3); kurt = float(((rv-mu)**4).mean()/sd**4 - 3)
        peak_z = float((tmax - mu)/sd)
        hi = sm + 0.6*(tmax - sm); hot = (roi > hi).astype(np.uint8)
        hot_frac = float(hot.mean())
        ncc, lbl = cv2.connectedComponents(hot); n_hot = ncc - 1
        big_frac = hcx = hcy = compact = 0.0
        if n_hot >= 1:
            sizes = np.bincount(lbl.ravel())[1:]; big = 1 + int(np.argmax(sizes)); bm = (lbl == big)
            big_frac = float(bm.sum())/hot.size
            ys, xs = np.where(bm); hcx = float(xs.mean())/roi.shape[1]; hcy = float(ys.mean())/roi.shape[0]
            cnts, _ = cv2.findContours(bm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            peri = cv2.arcLength(cnts[0], True) if cnts else 0
            compact = float(4*np.pi*bm.sum()/(peri*peri)) if peri > 1 else 0.0
        h_, w_ = roi.shape; c = roi[h_//3:2*h_//3, w_//3:2*w_//3]
        center = float(c.mean()) if c.size else mu
        edge = float((roi.sum() - c.sum())/max(roi.size - c.size, 1))
        feat += [hot_frac, float(n_hot), big_frac, hcx, hcy, compact, peak_z, skew, kurt, center - edge]
        py, px = np.unravel_index(int(np.argmax(roi)), roi.shape)
        extras = dict(sib_tmax=[round(s, 1) for s in sib], sib_ref=sib_ref, n_sib=len(sib),
                      peak_xy_in_roi=(int(px), int(py)), roi_shape=(int(roi.shape[1]), int(roi.shape[0])),
                      t_min=float(roi.min()), t_p50=float(np.median(roi)))
        out.append((i, feat, extras))
    return out

def onehot(fac, vocab):
    O = np.zeros((len(fac), len(vocab)), np.float32); vi = {f: i for i, f in enumerate(vocab)}
    for k, f in enumerate(fac):
        if f in vi: O[k, vi[f]] = 1
    return O

def load_rf():
    import joblib
    d = joblib.load(RF_PATH); return d['rf'], d['vocab']

def rf_prob(rf, vocab, feats, facs):
    X = np.hstack([np.array(feats, np.float32), onehot(facs, vocab)])
    return rf.predict_proba(X)[:, 1]

def colorize(T):
    return cv2.applyColorMap((np.clip((T-LO)/(HI-LO), 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)

def colorbar(h, w=46):
    bar = np.linspace(1, 0, h, dtype=np.float32)[:, None].repeat(w, 1)
    cb = cv2.applyColorMap((bar*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cb[:, :6] = 0
    for t in range(int(LO), int(HI)+1, 10):
        y = int((1-(t-LO)/(HI-LO))*(h-1)); y = min(max(y, 8), h-2)
        cv2.putText(cb, f"{t}", (8, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return cb

def render_full(T, anns, focus_i, out):
    """Whole frame at fixed scale, all bboxes drawn (focus = thick white, siblings = thin grey), colorbar."""
    col = colorize(T)
    for i, a in enumerate(anns):
        b = a['bbox']; c = (255, 255, 255) if i == focus_i else (160, 160, 160); th = 3 if i == focus_i else 1
        cv2.rectangle(col, (b['x'], b['y']), (b['x']+b['width'], b['y']+b['height']), c, th)
        if i == focus_i:
            cv2.putText(col, "TARGET", (b['x'], max(12, b['y']-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    img = np.hstack([col, colorbar(col.shape[0])])
    cv2.imwrite(out, img); return True

def render_crop(T, b, out, ctx=2.4, size=336, mark=None):
    H, W = T.shape; col = colorize(T)
    bw, bh = b['width'], b['height']; cx = b['x']+bw/2; cy = b['y']+bh/2; s = max(bw, bh)*ctx
    x = int(max(0, cx-s/2)); y = int(max(0, cy-s/2)); x2 = int(min(W, cx+s/2)); y2 = int(min(H, cy+s/2))
    cv2.rectangle(col, (b['x'], b['y']), (b['x']+bw, b['y']+bh), (255, 255, 255), 2)
    if mark: cv2.drawMarker(col, mark, (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
    c = col[y:y2, x:x2]
    if c.size == 0: return None
    cv2.imwrite(out, cv2.resize(c, (size, size))); return (x, y, x2, y2)

def render_rgb_crop(rgb, box, b, out, size=336):
    if rgb is None: return False
    x, y, x2, y2 = box
    if rgb.shape[0] < y2 or rgb.shape[1] < x2: return False
    im = rgb.copy(); cv2.rectangle(im, (b['x'], b['y']), (b['x']+b['width'], b['y']+b['height']), (0, 255, 0), 2)
    cv2.imwrite(out, cv2.resize(im[y:y2, x:x2], (size, size))); return True

def pos_words(cx, cy):
    v = 'top' if cy < 0.34 else ('bottom' if cy > 0.66 else 'middle')
    h = 'left' if cx < 0.34 else ('right' if cx > 0.66 else 'center')
    return 'center' if (v == 'middle' and h == 'center') else f"{v}-{h}"

POS_KO = {'top':'상단','middle':'중단','bottom':'하단','left':'좌측','center':'중앙','right':'우측'}
def pos_ko(p):
    if p == 'center': return '중앙'
    v, h = p.split('-'); return f"{POS_KO[v]} {POS_KO[h]}"
