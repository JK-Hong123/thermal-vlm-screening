# -*- coding: utf-8 -*-
"""gemini_subject.py — run a Gemini model as the SUBJECT of the blind pass-1 protocol (identical prompt and images as the
small models and the frontier model), for the marker-displacement control: conditions v2 (marker at the true peak) and
v2_markerpert (marker drawn in a different cell). Direct REST (AQ.-format keys). Resumable: skips existing outputs.
usage: GOOGLE_API_KEY=... python gemini_subject.py --model gemini-3.5-flash-lite [--cond v2,v2_markerpert] [--limit N]
then:  python gemini_subject.py --score --model gemini-3.5-flash-lite
"""
import json, os, sys, io, re, time, base64, argparse, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, requests
from common import MAN, parse_csv, clip_roi
import protocol_v2 as P
L = r"<PROJECT_ROOT>\results\cases_lock"; P1T = r"<PROJECT_ROOT>\prompts\pass1_blind_v2.txt"
def out_dir(model): return os.path.join(r"<PROJECT_ROOT>\results\vlm_out", 'subject_' + model.replace('.', '_').replace('-', '_'))
def b64(p): return base64.b64encode(open(p, 'rb').read()).decode('ascii')
def perturbed_point(b, px, py):
    w, h = b['width'], b['height']; gx = min(2, int(3*px/max(w, 1))); gy = min(2, int(3*py/max(h, 1))); nx, ny = 2-gx, 2-gy
    if (nx, ny) == (gx, gy): nx = (gx+1) % 3
    return int(b['x'] + (nx+0.5)*w/3), int(b['y'] + (ny+0.5)*h/3)
def cell_of(b, x, y):
    gx = min(2, int(3*(x-b['x'])/max(b['width'], 1))); gy = min(2, int(3*(y-b['y'])/max(b['height'], 1))); return P.GRID_KO[gy*3+gx]
def parse_json(txt):
    m = re.search(r'\{.*\}', txt or '', re.S)
    if not m: return None
    s = m.group(0)
    for c in (s, s.replace('\n', ' '), re.sub(r',\s*}', '}', s), re.sub(r',\s*]', ']', s)):
        try: return json.loads(c)
        except Exception: pass
    return None
def kappa(pairs):
    if not pairs: return float('nan')
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs); po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats); return (po-pe)/(1-pe) if pe < 1 else float('nan')

def build(cid, cond, t1):
    d = os.path.join(L, cid); f = json.load(open(os.path.join(d, 'facts.json'), encoding='utf-8'))
    crop = {'v2': 'thermal_crop_v2', 'v2_markerpert': 'thermal_crop_v2_pert'}[cond]
    use_rgb = bool(f['images'].get('rgb_crop_v2'))
    parts = ' / '.join(P.PARTS.get(f['equipment'], []) + ['확인 불가'])
    rgb_line = "이미지 3: 같은 위치의 가시광 사진(참고용; 열화상과 정렬은 근사, 상자 없음)." if use_rgb else "가시광 사진 없음."
    prompt = t1.format(equipment=f['equipment'], parts=parts, rgb_line=rgb_line)
    parts_ = [{"text": "[이미지 1]"}, {"inline_data": {"mime_type": "image/png", "data": b64(os.path.join(d, 'thermal_full.png'))}},
              {"text": "[이미지 2]"}, {"inline_data": {"mime_type": "image/png", "data": b64(os.path.join(d, crop + '.png'))}}]
    if use_rgb: parts_ += [{"text": "[이미지 3]"}, {"inline_data": {"mime_type": "image/png", "data": b64(os.path.join(d, 'rgb_crop_v2.png'))}}]
    parts_.append({"text": prompt})
    return parts_

def call(model, parts_, key, temperature=0):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    gen = {"responseMimeType": "application/json", "temperature": temperature}
    if "lite" not in model: gen["thinkingConfig"] = {"thinkingBudget": 0}
    r = requests.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json={"contents": [{"role": "user", "parts": parts_}], "generationConfig": gen}, timeout=240)
    if r.status_code != 200: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

def run(model, conds, limit):
    key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY'); assert key, 'no key'
    t1 = open(P1T, encoding='utf-8').read(); idx = [json.loads(l) for l in open(os.path.join(L, 'index.jsonl'), encoding='utf-8')][:limit]
    for cond in conds:
        od = os.path.join(out_dir(model), cond); os.makedirs(od, exist_ok=True); ok = fail = 0
        for it in idx:
            cid = it['case_id']; of = os.path.join(od, cid + '.json')
            if os.path.exists(of): ok += 1; continue
            for attempt in range(4):
                try:
                    raw = call(model, build(cid, cond, t1), key); json.dump({'raw': raw, 'model': model, 'cond': cond}, open(of, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); ok += 1; print(cond, cid, 'ok'); break
                except Exception as e:
                    msg = repr(e); print(cond, cid, 'retry' if attempt < 3 else 'FAILED', msg[:120])
                    time.sleep(20 if '429' in msg else 5)
            else: fail += 1
            time.sleep(4.2)   # free tier: 15 requests / minute
        print(f"{cond}: {ok} ok, {fail} failed")

def score(model):
    recs = {json.loads(l)['frame_key']: json.loads(l) for l in open(MAN, encoding='utf-8')}
    idx = [json.loads(l) for l in open(os.path.join(L, 'index.jsonl'), encoding='utf-8')]; rows = {'v2': [], 'v2_markerpert': []}
    for it in idx:
        f = json.load(open(os.path.join(L, it['case_id'], 'facts.json'), encoding='utf-8')); b = f['bbox']; truth = P.grid_ko(f['thermal']['hotspot_position_in_target'])
        T = parse_csv(recs[f['frame_key']]['csv']); roi, (x, y, x2, y2) = clip_roi(T, b); py, px = np.unravel_index(int(np.argmax(roi)), roi.shape); pert = cell_of(b, *perturbed_point(b, px, py))
        for cond in rows:
            fp = os.path.join(out_dir(model), cond, it['case_id'] + '.json')
            if not os.path.exists(fp): continue
            raw = json.load(open(fp, encoding='utf-8')).get('raw', ''); obs, viol = P.validate_pass1(parse_json(raw), f); cc = P.cross_check(obs, f)
            rows[cond].append(dict(pg=obs['p_grid'], truth=truth, pert=pert, sp=cc['spatial_check'], pat=cc['pattern_check'], vf=obs['visual_flags'], digits=bool(re.search(r'\d', raw))))
    out = []
    for cond, R in rows.items():
        if not R: out.append(f"{cond}: no outputs"); continue
        n = len(R); sp = collections.Counter(r['sp'] for r in R); pairs = [(r['pg'], r['truth']) for r in R if r['pg'] in P.GRID_XY]
        line = f"[{model}/{cond}] n={n} | p_grid exact {sp['exact']/n:.2f} adj+ {(sp['exact']+sp['adjacent'])/n:.2f} | κ vs truth {kappa(pairs):.3f}"
        if cond == 'v2_markerpert': line += f" | follows moved marker {np.mean([r['pg']==r['pert'] for r in R]):.2f} (κ vs moved {kappa([(r['pg'], r['pert']) for r in R if r['pg'] in P.GRID_XY]):.3f})"
        line += f" | pattern match {np.mean([r['pat']=='match' for r in R]):.2f} | digits {np.mean([r['digits'] for r in R]):.2f}"
        out.append(line); out.append("   answers: " + str(collections.Counter(r['pg'] for r in R).most_common(6))); out.append("   visual flags: " + str(dict(collections.Counter(v for r in R for v in r['vf']))))
    txt = "\n".join(out); print(txt); open(os.path.join(out_dir(model), 'scores.txt'), 'w', encoding='utf-8').write(txt)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='gemini-3.5-flash-lite'); ap.add_argument('--cond', default='v2,v2_markerpert'); ap.add_argument('--limit', type=int, default=None); ap.add_argument('--score', action='store_true')
    a = ap.parse_args()
    if a.score: score(a.model)
    else: run(a.model, a.cond.split(','), a.limit)
