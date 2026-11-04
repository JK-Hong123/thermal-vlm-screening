# -*- coding: utf-8 -*-
"""probe_api.py — the synthetic spatial probe (L1 blob, L2 marked blob; 36 images each, truth cell = index mod 9) and the
fact-falsification control for API/agent models.
  --build            write results/probe/tasks_probe.json (prompts + image paths + out paths) for agent batches (frontier)
  --gemini MODEL     run L1/L2 probe AND the fact-falsification tasks through the Gemini REST API (free tier pacing)
  --score DIR        score a directory of outputs {"raw": ...} named L1_synth_blob_NN.json / L2_synth_mark_NN.json
"""
import json, os, sys, io, re, time, base64, argparse, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import requests
from protocol_v2 import GRID_KO, GRID_XY
OUTD = r"<PROJECT_ROOT>\results\probe"; F = r"<PROJECT_ROOT>\results\vlm_out\frontier"
PROMPT = """이미지에 흰색 사각형(대상 상자)이 있습니다. 상자 내부를 가로 3칸 × 세로 3칸으로 나누었을 때,
{what} 어느 칸에 있습니까?
다음 중 하나만 정확히 출력하십시오(다른 말 금지):
상단 좌측 / 상단 중앙 / 상단 우측 / 중단 좌측 / 중앙 / 중단 우측 / 하단 좌측 / 하단 중앙 / 하단 우측"""
def tasks(out_root):
    T = []
    for level, marker in (('L1_synth_blob', False), ('L2_synth_mark', True)):
        for k in range(36):
            # no 'truth' field in the task file an agent reads; the truth cell is recomputed at scoring time (index mod 9)
            T.append({'id': f"{level}_{k:02d}", 'level': level, 'images': [os.path.join(OUTD, f"{level}_{k:02d}.png")],
                      'prompt': PROMPT.format(what='하늘색 원+십자 표식이' if marker else '가장 밝은(고온) 지점이'), 'out': os.path.join(out_root, f"{level}_{k:02d}.json")})
    return T
def truth_of(t): return GRID_KO[int(t['id'][-2:]) % 9]
def norm(ans):
    # longest names first: '중앙' is a substring of '상단 중앙' / '하단 중앙'
    for c in sorted(GRID_KO, key=len, reverse=True):
        if c in str(ans): return c
    return '파싱실패'
def kappa(pairs):
    if not pairs: return float('nan')
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs); po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats); return (po-pe)/(1-pe) if pe < 1 else float('nan')
def score(d):
    T = tasks(d); res = {}
    for level in ('L1_synth_blob', 'L2_synth_mark'):
        pairs = []
        for t in T:
            if t['level'] != level: continue
            fp = os.path.join(d, t['id'] + '.json')
            if not os.path.exists(fp): continue
            pairs.append((norm(json.load(open(fp, encoding='utf-8')).get('raw', '')), truth_of(t)))
        if not pairs: continue
        acc = sum(p == t for p, t in pairs)/len(pairs); k_ = kappa([(p, t) for p, t in pairs if p in GRID_XY])
        res[level] = dict(n=len(pairs), acc=round(acc, 3), kappa=round(k_, 3), parse_fail=sum(p == '파싱실패' for p, _ in pairs), answers=dict(collections.Counter(p for p, _ in pairs)))
        print(level, res[level])
    json.dump(res, open(os.path.join(d, 'probe_scores.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print('chance level = 0.111')
def b64(p): return base64.b64encode(open(p, 'rb').read()).decode('ascii')
def call(model, key, prompt, images, temperature=0):
    parts = [{"inline_data": {"mime_type": "image/png", "data": b64(p)}} for p in images] + [{"text": prompt}]
    gen = {"temperature": temperature}
    if "lite" not in model: gen["thinkingConfig"] = {"thinkingBudget": 0}
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                      json={"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}, timeout=240)
    if r.status_code != 200: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
def run_gemini(model):
    key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY'); assert key
    tag = model.replace('.', '_').replace('-', '_')
    jobs = tasks(os.path.join(OUTD, 'api_' + tag))
    fp = os.path.join(F, 'tasks_factpert.json')
    if os.path.exists(fp):
        for t in json.load(open(fp, encoding='utf-8')):
            jobs.append({'id': t['case_id'], 'images': t['images'], 'prompt': t['prompt'], 'out': os.path.join(F, 'factpert_' + tag, t['case_id'] + '.json')})
    ok = fail = 0
    for t in jobs:
        os.makedirs(os.path.dirname(t['out']), exist_ok=True)
        if os.path.exists(t['out']): ok += 1; continue
        for attempt in range(4):
            try:
                raw = call(model, key, t['prompt'], t['images']); json.dump({'raw': raw, 'model': model}, open(t['out'], 'w', encoding='utf-8'), ensure_ascii=False, indent=1); ok += 1; print(t['id'], 'ok'); break
            except Exception as e:
                msg = repr(e); print(t['id'], 'retry' if attempt < 3 else 'FAILED', msg[:100]); time.sleep(20 if '429' in msg else 5)
        else: fail += 1
        time.sleep(4.2)
    print(f"gemini probe+factpert: {ok} ok, {fail} failed")
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--build', action='store_true'); ap.add_argument('--gemini', default=None); ap.add_argument('--score', default=None); a = ap.parse_args()
    if a.build:
        T = tasks(os.path.join(OUTD, 'frontier')); json.dump(T, open(os.path.join(OUTD, 'tasks_probe.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print('wrote', len(T), 'probe tasks for agents')
    if a.gemini: run_gemini(a.gemini)
    if a.score: score(a.score)
