# -*- coding: utf-8 -*-
"""gemini_noimg.py — NO-IMAGE (facts-only) arm written by the LOW-COST commercial model (Gemini Flash-Lite), text only.
Same 80 prompts as the frontier no-image control (results/vlm_out/frontier/tasks_noimg.json, built by build_noimg_tasks.py):
the model fills the descriptive fields from the band-worded, number-free facts alone; no image is attached. Tests whether
tier 1 of the deployment recipe (facts-to-narrative without the image) needs a frontier model at all.
  raw -> results/vlm_out/subject_gemini_3_5_flash_lite/noimg/<case>.json  ({"raw": ..., "model": ...})
  then: python assemble_noimg.py --src subject_gemini_3_5_flash_lite --out v2gemini
usage: GOOGLE_API_KEY=... python gemini_noimg.py [--model gemini-3.5-flash-lite]
"""
import json, os, sys, time, argparse
import probe_api as PA
_keep = sys.stdout
F = r"<PROJECT_ROOT>\results\vlm_out\frontier"
ap = argparse.ArgumentParser(); ap.add_argument('--model', default='gemini-3.5-flash-lite'); a = ap.parse_args()
S = os.path.join(r"<PROJECT_ROOT>\results\vlm_out", 'subject_' + a.model.replace('.', '_').replace('-', '_'), 'noimg'); os.makedirs(S, exist_ok=True)
key = os.environ.get('GOOGLE_API_KEY'); assert key
tasks = json.load(open(os.path.join(F, 'tasks_noimg.json'), encoding='utf-8')); ok = fail = skip = 0
for t in tasks:
    assert t['images'] == []
    out = os.path.join(S, t['case_id'] + '.json')
    if os.path.exists(out): skip += 1; continue
    for attempt in range(4):
        try:
            raw = PA.call(a.model, key, t['prompt'], []); json.dump({'raw': raw, 'model': a.model, 'prompt': t['prompt']}, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); ok += 1; print(t['case_id'], 'ok'); break
        except Exception as e:
            msg = repr(e); print(t['case_id'], 'retry' if attempt < 3 else 'FAILED', msg[:100]); time.sleep(30 if '429' in msg else 5)
    else: fail += 1
    time.sleep(4.2)
print(f"gemini no-image: {ok} ok, {skip} existing, {fail} failed -> {S}")
