# -*- coding: utf-8 -*-
"""build_samevendor_judge_tasks.py — tasks for re-judging the 32 locked cases with the SAME-VENDOR judge (Claude, agent
interface) under the CROSS-VENDOR forced-grounding protocol (INSTRUCTIONS_FULL: two images + two reports, NO facts file,
judge reads the marker itself before comparing). Fills the missing cell of the judge x protocol 2x2:
   same-model judge x original protocol (facts given)  = 21/32   [done 2026-09-17]
   cross-vendor judge x forced-grounding (no facts)      = 27/32   [done 2026-10-16]
   same-model judge x forced-grounding (no facts)        = THIS RUN
   cross-vendor judge x original protocol                = not run (Gemini cannot read the facts file through tools)
usage: python build_samevendor_judge_tasks.py [--v2_dir DIR --tag TAG]   (default: image-based frontier arm, tag samevendor_full)
Writes results/judge_<tag>/tasks.json; agents write results/judge_<tag>/<case>.json.
"""
import json, os, sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import judge_crossfamily as JC
ap = argparse.ArgumentParser(); ap.add_argument('--v2_dir', default=None); ap.add_argument('--tag', default='samevendor_full'); ap.add_argument('--key', default=None)
ap.add_argument('--schema', default='full', choices=['simple', 'full', 'full2']); ap.add_argument('--strip_tells', action='store_true'); ap.add_argument('--facts', action='store_true')
a = ap.parse_args()
if a.v2_dir: JC.set_v2_dir(a.v2_dir)
JC.STRIP_TELLS = a.strip_tells; JC.FACTS_INLINE = a.facts
key = JC.load_key(a.key); cases = sorted(key.keys())
od = os.path.join(JC.R, 'judge_' + a.tag); os.makedirs(od, exist_ok=True)
tasks = []
for cid in cases:
    imgs = [JC.img_path(cid, n) for n in ('thermal_crop_v2.png', 'rgb_crop_v2.png') if os.path.exists(JC.img_path(cid, n))]
    tasks.append({'case_id': cid, 'images': imgs, 'prompt': JC.build_prompt(cid, key, schema=a.schema), 'out': os.path.join(od, cid + '.json')})
json.dump(tasks, open(os.path.join(od, 'tasks.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f"wrote {len(tasks)} judge tasks -> {os.path.join(od, 'tasks.json')} (v2 arm dir: {JC.V2_DIR})")
