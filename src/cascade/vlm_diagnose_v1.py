# -*- coding: utf-8 -*-
"""
vlm_diagnose_v1.py — protocol v1 runner: code-owned verdict/severity/numbers + VLM descriptive fields + validation + assembly.
Conditions: full (thermal_full+crop+rgb), no_rgb, facts_only (no images), template (no VLM).
Outputs results/vlm_out/<tag>/<cond>/<case>.json with {raw, vlm_parsed, report(assembled), violations, sec}.
"""
import json, os, sys, io, time, argparse, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import CASES
from protocol_v1 import *
OUT = r"<PROJECT_ROOT>\results\vlm_out"

def parse_json(txt):
    m = re.search(r'\{.*\}', txt, re.S)
    if not m: return None
    s = m.group(0)
    for cand in (s, s.replace('\n', ' '), re.sub(r',\s*}', '}', s)):
        try: return json.loads(cand)
        except Exception: pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v1'); ap.add_argument('--prompt', default=r"<PROJECT_ROOT>\prompts\diagnose_v1.txt")
    ap.add_argument('--cond', default='template,full,no_rgb,facts_only'); ap.add_argument('--cases', default='all')
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct'); ap.add_argument('--max_new', type=int, default=400)
    a = ap.parse_args()
    tmpl = open(a.prompt, encoding='utf-8').read()
    idx = [json.loads(l) for l in open(os.path.join(CASES, 'index.jsonl'), encoding='utf-8')]
    if a.cases != 'all': keep = set(a.cases.split(',')); idx = [i for i in idx if i['case_id'] in keep]
    conds = a.cond.split(',')
    if any(c != 'template' for c in conds):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map='auto')
        proc = AutoProcessor.from_pretrained(a.model)
        def run(content):
            msgs = [{"role": "user", "content": content}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ii, vi = process_vision_info(msgs)
            inp = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad(): g = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False, repetition_penalty=1.05)
            return proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip(), time.time()-t0, int(inp.input_ids.shape[1]), int(g.shape[1]-inp.input_ids.shape[1])
    for cond in conds:
        od = os.path.join(OUT, a.tag, cond); os.makedirs(od, exist_ok=True)
        print(f"\n=== {a.tag}/{cond} ({len(idx)} cases) ===")
        for it in idx:
            cid = it['case_id']; d = os.path.join(CASES, cid); of = os.path.join(od, cid + '.json')
            if os.path.exists(of): continue
            f = json.load(open(os.path.join(d, 'facts.json'), encoding='utf-8'))
            sev = severity(f); ct = case_type(f, sev); cf = confidence(f, sev)
            if cond == 'template':
                fields = template_fields(f, sev, ct); viol = []; raw = None; parsed = None; sec = itok = otok = 0
            else:
                fb = facts_block(perturb_facts(f) if cond == 'full_perturbed' else f, sev, ct, cf)
                prompt = tmpl.format(facts_block=fb, patterns=' / '.join(PATTERNS), causes=' / '.join(CAUSES_BULK if f['equipment'] in BULK else CAUSES_ELEC))
                content = []
                if cond in ('full', 'no_rgb', 'full_perturbed'):
                    content += [{"type": "text", "text": "[이미지 1] 열화상 전체 화면"}, {"type": "image", "image": os.path.join(d, 'thermal_full.png')},
                                {"type": "text", "text": "[이미지 2] 대상 확대 열화상"}, {"type": "image", "image": os.path.join(d, 'thermal_crop.png')}]
                    if cond in ('full', 'full_perturbed') and f['images'].get('rgb_crop'):
                        content += [{"type": "text", "text": "[이미지 3] 가시광 사진"}, {"type": "image", "image": os.path.join(d, 'rgb_crop.png')}]
                content.append({"type": "text", "text": prompt})
                raw, sec, itok, otok = run(content); parsed = parse_json(raw)
                fields, viol = validate(parsed or {}, f, sev, ct)
            rep = assemble(f, fields, viol, sev, ct, cf)
            rec = {'case_id': cid, 'cond': cond, 'raw': raw, 'vlm_parsed': parsed, 'report': rep, 'sec': round(sec, 2), 'in_tok': itok, 'out_tok': otok}
            json.dump(rec, open(of, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(f"  {cid} [{it['stratum'][:14]:14s}] {sec:5.1f}s json={'ok' if (parsed or cond=='template') else 'FAIL'} {rep['verdict']} {rep['severity']} {ct[0]:18s} conf={cf[0]:6s} viol={viol}")

if __name__ == '__main__':
    main()
