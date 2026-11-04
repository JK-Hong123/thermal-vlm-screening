# -*- coding: utf-8 -*-
"""
vlm_diagnose_v2.py — GTD-v2 two-pass runner.
conditions:
  v2            pass-1 blind (thermal_full + thermal_crop_v2 [+ rgb_crop_v2]) -> cross-check -> pass-2 (number-free facts + observations)
  v2_norgb      same without the RGB image
  v2_markerpert pass-1 only, marker drawn in a wrong cell (does p_grid follow the marker? = reads the image)
  v2_nomarker   pass-1 only, no marker (does it locate the hotspot from the colour map alone?)
  v2_pass2only  pass-2 with template observations (no images at all) -> facts-only control
  template      no VLM
usage: python vlm_diagnose_v2.py --tag v2 --cond template,v2,v2_markerpert,v2_nomarker,v2_norgb,v2_pass2only [--cases_dir DIR] [--cases a,b]
"""
import json, os, sys, io, time, argparse, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import CASES
import protocol_v2 as P
OUT = r"<PROJECT_ROOT>\results\vlm_out"
P1T = r"<PROJECT_ROOT>\prompts\pass1_blind_v2.txt"; P2T = r"<PROJECT_ROOT>\prompts\pass2_reason_v2.txt"

def parse_json(txt):
    m = re.search(r'\{.*\}', txt, re.S)
    if not m: return None
    s = m.group(0)
    for cand in (s, s.replace('\n', ' '), re.sub(r',\s*}', '}', s), re.sub(r',\s*]', ']', s)):
        try: return json.loads(cand)
        except Exception: pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v2'); ap.add_argument('--cond', default='template,v2,v2_markerpert,v2_nomarker,v2_norgb,v2_pass2only')
    ap.add_argument('--cases_dir', default=CASES); ap.add_argument('--cases', default='all')
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct'); ap.add_argument('--quant', default=None, choices=[None, '4bit', '8bit']); ap.add_argument('--adapter', default=None); ap.add_argument('--max1', type=int, default=200); ap.add_argument('--max2', type=int, default=180)
    a = ap.parse_args()
    t1 = open(P1T, encoding='utf-8').read(); t2 = open(P2T, encoding='utf-8').read()
    idx = [json.loads(l) for l in open(os.path.join(a.cases_dir, 'index.jsonl'), encoding='utf-8')]
    if a.cases != 'all': keep = set(a.cases.split(',')); idx = [i for i in idx if i['case_id'] in keep]
    conds = a.cond.split(',')
    if any(c not in ('template', 'a0plus') for c in conds):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        kw = dict(device_map='auto')
        if a.quant:
            from transformers import BitsAndBytesConfig
            kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4') if a.quant == '4bit' else BitsAndBytesConfig(load_in_8bit=True)
        else: kw['torch_dtype'] = torch.bfloat16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, **kw)
        if a.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter); print('adapter loaded:', a.adapter)
        proc = AutoProcessor.from_pretrained(a.model)
        def run(content, max_new):
            msgs = [{"role": "user", "content": content}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ii, vi = process_vision_info(msgs)
            inp = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad(): g = model.generate(**inp, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05)
            return proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip(), round(time.time()-t0, 2)
    for cond in conds:
        od = os.path.join(OUT, a.tag, cond); os.makedirs(od, exist_ok=True)
        print(f"\n=== {a.tag}/{cond} ({len(idx)} cases) ===")
        for it in idx:
            cid = it['case_id']; d = os.path.join(a.cases_dir, cid); of = os.path.join(od, cid + '.json')
            if os.path.exists(of): continue
            f = json.load(open(os.path.join(d, 'facts.json'), encoding='utf-8'))
            sev = P.severity(f); ct = P.case_type(f, sev); qf = P.quality_flags(f, sev); f['_qf'] = qf; conf = P.confidence(f, sev, qf)
            raw = {}; secs = {}; viol = []
            # ---- pass 1 ----
            if cond in ('v2', 'v2_norgb', 'v2_markerpert', 'v2_nomarker'):
                crop = {'v2': 'thermal_crop_v2', 'v2_norgb': 'thermal_crop_v2', 'v2_markerpert': 'thermal_crop_v2_pert', 'v2_nomarker': 'thermal_crop_v2_nomark'}[cond]
                use_rgb = cond in ('v2', 'v2_markerpert', 'v2_nomarker') and f['images'].get('rgb_crop_v2')
                parts = ' / '.join(P.PARTS.get(f['equipment'], []) + ['확인 불가'])
                rgb_line = "이미지 3: 같은 위치의 가시광 사진(참고용; 열화상과 정렬은 근사, 상자 없음)." if use_rgb else "가시광 사진 없음."
                p1 = t1.format(equipment=f['equipment'], parts=parts, rgb_line=rgb_line)
                if cond == 'v2_nomarker': p1 = p1.replace("**하늘색 원+십자 표식(마커)** 이 대상 내 최고온 지점입니다.", "마커는 없습니다. 색이 가장 밝은(고온) 지점을 스스로 찾으십시오.").replace('마커가', '최고온 지점이').replace('마커', '최고온 지점')
                content = [{"type": "text", "text": "[이미지 1]"}, {"type": "image", "image": os.path.join(d, 'thermal_full.png')},
                           {"type": "text", "text": "[이미지 2]"}, {"type": "image", "image": os.path.join(d, crop + '.png')}]
                if use_rgb: content += [{"type": "text", "text": "[이미지 3]"}, {"type": "image", "image": os.path.join(d, 'rgb_crop_v2.png')}]
                content.append({"type": "text", "text": p1})
                raw['pass1'], secs['pass1'] = run(content, a.max1)
                obs, v1 = P.validate_pass1(parse_json(raw['pass1']), f); viol += v1
            elif cond == 'a0plus':
                obs = None
            else:
                obs = {'equipment_plausible': '확인 불가', 'part': '확인 불가', 'p_grid': '확인 불가', 'heat_pattern': '불명', 'visual_flags': ['특이 없음'], 'image_quality': '보통', 'part_location_ko': ''}
            if cond == 'a0plus':
                cc0 = P.cross_check({'p_grid': '확인 불가', 'heat_pattern': '불명'}, f)
                obs, p2f = P.a0plus_fields(f, sev, ct, cc0)
            cc = P.cross_check(obs, f)
            # ---- pass 2 ----
            if cond == 'a0plus':
                pass
            elif cond in ('v2', 'v2_norgb', 'v2_pass2only'):
                fv = P.facts_vlm(f, sev, ct, conf, obs)
                causes = ' / '.join(P.CAUSES_BULK if f['equipment'] in P.BULK else P.CAUSES_ELEC)
                p2 = t2.format(facts=fv, causes=causes, checks=' / '.join(P.CHECKS))
                raw['pass2'], secs['pass2'] = run([{"type": "text", "text": p2}], a.max2)
                p2f, v2 = P.validate_pass2(parse_json(raw['pass2']), f, sev, ct, obs, cc); viol += v2
            else:
                p2f = {'cause_hypothesis': ['확인 불가'], 'cause_basis': [], 'field_checks': [], 'operator_note_ko': ''}
            rep = P.assemble(f, sev, ct, conf, qf, obs, cc, p2f, viol, raw)
            rec = {'case_id': cid, 'cond': cond, 'raw': raw, 'secs': secs, 'report': rep}
            json.dump(rec, open(of, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(f"  {cid} [{it['stratum'][:14]:14s}] {sum(secs.values()):5.1f}s {rep['verdict']} {rep['severity']} {ct[0]:16s} grid={obs['p_grid']}/{cc['p_grid_truth']}:{cc['spatial_check']:8s} pat={obs['heat_pattern']}/{cc['pattern_code']} part={obs['part'][:12]} viol={viol}")

if __name__ == '__main__':
    main()
