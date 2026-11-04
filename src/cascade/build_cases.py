# -*- coding: utf-8 -*-
"""
build_cases.py — stratified evaluation case set for the grounded-VLM diagnosis stage.
For each selected bbox: fixed-scale full-frame render (+colorbar, all bboxes), thermal crop, paired RGB crop,
and a facts.json holding every number the VLM is allowed to use (RF verdict/prob, dT_scene, dT_sib, abs temps,
equipment, spec, calib, hotspot location) plus hidden ground truth (label) for grading.
Strata (within-test unless noted; max 1 bbox per session per stratum):
  A clear_danger      label=danger, dT_scene>=8, RF>=0.5
  B hot_but_normal    label=normal, dT_scene>6,  RF<0.5     (rule FP that the gate correctly rejected)
  C borderline        4<=dT_scene<=8, any label
  D subtle_fault      label=danger, dT_scene<4                (no thermal evidence in single image)
  E cool_normal       label=normal, dT_scene<3,  RF<0.5
  F rf_fp             label=normal, RF>=0.5
  G rf_fn             label=danger, dT_scene>=5.5, RF<0.5
  H unseen_equip      gen_test, label=danger & normal, RF any
"""
import json, os, sys, io, random, collections
import numpy as np
from multiprocessing import Pool
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import *

import argparse
PER = 10; SEED = 0; OUTDIR = CASES; EXCLUDE = set()
def rec_bboxes(rec):
    T = parse_csv(rec['csv'])
    if T is None: return []
    anns = [a for a in rec['annotations'] if a.get('bbox') and a.get('status') in ('danger', 'normal')]
    out = []
    for i, feat, ex in frame_feats_idx(T, anns):
        out.append(dict(frame_key=rec['frame_key'], ai=i, feat=feat, ex=ex, label=anns[i]['status']))
    return out

def stratum(label, dT, p, grp):
    if grp == 'gen_test': return 'H_unseen_equip'
    if label == 'danger' and dT >= 8 and p >= 0.5: return 'A_clear_danger'
    if label == 'normal' and dT > 6 and p < 0.5: return 'B_hot_but_normal'
    if label == 'danger' and dT < 4: return 'D_subtle_fault'
    if label == 'normal' and dT < 3 and p < 0.5: return 'E_cool_normal'
    if label == 'normal' and p >= 0.5: return 'F_rf_fp'
    if label == 'danger' and dT >= 5.5 and p < 0.5: return 'G_rf_fn'
    if 4 <= dT <= 8: return 'C_borderline'
    return None

def band(dT):
    """placeholder severity band from ΔT-over-reference; final table comes from the protocol spec."""
    if dT < 4: return 'L0_none'
    if dT < 10: return 'L1_low'
    if dT < 15: return 'L2_medium'
    if dT < 40: return 'L3_high'
    return 'L4_critical'

def main():
    global PER, SEED, OUTDIR, EXCLUDE
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--per', type=int, default=10); ap.add_argument('--out', default=CASES); ap.add_argument('--exclude_index', default=None); ap.add_argument('--fps', type=int, default=3); ap.add_argument('--faccap', type=int, default=3)
    a = ap.parse_args(); PER, SEED, OUTDIR = a.per, a.seed, a.out
    if a.exclude_index:
        ex = [json.loads(l) for l in open(a.exclude_index, encoding='utf-8')]
        EXCLUDE = set()
        for e in ex:
            EXCLUDE.add(json.load(open(os.path.join(os.path.dirname(a.exclude_index), e['case_id'], 'facts.json'), encoding='utf-8'))['session'])
        print('excluding sessions:', len(EXCLUDE))
    random.seed(SEED)
    recs = [json.loads(l) for l in open(MAN, encoding='utf-8')]
    byk = {r['frame_key']: r for r in recs}
    pool = [r for r in recs if r.get('csv') and r['split_group'] in ('test', 'gen_test') and r['session'] not in EXCLUDE]
    random.shuffle(pool)
    # limit CSV parsing: sample up to 3 frames per session
    per_sess = collections.defaultdict(list)
    for r in pool: per_sess[r['session']].append(r)
    sample = [r for s in per_sess for r in per_sess[s][:a.fps]]
    print(f"frames to parse: {len(sample)} from {len(per_sess)} sessions")
    rf, vocab = load_rf()
    rows = []
    with Pool(8) as p:
        for k, res in enumerate(p.imap(rec_bboxes, sample, chunksize=16)):
            rows += res
            if (k+1) % 400 == 0: print(f"  {k+1}/{len(sample)}")
    facs = [byk[r['frame_key']]['facility'] for r in rows]
    probs = rf_prob(rf, vocab, [r['feat'] for r in rows], facs)
    di = FEAT_NAMES.index('dT_scene_max')
    buckets = collections.defaultdict(list)
    for r, pr in zip(rows, probs):
        rec = byk[r['frame_key']]; dT = r['feat'][di]
        s = stratum(r['label'], dT, pr, rec['split_group'])
        if s: r['prob'] = float(pr); r['stratum'] = s; buckets[s].append(r)
    print("bucket sizes:", {k: len(v) for k, v in sorted(buckets.items())})
    # pick PER per stratum, max 1 per session, spread equipment
    chosen = []
    for s, lst in sorted(buckets.items()):
        random.shuffle(lst); used = set(); fac_cnt = collections.Counter()
        lst.sort(key=lambda r: fac_cnt[byk[r['frame_key']]['facility']])
        for r in lst:
            rec = byk[r['frame_key']]
            if rec['session'] in used or fac_cnt[rec['facility']] >= a.faccap: continue
            if s == 'H_unseen_equip' and sum(1 for c in chosen if c['stratum'] == s and c['label'] == r['label']) >= PER//2: continue
            used.add(rec['session']); fac_cnt[rec['facility']] += 1; chosen.append(r)
            if len([c for c in chosen if c['stratum'] == s]) >= PER: break
    print("chosen:", collections.Counter(c['stratum'] for c in chosen))
    os.makedirs(OUTDIR, exist_ok=True); index = []
    for n, r in enumerate(chosen):
        rec = byk[r['frame_key']]; T = parse_csv(rec['csv'])
        anns = [a for a in rec['annotations'] if a.get('bbox') and a.get('status') in ('danger', 'normal')]
        a = anns[r['ai']]; b = a['bbox']; cid = f"case{n:03d}"; d = os.path.join(OUTDIR, cid); os.makedirs(d, exist_ok=True)
        f = dict(zip(FEAT_NAMES, [float(v) for v in r['feat']])); ex = r['ex']
        px, py = ex['peak_xy_in_roi']; roi, (x, y, x2, y2) = clip_roi(T, b)
        render_full(T, anns, r['ai'], os.path.join(d, 'thermal_full.png'))
        box = render_crop(T, b, os.path.join(d, 'thermal_crop.png'), mark=(x+px, y+py))
        has_rgb = render_rgb_crop(imread_u(rec.get('rgb')), box, b, os.path.join(d, 'rgb_crop.png')) if box else False
        H, W = T.shape; fmax = float(T.max()); fy, fx = np.unravel_index(int(np.argmax(T)), T.shape)
        pos_in_frame = pos_words((b['x']+b['width']/2)/W, (b['y']+b['height']/2)/H)
        hs_pos = pos_words(px/max(b['width'],1), py/max(b['height'],1))
        facts = {
            'case_id': cid, 'frame_key': rec['frame_key'], 'session': rec['session'], 'split_group': rec['split_group'],
            'stratum': r['stratum'],
            'equipment': rec['facility'], 'component_class': a.get('class'), 'spec': a.get('standard'), 'site': rec['region'],
            'n_components_in_frame': len(anns), 'target_index': r['ai'],
            'bbox': b, 'bbox_position_in_frame': pos_in_frame,
            'calib': rec.get('calib'),
            'thermal': {
                't_max': round(f['t_max'], 1), 't_mean': round(f['t_mean'], 1), 't_min': round(ex['t_min'], 1),
                't_p95': round(f['t_p95'], 1), 't_std': round(f['t_std'], 2),
                'scene_median': round(f['scene_med'], 1), 'scene_p95': round(f['scene_p95'], 1), 'scene_max': round(fmax, 1),
                'frame_max_inside_target': bool(y <= fy < y2 and x <= fx < x2),
                'dT_scene_max': round(f['dT_scene_max'], 1), 'dT_scene_mean': round(f['dT_scene_mean'], 1),
                'n_siblings': ex['n_sib'], 'sibling_t_max': ex['sib_tmax'], 'sibling_reference': round(ex['sib_ref'], 1) if ex['n_sib'] else None,
                'dT_sibling_max': round(f['dT_sib_max'], 1) if ex['n_sib'] else None,
                'hotspot_area_frac': round(f['hot_frac'], 3), 'n_hotspots': int(f['n_hot']), 'hotspot_compactness': round(f['compact'], 2),
                'hotspot_position_in_target': hs_pos, 'peak_z': round(f['peak_z'], 2),
                'ambient_temp': rec.get('calib', {}).get('atmospheric_temp'),
                'dT_over_ambient': round(f['t_max'] - rec['calib']['atmospheric_temp'], 1) if rec.get('calib') else None,
            },
            'gate': {'rf_prob': round(r['prob'], 3), 'rf_verdict': 'flag' if r['prob'] >= 0.5 else 'pass',
                     'rule_verdict': 'flag' if f['dT_scene_max'] > 5.5 else 'pass', 'band_placeholder': band(f['dT_scene_max'])},
            'images': {'thermal_full': 'thermal_full.png', 'thermal_crop': 'thermal_crop.png', 'rgb_crop': 'rgb_crop.png' if has_rgb else None},
            'hidden': {'label': r['label']},
        }
        json.dump(facts, open(os.path.join(d, 'facts.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        index.append({'case_id': cid, 'stratum': r['stratum'], 'equipment': rec['facility'], 'label': r['label'],
                      'dT_scene_max': facts['thermal']['dT_scene_max'], 'rf_prob': facts['gate']['rf_prob'], 'has_rgb': has_rgb})
    with open(os.path.join(OUTDIR, 'index.jsonl'), 'w', encoding='utf-8') as fo:
        for it in index: fo.write(json.dumps(it, ensure_ascii=False) + '\n')
    print(f"\nwrote {len(index)} cases to {OUTDIR}")
    for s in sorted(set(i['stratum'] for i in index)):
        its = [i for i in index if i['stratum'] == s]
        print(f"  {s:18s} n={len(its):2d}  equip={sorted(set(i['equipment'] for i in its))}")

if __name__ == '__main__':
    main()
