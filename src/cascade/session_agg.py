# -*- coding: utf-8 -*-
"""session_agg.py — session-level (multi-frame) facts for each case: the CCTV session has ~40-80 near-static frames.
For the case's target bbox, find the matching component in every other frame of the same session (same class, IoU>0.5),
run the RF gate on each, and aggregate: k/N frames flagged, prob median, dT range, hotspot-position stability, T_max max.
Also adds 'norms' block (percentile vs normal population of the same equipment, from equipment_norms.json).
Writes both blocks INTO each case's facts.json (keys 'session_agg', 'norms')."""
import json, os, sys, io, collections
import numpy as np
from multiprocessing import Pool
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import *
NORMS = json.load(open(r"<PROJECT_ROOT>\results\equipment_norms.json", encoding='utf-8'))

def iou(a, b):
    ax2, ay2 = a['x']+a['width'], a['y']+a['height']; bx2, by2 = b['x']+b['width'], b['y']+b['height']
    iw = max(0, min(ax2, bx2)-max(a['x'], b['x'])); ih = max(0, min(ay2, by2)-max(a['y'], b['y']))
    inter = iw*ih; u = a['width']*a['height'] + b['width']*b['height'] - inter
    return inter/u if u > 0 else 0

def frame_job(args):
    rec, tb, tcls = args
    T = parse_csv(rec['csv'])
    if T is None: return None
    anns = [a for a in rec['annotations'] if a.get('bbox') and a.get('status') in ('danger', 'normal')]
    best = None; bi = -1
    for i, a in enumerate(anns):
        if a.get('class') != tcls: continue
        v = iou(a['bbox'], tb)
        if v > 0.5 and (best is None or v > best): best, bi = v, i
    if bi < 0: return {'frame_key': rec['frame_key'], 'matched': False}
    for i, feat, ex in frame_feats_idx(T, anns):
        if i == bi:
            b = anns[i]['bbox']; px, py = ex['peak_xy_in_roi']
            return {'frame_key': rec['frame_key'], 'matched': True, 'feat': feat, 'facility': rec['facility'],
                    'hs_pos': pos_words(px/max(b['width'], 1), py/max(b['height'], 1))}
    return {'frame_key': rec['frame_key'], 'matched': False}

def pct_of(sorted_vals, v):
    a = np.asarray(sorted_vals); return round(100*float((a < v).mean()), 1) if len(a) else None

def main():
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument('--cases_dir', default=CASES); a = ap.parse_args(); CD = a.cases_dir
    recs = [json.loads(l) for l in open(MAN, encoding='utf-8')]
    by_sess = collections.defaultdict(list)
    for r in recs:
        if r.get('csv'): by_sess[r['session']].append(r)
    rf, vocab = load_rf(); di = FEAT_NAMES.index('dT_scene_max'); ti = FEAT_NAMES.index('t_max'); si = FEAT_NAMES.index('dT_sib_max')
    idx = [json.loads(l) for l in open(os.path.join(CD, 'index.jsonl'), encoding='utf-8')]
    pool = Pool(8)
    for it in idx:
        cid = it['case_id']; fp = os.path.join(CD, cid, 'facts.json'); f = json.load(open(fp, encoding='utf-8'))
        frames = [r for r in by_sess[f['session']] if r['frame_key'] != f['frame_key']]
        jobs = [(r, f['bbox'], f['component_class']) for r in frames]
        res = [x for x in pool.imap(frame_job, jobs, chunksize=8) if x]
        matched = [x for x in res if x['matched']]
        agg = {'n_frames_in_session': len(res)+1, 'n_frames_component_matched': len(matched)}
        if matched:
            P = rf_prob(rf, vocab, [m['feat'] for m in matched], [m['facility'] for m in matched])
            dT = np.array([m['feat'][di] for m in matched]); tm = np.array([m['feat'][ti] for m in matched])
            hs = collections.Counter(m['hs_pos'] for m in matched)
            agg.update({
                'n_flagged': int((P >= 0.5).sum()), 'flag_frac': round(float((P >= 0.5).mean()), 2),
                'rf_prob_median': round(float(np.median(P)), 3), 'rf_prob_min': round(float(P.min()), 3), 'rf_prob_max': round(float(P.max()), 3),
                'dT_scene_median': round(float(np.median(dT)), 1), 'dT_scene_min': round(float(dT.min()), 1), 'dT_scene_max_over_session': round(float(dT.max()), 1),
                't_max_median': round(float(np.median(tm)), 1), 't_max_max': round(float(tm.max()), 1),
                'hotspot_pos_mode': hs.most_common(1)[0][0], 'hotspot_pos_stability': round(hs.most_common(1)[0][1]/len(matched), 2),
                'hotspot_pos_same_as_this_frame': round(hs[f['thermal']['hotspot_position_in_target']]/len(matched), 2),
            })
        f['session_agg'] = agg
        # norms vs normal population of same equipment (train)
        nm = NORMS.get(f['equipment']); t = f['thermal']
        if nm:
            f['norms'] = {
                'equipment_seen_in_training': True, 'n_normal_train': nm['n_normal'],
                't_max_pct_vs_normal': pct_of(nm['t_max']['normal_sorted'], t['t_max']),
                'dT_scene_pct_vs_normal': pct_of(nm['dT_scene_max']['normal_sorted'], t['dT_scene_max']),
                'dT_sib_pct_vs_normal': pct_of(nm['dT_sib_max']['normal_sorted'], t['dT_sibling_max']) if t['dT_sibling_max'] is not None else None,
                'normal_t_max_p50': nm['t_max']['normal_q']['50'], 'normal_t_max_p95': nm['t_max']['normal_q']['95'], 'normal_t_max_p99': nm['t_max']['normal_q']['99'],
                'normal_dT_scene_p50': nm['dT_scene_max']['normal_q']['50'], 'normal_dT_scene_p95': nm['dT_scene_max']['normal_q']['95'], 'normal_dT_scene_p99': nm['dT_scene_max']['normal_q']['99'],
                'danger_t_max_p50': nm['t_max']['danger_q']['50'] if nm['t_max']['danger_q'] else None,
                'danger_dT_scene_p50': nm['dT_scene_max']['danger_q']['50'] if nm['dT_scene_max']['danger_q'] else None,
            }
        else:
            f['norms'] = {'equipment_seen_in_training': False}
        json.dump(f, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        a = agg; print(f"  {cid} [{it['stratum'][:14]:14s}] frames {a['n_frames_in_session']:3d} matched {a['n_frames_component_matched']:3d} flagged {a.get('n_flagged','-')!s:>3s} ({a.get('flag_frac','-')}) dTmed {a.get('dT_scene_median','-')} hsStab {a.get('hotspot_pos_stability','-')} | tmax pct {f['norms'].get('t_max_pct_vs_normal')}")
    pool.close()

if __name__ == '__main__':
    main()
