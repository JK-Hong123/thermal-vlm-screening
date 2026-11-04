# -*- coding: utf-8 -*-
"""aggregate_judge_frontier.py — aggregate the blinded frontier-VLM vs A0+ judgments (results/judge_frontier/*.json)
with case-clustered bootstrap CIs; writes results/judge_frontier_summary.txt"""
import json, os, sys, io, glob, collections, statistics as st
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
R = r"<PROJECT_ROOT>\results"; D = os.path.join(R, 'judge_frontier'); key = json.load(open(os.path.join(D, '_key.json'), encoding='utf-8'))
idx = {json.loads(l)['case_id']: json.loads(l) for l in open(os.path.join(R, 'cases_lock', 'index.jsonl'), encoding='utf-8')}
J = []
for fp in sorted(glob.glob(os.path.join(D, 'case*.json'))):
    try: j = json.load(open(fp, encoding='utf-8'))
    except Exception as e: print('bad json', fp, e); continue
    cid = os.path.basename(fp)[:-5]; j['case_id'] = cid; j['A'] = key[cid]['A']; j['B'] = key[cid]['B']; J.append(j)
n = len(J); rng = np.random.default_rng(0)
def arm(j): return 'tie' if j['preferred_report'] == 'tie' else (j['A'] if j['preferred_report'] == 'A' else j['B'])
def g(j, k, a): return j[k + '_A_1to5'] if j['A'] == a else j[k + '_B_1to5']
def boot(stat):
    v = [stat([J[i] for i in rng.integers(0, n, n)]) for _ in range(2000)]; return np.percentile(v, 2.5), np.percentile(v, 97.5)
pv = lambda items: np.mean([arm(j) == 'v2' for j in items]); av = lambda items: np.mean([j['vlm_adds_value_over_a0plus'] == 'yes' for j in items])
du = lambda items: np.mean([g(j, 'usefulness', 'a0plus') - g(j, 'usefulness', 'v2') for j in items])
out = [f"=== 프런티어 VLM(Claude, 블라인드 관찰+추론) 보고서 vs 코드 전용(A0+) — 잠금셋 블라인드 판정 ({n}케이스) ==="]
lo, hi = boot(pv); out.append(f"선호: VLM {sum(arm(j)=='v2' for j in J)}/{n} = {pv(J):.3f} [{lo:.3f},{hi:.3f}] | A0+ {sum(arm(j)=='a0plus' for j in J)} | tie {sum(arm(j)=='tie' for j in J)}")
lo, hi = boot(av); out.append(f"★ VLM이 코드 전용에 없는 유용 정보를 더함: {sum(j['vlm_adds_value_over_a0plus']=='yes' for j in J)}/{n} = {av(J):.3f} [{lo:.3f},{hi:.3f}]")
lo, hi = boot(du); out.append(f"유용성: VLM {st.mean([g(j,'usefulness','v2') for j in J]):.2f} vs A0+ {st.mean([g(j,'usefulness','a0plus') for j in J]):.2f} (차이 A0+−VLM {du(J):+.2f} [{lo:+.2f},{hi:+.2f}])")
out.append(f"실행가능성: VLM {st.mean([g(j,'actionability','v2') for j in J]):.2f} vs A0+ {st.mean([g(j,'actionability','a0plus') for j in J]):.2f}")
c = collections.Counter(j['vlm_grid_correct'] for j in J); out.append(f"VLM 열점 위치(판정자 픽셀 실측 대비): {dict(c)}")
for k, nm in (('vlm_part_correct', 'VLM 부위'), ('a0plus_part_correct', 'A0+ 부위')):
    c = collections.Counter(j[k] for j in J); d = c['yes'] + c['partial'] + c['no']
    out.append(f"{nm}: yes {c['yes']} partial {c['partial']} no {c['no']} 불가 {c['undeterminable']} → 정답 {c['yes']/max(d,1):.2f}, 부분포함 {(c['yes']+c['partial'])/max(d,1):.2f}")
out.append(f"VLM 육안 플래그 타당성: {dict(collections.Counter(j['vlm_visual_flags_valid'] for j in J))}")
out.append(f"근거 없는 주장(케이스당): VLM {st.mean([float(j['n_vlm_ungrounded']) for j in J]):.1f} vs A0+ {st.mean([float(j['n_a0plus_ungrounded']) for j in J]):.1f}")
out.append(f"현장 배포 가능(운영자 관점): {dict(collections.Counter(j['report_trustworthy_for_operator'] for j in J))}")
out.append(""); out.append("층별 (VLM 선호 / 가치 추가):")
for s in sorted(set(idx[j['case_id']]['stratum'] for j in J)):
    js = [j for j in J if idx[j['case_id']]['stratum'] == s]
    out.append(f"  {s:18s} n={len(js):2d} | VLM 선호 {sum(arm(j)=='v2' for j in js)} | 가치 추가 {sum(j['vlm_adds_value_over_a0plus']=='yes' for j in js)}")
out.append(""); out.append("가치를 더했다고 판정된 케이스:")
for j in J:
    if j['vlm_adds_value_over_a0plus'] == 'yes': out.append(f"  [{j['case_id']}] {str(j['value_added_what'])[:220]}")
out.append(""); out.append("비교(같은 잠금셋): 3B VLM 선호 0/34, 가치추가 —; 3B+LoRA 선호 0/32, 가치추가 0/32")
txt = "\n".join(out); open(os.path.join(R, 'judge_frontier_summary.txt'), 'w', encoding='utf-8').write(txt); print(txt)
