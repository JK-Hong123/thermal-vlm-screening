# -*- coding: utf-8 -*-
"""
build_human_eval_kit.py — blinded A/B evaluation kit for field engineers (paper: human usefulness study).
30 stratified locked cases; per case: images (full thermal, zoomed thermal, visible), report A and report B
(randomised assignment of code-only A0+ vs best VLM-augmented v2lora), a Korean instruction sheet, a scoring CSV,
and a SEPARATE hidden answer key (which letter was which arm). Output: results/human_eval_kit/
"""
import json, os, sys, io, random, shutil, csv, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
L = r"<PROJECT_ROOT>\results\cases_lock"; V = r"<PROJECT_ROOT>\results\vlm_out\v2lora"
K = r"<PROJECT_ROOT>\results\human_eval_kit"; ARMS = {'code': 'a0plus', 'vlm': 'v2'}
random.seed(2026)
idx = [json.loads(l) for l in open(os.path.join(L, 'index.jsonl'), encoding='utf-8')]
by_s = collections.defaultdict(list)
for it in idx: by_s[it['stratum']].append(it)
quota = {'A_clear_danger': 5, 'B_hot_but_normal': 5, 'C_borderline': 4, 'D_subtle_fault': 2, 'E_cool_normal': 4, 'F_rf_fp': 5, 'G_rf_fn': 2, 'H_unseen_equip': 3}
chosen = []
for s, q in quota.items():
    pool = by_s.get(s, []); random.shuffle(pool); chosen += pool[:q]
random.shuffle(chosen)
os.makedirs(K, exist_ok=True); key = []; sheet = []
def strip_validation(txt):   # evaluators should not see the machine-validation line or arm-revealing labels
    return "\n".join(l for l in txt.splitlines() if not l.startswith('■ 기계 검증'))
for n, it in enumerate(chosen, 1):
    cid = it['case_id']; d = os.path.join(K, f"case{n:02d}"); os.makedirs(d, exist_ok=True)
    for src, dst in (('thermal_full.png', '1_열화상_전체.png'), ('thermal_crop_v2.png', '2_열화상_확대.png'), ('rgb_crop_v2.png', '3_가시광.png')):
        p = os.path.join(L, cid, src)
        if os.path.exists(p): shutil.copy(p, os.path.join(d, dst))
    swap = random.random() < 0.5
    a_arm, b_arm = ('vlm', 'code') if swap else ('code', 'vlm')
    for letter, arm in (('A', a_arm), ('B', b_arm)):
        txt = open(os.path.join(V, ARMS[arm], cid + '.txt'), encoding='utf-8').read()
        open(os.path.join(d, f"보고서_{letter}.txt"), 'w', encoding='utf-8').write(strip_validation(txt))
    key.append(dict(kit_case=f"case{n:02d}", case_id=cid, stratum=it['stratum'], equipment=it['equipment'], A=a_arm, B=b_arm, hidden_label=it['label']))
    sheet.append([f"case{n:02d}", it['equipment'], '', '', '', '', '', '', ''])
with open(os.path.join(K, '채점표.csv'), 'w', encoding='utf-8-sig', newline='') as fo:
    w = csv.writer(fo); w.writerow(['케이스', '설비', '더 유용한 보고서(A/B/동일)', 'A 유용성(1-5)', 'B 유용성(1-5)', 'A 실행가능성(1-5)', 'B 실행가능성(1-5)', '현장에 그대로 배포 가능한 쪽(A/B/둘다/없음)', '의견(자유)'])
    w.writerows(sheet)
json.dump(key, open(os.path.join(K, '_answer_key_DO_NOT_SHARE.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
open(os.path.join(K, '안내문.txt'), 'w', encoding='utf-8').write("""열화상 진단 보고서 블라인드 평가 안내

목적: 산업설비 열화상 CCTV 한 장면에 대해 자동 생성된 두 종류의 진단 보고서(A, B) 중 어느 쪽이 현장 유지보수 관점에서 더 유용한지 평가합니다.
두 보고서는 같은 계측 수치와 판정·등급을 공유하고, "이미지 판독 / 원인 후보 / 현장 확인 항목 / 작업자 메모" 부분만 다릅니다. 어느 쪽이 어떤 방식으로 만들어졌는지는 알려드리지 않습니다(케이스마다 A/B 순서가 무작위입니다).

케이스 폴더 구성: 1_열화상_전체.png(흰 굵은 상자 = 대상 부품, 회색 상자 = 동종 부품, 우측 컬러바 10~80°C), 2_열화상_확대.png(하늘색 원+십자 = 최고온 지점), 3_가시광.png(참고용, 열화상과 정렬은 근사), 보고서_A.txt, 보고서_B.txt

평가 방법(케이스당 약 3분): 이미지 3장을 보고 보고서 A, B를 읽은 뒤 채점표.csv에 기입하십시오.
 - 더 유용한 보고서: A / B / 동일
 - 유용성(1~5): 현장 판단에 도움이 되는 정도 (1 전혀, 5 매우)
 - 실행가능성(1~5): 보고서만 보고 바로 조치를 결정할 수 있는 정도
 - 현장 배포 가능: 수정 없이 그대로 작업자에게 내보내도 되는 쪽 (A / B / 둘다 / 없음)
 - 의견: 틀렸거나 근거 없다고 느낀 문장, 빠진 정보 등 자유롭게

주의: 보고서의 수치는 실제 계측값입니다. 판정·등급이 현장 경험과 다르다고 느끼시면 의견란에 적어 주십시오(그 자체가 중요한 평가 자료입니다).
소요 시간 약 90분. 감사합니다.
""")
print(f"kit: {len(chosen)} cases -> {K}"); print("strata:", dict(collections.Counter(k['stratum'] for k in key)), "| A=vlm in", sum(k['A']=='vlm' for k in key), "cases")
