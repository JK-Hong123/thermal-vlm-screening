# -*- coding: utf-8 -*-
"""make_figs.py — all manuscript figures from results/ (PNG 300 dpi + PDF)."""
import json, os, re, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R = r"<PROJECT_ROOT>\results"; F = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120})
C = {'rule': '#7f7f7f', 'rf': '#1f77b4', 'gb': '#2ca02c', 'vlm': '#d62728', 'front': '#9467bd', 'lo': '#ff7f0e'}
def save(fig, name):
    fig.savefig(os.path.join(F, name + '.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(F, name + '.pdf'), bbox_inches='tight'); plt.close(fig); print('saved', name)

# ---------- Fig 1: pipeline schematic ----------
# box-diagram rules: box size from *rendered* text extent (not char counts); fixed inner padding;
# inter-box gap >= 0.35 x median box width so arrows are visible; grey = code-owned (deterministic),
# one accent colour + dashed edge = the only model contribution; no in-figure subtitle (caption owns meaning).
from matplotlib.patches import Patch
FS = 8                      # in-figure font size; figure is ~7.7 in wide -> >= 7 pt after scaling to text width
PAD_X, PAD_Y, GAP, MARG = 0.08, 0.09, 0.42, 0.05   # inches
labels = ['Radiometric\nCSV + JSON\n(512×640,\nbboxes)',
          'Radiometric\n+ spatial\nfeatures (26)\n+ equipment',
          'Equipment-\nconditioned\ngate\n(tree\nensemble)',
          'Code-owned\nseverity:\nS/A/P axes,\nlimits,\nreview queue,\nflags',
          'Language model\ntier 1: narrative\nfrom facts, no image\ntier 2: blind image\nobservation\n(screened only)',
          'Validated\nauditable\nreport']
is_model = [False, False, False, False, True, False]
GREY, GREY_EDGE, ACC, ACC_EDGE = '#efefef', '#444444', '#fde9d9', '#c0504d'
_m = plt.figure(figsize=(2, 2)); _m.canvas.draw(); _r = _m.canvas.get_renderer()
def _size_in(s, fs=FS):
    t = _m.text(0, 0, s, fontsize=fs, ha='center', va='center'); bb = t.get_window_extent(_r); t.remove()
    return bb.width / _m.dpi, bb.height / _m.dpi
sizes = [_size_in(s) for s in labels]; lab_w = _size_in('validation', FS - 1.5)[0]; plt.close(_m)
widths = [w + 2 * PAD_X for w, _ in sizes]
box_h = max(h for _, h in sizes) + 2 * PAD_Y
assert GAP >= 0.35 * float(np.median(widths)), 'gap too small for arrows'
gaps = [GAP] * (len(widths) - 1); gaps[4] = max(GAP, lab_w + 0.18)   # the labelled arrow needs room for its label
total_w = sum(widths) + sum(gaps) + 2 * MARG
leg_h = 0.32
total_h = box_h + leg_h + 2 * MARG
fig = plt.figure(figsize=(total_w, total_h)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
ax.set_xlim(0, total_w); ax.set_ylim(0, total_h)
y0 = MARG + leg_h; yc = y0 + box_h / 2
x = MARG; edges = []
for i, (s, w, m) in enumerate(zip(labels, widths, is_model)):
    ax.add_patch(FancyBboxPatch((x, y0), w, box_h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                fc=ACC if m else GREY, ec=ACC_EDGE if m else GREY_EDGE, lw=1.1, ls='--' if m else '-'))
    ax.text(x + w / 2, yc, s, ha='center', va='center', fontsize=FS, linespacing=1.15)
    edges.append((x, x + w)); x += w + (gaps[i] if i < len(gaps) else 0)
for i in range(len(edges) - 1):
    x0, x1 = edges[i][1] + 0.03, edges[i + 1][0] - 0.03
    ax.add_patch(FancyArrowPatch((x0, yc), (x1, yc), arrowstyle='-|>', mutation_scale=18, lw=1.5, color='#333333', shrinkA=0, shrinkB=0))
# the one arrow that carries a condition: VLM output enters the report only through validation
xv0, xv1 = edges[4][1], edges[5][0]
ax.text((xv0 + xv1) / 2, yc + 0.16, 'validation', ha='center', va='bottom', fontsize=FS - 1.5, color='#333333')
fig.legend(handles=[Patch(fc=GREY, ec=GREY_EDGE, lw=1.1, label='owned by deterministic code (verdict, severity, every number)'),
                    Patch(fc=ACC, ec=ACC_EDGE, lw=1.1, ls='--', label='language model (descriptive fields only; tier 2 only after passing the screening protocol)')],
           loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=FS - 1, frameon=False, handlelength=1.6, columnspacing=1.5)
save(fig, 'fig1_pipeline')

# ---------- Fig 2: equipment normal/danger distributions ----------
norms = json.load(open(os.path.join(R, 'equipment_norms.json'), encoding='utf-8'))
EN = {'계기용변류기(CT)': 'CT', '계기용변성기(MOF)': 'MOF', '계기용변압기(PT)': 'PT', '고압전동기': 'HV motor', '기어박스': 'Gearbox', '단로기(DS)': 'Disconnector',
      '변압기 접속부': 'Transformer conn.', '변압기 케이블 연결개소 포함': 'Transformer cable', '보일러': 'Boiler', '전자접촉기': 'Contactor', '차단기': 'Breaker',
      '차축베어링': 'Axle bearing', '케이블&부스 연결개소': 'Cable/bus joint', '펌프': 'Pump', '신재생 설비': 'PV (unseen)', '접지용 콘덴서': 'Grounding cap. (unseen)'}
eq = sorted(norms, key=lambda k: norms[k]['t_max']['normal_q']['50']); y = np.arange(len(eq))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
for ax, key, lab in ((axs[0], 't_max', 'Target max temperature (°C)'), (axs[1], 'dT_scene_max', 'ΔT over scene median (°C)')):
    for i, e in enumerate(eq):
        q = norms[e][key]['normal_q']; d = norms[e][key]['danger_q']
        ax.plot([q['5'], q['95']], [i, i], color='#9ecae1', lw=6, solid_capstyle='butt'); ax.plot(q['50'], i, 'o', color=C['rf'], ms=4)
        if d: ax.plot(d['50'], i, 'x', color=C['vlm'], ms=6, mew=1.5)
    ax.set_xlabel(lab); ax.grid(axis='x', alpha=0.3)
axs[0].set_yticks(y); axs[0].set_yticklabels([EN.get(e, e) for e in eq])
h1, = axs[0].plot([], [], color='#9ecae1', lw=6, label='normal frames, P5–P95'); h2, = axs[0].plot([], [], 'o', color=C['rf'], label='normal median'); h3, = axs[0].plot([], [], 'x', color=C['vlm'], label='danger median')
# legend placed above both panels (not inside the plot area) so it never sits on top of a bar
fig.legend(handles=[h1, h2, h3], loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=7.5, frameon=False)
fig.subplots_adjust(top=0.86)
save(fig, 'fig2_equipment_norms')

# ---------- Fig 3: label vs standards + boundary contamination ----------
fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.6), gridspec_kw={'wspace': 0.65})
S = {'train': [0.03, 0.20, 0.33, 0.28, 0.07], 'test': [0.05, 0.21, 0.34, 0.32, 0.03]}; A = {'test': [0.03, 0.12, 0.26, 0.12, 0.23], 'gen_test': [0.05, 0.42, 0.39, 0.69, 0.99]}
x = np.arange(5); w = 0.38
axs[0].bar(x-w/2, S['train'], w, color='#c6dbef', label='train'); axs[0].bar(x+w/2, S['test'], w, color=C['rf'], label='test'); axs[0].set_title('S axis: sibling ΔT (electrical)', fontsize=8)
axs[1].bar(x-w/2, A['test'], w, color=C['rf'], label='test (seen equip.)'); axs[1].bar(x+w/2, A['gen_test'], w, color=C['lo'], label='unseen equip.'); axs[1].set_title('A axis: scene ΔT', fontsize=8)
for ax in axs[:2]: ax.set_xticks(x); ax.set_xticklabels(['L0', 'L1', 'L2', 'L3', 'L4']); ax.set_ylim(0, 1); ax.set_ylabel('fraction labelled danger'); ax.legend(fontsize=7, frameon=False)
bd = {'Gearbox': (81.2, 42.0, 92.8), 'Contactor': (35.2, 4.8, 73.4), 'Axle bearing': (30.5, 15.6, 31.9), 'Breaker': (22.6, 1.2, 79.2), 'Disconnector': (26.4, 2.5, 44.6), 'Cable/bus joint': (18.4, 0.7, 52.0), 'PV': (11.8, 0.0, 24.4), 'PT': (0.0, 0.0, 0.0)}
names = list(bd); yy = np.arange(len(names))
axs[2].barh(yy+0.2, [bd[n][0] for n in names], 0.4, color='#9ecae1', label='peak on box border (<3 px)'); axs[2].barh(yy-0.2, [bd[n][1] for n in names], 0.4, color=C['vlm'], label='ΔT drops >2 °C if border removed')
axs[2].set_yticks(yy); axs[2].set_yticklabels(names, fontsize=7); axs[2].set_xlabel('% of bboxes'); axs[2].set_title('Boundary contamination', fontsize=8)
# legend placed fully below the panel (outside the data/tick-label area) so it can never overlap a bar or a y-tick label
axs[2].legend(fontsize=6.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1)
save(fig, 'fig3_labels_boundary')

# ---------- Fig 4: detection baselines ----------
rows = []
for line in open(os.path.join(R, 'detect_baselines.txt'), encoding='utf-8'):
    m = re.match(r'(.+?)\s+(\d\.\d{3}) \[(\d\.\d{3}),(\d\.\d{3})\]\s+(\d\.\d{3}) \[(\d\.\d{3}),(\d\.\d{3})\]', line.strip())
    if m: rows.append((m.group(1).strip(), *[float(m.group(i)) for i in range(2, 8)]))
fig, ax = plt.subplots(figsize=(7.2, 2.8)); n = len(rows); x = np.arange(n); w = 0.38
for k, (off, lab, col) in enumerate(((-w/2, 'seen equipment, new sessions', C['rf']), (w/2, 'unseen equipment', C['lo']))):
    vals = [r[1+3*k] for r in rows]; lo = [r[1+3*k]-r[2+3*k] for r in rows]; hi = [r[3+3*k]-r[1+3*k] for r in rows]
    ax.bar(x+off, vals, w, yerr=[lo, hi], color=col, capsize=2, label=lab, error_kw=dict(lw=0.8))
ax.set_xticks(x); ax.set_xticklabels([r[0].replace('rule dT_scene', 'ΔT rule').replace(' (ours)', '').replace(' (std)', '').replace(' (std, 2x128)', '') for r in rows], rotation=20, ha='right'); ax.set_ylim(0.5, 1.0); ax.set_ylabel('AUROC (95% CI)'); ax.legend(fontsize=7, frameon=False, loc='lower left'); ax.grid(axis='y', alpha=0.3)
save(fig, 'fig4_detection')

# ---------- Fig 5: VLM controls (families + frontier) ----------
fam = []
for fp in sorted(os.listdir(os.path.join(R, 'vlm_out', 'families'))):
    if fp.endswith('.json'):
        s = json.load(open(os.path.join(R, 'vlm_out', 'families', fp), encoding='utf-8'))['summary']
        fam.append((s['model'].split('/')[-1].replace('-Instruct', '').replace('-hf', '').replace('llava-onevision-qwen2-7b-ov', 'LLaVA-OV-7B') + (' (4bit)' if s['quant'] else '') + f" [{s['lang']}]", s['kappa'], s['follows_moved_marker']))
# values below recomputed 2026-10-29 with the corrected (longest-first) cell parser; see results/marker_follow_recompute.txt
fam.append(('Qwen2.5-VL-7B (4bit) [ko], dev set n=70', 0.095, 0.04)); fam.append(('Qwen2.5-VL-3B + adapter, in-pipeline [ko]', 0.356, np.nan))
fam.append(('Gemini 3.5 Flash-Lite [ko]', 0.708, 0.75)); fam.append(('Frontier (Claude Fable 5.1), tool-free [ko]', 0.943, 0.95))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True); yy = np.arange(len(fam))
cols = [C['front'] if 'Frontier' in f[0] else ('#2a9d8f' if 'Gemini' in f[0] else (C['lo'] if 'adapter' in f[0] else C['vlm'])) for f in fam]
axs[0].barh(yy, [f[1] for f in fam], color=cols); axs[0].axvline(0, color='k', lw=0.6); axs[0].set_xlabel("marker-cell κ vs CSV truth"); axs[0].set_xlim(-0.1, 1)
axs[1].barh(yy, [0 if np.isnan(f[2]) else f[2] for f in fam], color=cols); axs[1].axvline(1/9, color='k', ls='--', lw=0.7); axs[1].set_xlabel('follows a moved marker (fraction)'); axs[1].set_xlim(0, 1)
axs[0].set_yticks(yy); axs[0].set_yticklabels([f[0] for f in fam], fontsize=7); axs[1].text(0.13, len(fam)-0.6, 'chance', fontsize=7)
for ax in axs: ax.grid(axis='x', alpha=0.3)
save(fig, 'fig5_vlm_controls')

# ---------- Fig 6: resolution ceiling + LoRA experiments ----------
ceil = {}
for line in open(os.path.join(R, 'resolution_ceiling.txt'), encoding='utf-8'):
    m = re.match(r'(p\S+ ctx[\d.]+@\d+)/(K\d+|pixel)\s+([\d.]+)\s+([\d.]+)', line.strip())
    if m: ceil[(m.group(1), m.group(2))] = float(m.group(4))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
for key, lab, col in (('p100 ctx2.4@336', 'context 2.4×, 336 px', C['rule']), ('p100 ctx1.4@672', 'context 1.4×, 672 px', C['rf'])):
    ks = [12, 24, 36, 48]; axs[0].plot(ks, [ceil[(key, f'K{k}')] for k in ks], 'o-', color=col, label=lab)
axs[0].axhline(ceil[('p100 ctx1.4@672', 'pixel')], color='k', ls='--', lw=0.8); axs[0].text(13, ceil[('p100 ctx1.4@672', 'pixel')]+0.02, 'pixel-level ceiling', fontsize=7)
axs[0].scatter([12, 24, 24], [0.271, 0.401, 0.674], marker='*', s=70, color=C['lo'], zorder=5, label='trained 3B (A, E, F)')
axs[0].set_xlabel('visual token grid K (K×K)'); axs[0].set_ylabel('κ (val, smoothed-argmax label)'); axs[0].set_xticks(ks); axs[0].legend(fontsize=7, frameon=False); axs[0].grid(alpha=0.3); axs[0].set_title('Recoverability ceiling vs token resolution', fontsize=8)
exps = [('zero-shot\n3B', 0.144, 0.032, 0.091), ('LoRA F\n(6k, 1 ep)', 0.539, 0.694, 0.417), ('LoRA H\n(8k, 2 ep)', 0.650, 0.761, 0.564), ('brightest\npixel', 0.870, 0.855, 0.878)]
x = np.arange(len(exps)); w = 0.27
axs[1].bar(x-w, [e[1] for e in exps], w, color=C['rule'], label='all (n=388)'); axs[1].bar(x, [e[2] for e in exps], w, color=C['rf'], label='seen equipment'); axs[1].bar(x+w, [e[3] for e in exps], w, color=C['lo'], label='unseen equipment')
axs[1].set_xticks(x); axs[1].set_xticklabels([e[0] for e in exps], fontsize=7.5); axs[1].set_ylabel('κ (independent 119-session set)'); axs[1].legend(fontsize=7, frameon=False); axs[1].grid(axis='y', alpha=0.3); axs[1].set_title('Hotspot localisation from the rendered image', fontsize=8)
save(fig, 'fig6_lora_resolution')

# ---------- Fig 7: blinded judging ----------
# Compact: five numbers with CIs do not need a full-width panel; sized for 0.5\linewidth (~3.35 in) so
# in-figure text stays >= 7 pt. No in-figure commentary: the caption/Discussion carry it.
# Horizontal bars grouped by judging protocol; numbers live in results/judging_bars.json so they can be updated without code edits.
J = [j for j in json.load(open(os.path.join(R, 'judging_bars.json'), encoding='utf-8')) if j['k'] is not None]
groups = []
for j in J:
    if j['group'] not in groups: groups.append(j['group'])
fig, ax = plt.subplots(figsize=(4.6, 0.42 * len(J) + 0.55 * len(groups) + 0.6))
ypos = []; y = 0.0; gy = []
for g in groups[::-1]:
    items = [j for j in J if j['group'] == g]
    for j in items[::-1]: ypos.append((y, j)); y += 1.0
    gy.append((y - 0.5 + 0.5, g)); y += 0.9
vals = [j['k'] / j['n'] for _, j in ypos]
ax.barh([p for p, _ in ypos], vals, height=0.62, color=[j['color'] for _, j in ypos],
        xerr=[[v - j['lo'] for v, (_, j) in zip(vals, ypos)], [j['hi'] - v for v, (_, j) in zip(vals, ypos)]], capsize=2.5, error_kw={'lw': 0.9})
for v, (p, j) in zip(vals, ypos): ax.text(1.02, p, f"{j['k']}/{j['n']}", va='center', fontsize=8, color='#333333')
for yg, g in gy: ax.text(-0.02, yg - 0.15, g, ha='right', va='center', fontsize=8.5, fontweight='bold', clip_on=False)
ax.axvline(0.5, color='k', ls='--', lw=0.7); ax.set_yticks([p for p, _ in ypos]); ax.set_yticklabels([j['label'] for _, j in ypos], fontsize=8.5)
ax.set_xlabel('frontier/VLM report preferred over the code-only report (fraction)', fontsize=9); ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, y - 0.4)
ax.tick_params(axis='x', labelsize=8.5); ax.grid(axis='x', alpha=0.3)
save(fig, 'fig7_judging')

# ---------- Fig 9: what the models see (one locked case + one synthetic probe image) ----------
import matplotlib.image as mpimg
L = r"<PROJECT_ROOT>\results\cases_lock"
panels = [(os.path.join(L, 'case001', 'thermal_full.png'), '(a) full frame, 10–80 °C\nwhite = target, grey = siblings'),
          (os.path.join(L, 'case001', 'thermal_crop_v2.png'), '(b) crop, marker at\nthe radiometric peak'),
          (os.path.join(L, 'case001', 'thermal_crop_v2_pert.png'), '(c) displacement control:\nmarker moved'),
          (os.path.join(R, 'probe', 'L2_synth_mark_04.png'), '(d) synthetic probe\n(level 2)')]
fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.9), gridspec_kw={'width_ratios': [1.25, 1, 1, 1], 'wspace': 0.08})
for ax, (p, cap) in zip(axs, panels):
    ax.imshow(mpimg.imread(p)); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(cap, fontsize=7.5, loc='left')
    for s in ax.spines.values(): s.set_visible(False)
save(fig, 'fig9_examples')

# ---------- Fig 8: example report ----------
txt = open(os.path.join(R, 'vlm_out', 'v2lora', 'a0plus', 'case001.txt'), encoding='utf-8').read()
fig = plt.figure(figsize=(7.2, 4.2)); fig.text(0.01, 0.99, txt, va='top', ha='left', fontsize=6.2, family='Malgun Gothic', linespacing=1.35)
fig.savefig(os.path.join(F, 'fig8_report_example.png'), dpi=300, bbox_inches='tight'); plt.close(fig); print('saved fig8_report_example (png only, Korean text)')
print('done')
