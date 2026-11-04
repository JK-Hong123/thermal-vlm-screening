# -*- coding: utf-8 -*-
"""audit_agents.py — tool-access audit of Claude Code agent batches from the retained transcripts.
For every Agent launch recorded in the orchestrating session on the given dates whose description starts with one of the
given prefixes, list model, tools called, image reads, non-image reads, shell commands, and any read/command touching a
truth-bearing file. Appends a dated section to results/frontier_tool_audit.txt (and the sanitised supplementary copy).
usage: python audit_agents.py --dates 2026-10-29 --prefixes "No-image frontier arm,Same-model judge,Frontier fact-falsification,Frontier synthetic probe"
"""
import json, os, re, sys, io, glob, collections, argparse, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
PROJ = r"<USER_HOME>\.claude\projects\<SESSION_DIR>"; S = 'dab93585-2101-4579-93da-5b01af9e2801'
TRUTH = re.compile(r'facts\.json|\.csv|manifest|index\.jsonl|hidden|_key|cases_lock[\\\\/]+case\d+[\\\\/]+(?!thermal|rgb)|truth', re.I)
def launches(dates):
    main = os.path.join(PROJ, S + '.jsonl'); L = []; idmap = {}
    for line in open(main, encoding='utf-8', errors='replace'):
        if not any(d in line for d in dates): continue
        try: o = json.loads(line)
        except Exception: continue
        mc = o.get('message', {}).get('content')
        if o.get('type') == 'assistant':
            for c in mc or []:
                if c.get('type') == 'tool_use' and c.get('name') == 'Agent': L.append((o.get('timestamp', '')[:16], c['input'].get('description', ''), c['id']))
        elif o.get('type') == 'user' and isinstance(mc, list):
            for c in mc:
                if c.get('type') == 'tool_result':
                    m = re.search(r'agentId:\s*([0-9a-f]{15,})', json.dumps(c.get('content', ''), ensure_ascii=False))
                    if m: idmap[c.get('tool_use_id')] = m.group(1)
    return [(ts, d, idmap.get(tid)) for ts, d, tid in L]
def audit(aid):
    fp = os.path.join(PROJ, S, 'subagents', f'agent-{aid}.jsonl')
    if not os.path.exists(fp): return None
    tools = collections.Counter(); reads = []; cmds = []; models = collections.Counter()
    for line in open(fp, encoding='utf-8', errors='replace'):
        try: o = json.loads(line)
        except Exception: continue
        if o.get('type') != 'assistant': continue
        m = o.get('message', {}); models[m.get('model')] += 1
        for c in m.get('content', []):
            if c.get('type') != 'tool_use': continue
            tools[c['name']] += 1; inp = c.get('input', {})
            if c['name'] == 'Read': reads.append(inp.get('file_path', ''))
            if c['name'] in ('Bash', 'PowerShell'): cmds.append(inp.get('command') or '')
    png = [r for r in reads if r.lower().endswith('.png')]; nonpng = sorted(set(os.path.basename(r) for r in reads if not r.lower().endswith('.png')))
    return dict(model=dict(models), tools=dict(tools), png=len(png), nonpng=nonpng, n_cmds=len(cmds),
                truth_reads=[r for r in reads if TRUTH.search(r)], truth_cmds=[c[:160].replace('\n', ' ') for c in cmds if TRUTH.search(c)])
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dates', required=True); ap.add_argument('--prefixes', required=True); ap.add_argument('--title', default='')
    a = ap.parse_args(); prefixes = [p.strip() for p in a.prefixes.split(',')]
    L = [f"", f"=== AUDIT SECTION added {datetime.date.today()} — {a.title or 'agent batches of ' + a.dates} ==="]
    tot = 0; n_truth = 0; n_agents = 0
    for ts, desc, aid in launches(a.dates.split(',')):
        if not any(desc.startswith(p) for p in prefixes): continue
        r = audit(aid) if aid else None; n_agents += 1
        if r is None: L.append(f"{ts} {desc} | agent {aid} | TRANSCRIPT MISSING"); continue
        role = 'JUDGE' if 'judge' in desc.lower() else 'SUBJECT'
        if role == 'SUBJECT': n_truth += len(r['truth_reads']) + len(r['truth_cmds'])
        L.append(f"{ts} {desc} [{role}] | agent-{aid} | {r['model']} | {r['tools']} | PNG reads {r['png']} | non-image reads {r['nonpng'] or 'none'} | shell {r['n_cmds']} | TRUTH reads {r['truth_reads'] or 'none'} | TRUTH cmds {r['truth_cmds'] or 'none'}")
    L.append(f"SUMMARY: {n_agents} agents; truth-bearing reads or commands by SUBJECT agents: {n_truth}. (Judge agents read only the task list and the listed images; they are not given facts.json in the forced-grounding protocol.)")
    txt = "\n".join(L)
    for p, san in ((r"<PROJECT_ROOT>\results\frontier_tool_audit.txt", False), (r"<PROJECT_ROOT>\paper\supp_frontier_tool_audit.txt", True)):
        t = txt
        if san: t = re.sub(r"C:\\\\Users\\\\<USER>\\\\AppData\\\\Local\\\\Temp\\\\claude\\\\[^'\" ]*?\\\\scratchpad", "<scratch>", t).replace("<USER>", "<user>").replace('—', '-').replace('×', 'x').replace('·', '-')
        open(p, 'a', encoding='utf-8').write(t + "\n")
    print(txt)
