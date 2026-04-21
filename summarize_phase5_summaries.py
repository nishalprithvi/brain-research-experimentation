#!/usr/bin/env python3
import csv
import glob

TH = {'acc': 0.74, 'mf1': 0.58, 'auc': 0.81, 'ad': 0.70, 'mci': 0.40}
rows = []
paths = sorted(set(glob.glob('job_logs/p5_*/summary.tsv') + glob.glob('job_logs/*/summary.tsv')))
for p in paths:
    try:
        with open(p, newline='') as f:
            rd = csv.DictReader(f, delimiter='\t')
            for r in rd:
                try:
                    acc = float(r['acc'])
                    mf1 = float(r['macro_f1'])
                    auc = float(r['auc'])
                    ad = float(r['ad_recall'])
                    mci = float(r['mci_recall'])
                except Exception:
                    continue
                rows.append({
                    'path': p,
                    'tag': r.get('tag', ''),
                    'acc': acc,
                    'mf1': mf1,
                    'auc': auc,
                    'ad': ad,
                    'mci': mci,
                    'pass': int(acc >= TH['acc'] and mf1 >= TH['mf1'] and auc >= TH['auc'] and ad >= TH['ad'] and mci >= TH['mci']),
                })
    except Exception:
        pass

if not rows:
    print('NO_ROWS')
    raise SystemExit(0)

rows_sorted = sorted(rows, key=lambda x: (x['pass'], x['mf1'], x['auc'], x['ad'], x['mci'], x['acc']), reverse=True)
print('TOP10')
for r in rows_sorted[:10]:
    print(
        f"pass={r['pass']} acc={r['acc']:.4f} mf1={r['mf1']:.4f} auc={r['auc']:.4f} "
        f"ad={r['ad']:.4f} mci={r['mci']:.4f} tag={r['tag']} file={r['path']}"
    )

best = rows_sorted[0]
print('BEST', best)
