"""Render only the public aggregate JSON. Standard library; no patient data."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/"assets/results.json").read_text(encoding="utf8"))
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720" role="img" aria-labelledby="title desc">',
         '<title id="title">FluoroQA retrospective classification results</title>',
         '<desc id="desc">Accuracy and pooled AUROC for three models, Raw and Local, on Test and one held-out specimen. Axes start at zero; no uncertainty intervals shown. Exact values and caveats accompany this figure.</desc>',
         '<rect width="1200" height="720" rx="20" fill="#f5f8fb"/><g font-family="Arial,Helvetica,sans-serif" fill="#193347">']

def text(x, y, value, size=16, extra=""):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{escape(str(value))}</text>')

text(40, 48, 'Does the score separate acceptable and failed poses?', 30, 'font-weight="700"')
text(40, 80, '256 pixels · success <2 mm · retrospective results · full-scale axes', 19)
for k, mode in enumerate(('Raw', 'Local')):
    parts.append(f'<rect x="{890+k*130}" y="101" width="18" height="18" fill="{("#6e9fbd", "#107f84")[k]}"/>')
    text(915+k*130, 117, mode)
for c, cohort in enumerate(('Test', 'Held-out')):
    for m, metric in enumerate(('accuracy', 'auroc')):
        x, y = 40+c*590, 150+m*244
        text(x, y, f'{cohort} / '+('Accuracy (%)' if metric=='accuracy' else 'Pooled AUROC'), 21, 'font-weight="700"')
        start, width = x+126, 360
        for tick in range(5):
            tx = start+width*tick/4
            parts.append(f'<path d="M{tx} {y+15}V{y+172}" stroke="#dbe5eb"/>')
            text(tx, y+194, str(tick*25) if metric=='accuracy' else f'{tick/4:.2f}', 13, 'text-anchor="middle"')
        for i, name in enumerate(('zero', 'mTRE', 'BCE')):
            row_y = y+31+i*53
            text(x, row_y+17, 'FluoroQA-'+name, 15)
            for j, mode in enumerate(('Raw', 'Local')):
                row = next(r for r in data['rows'] if r['model']=='FluoroQA-'+name and r['cohort']==cohort and r['mode']==mode)
                value = row[metric]; yy = row_y+j*20
                parts.append(f'<rect x="{start}" y="{yy}" width="{value*width:.4f}" height="15" rx="2" fill="{("#6e9fbd", "#107f84")[j]}"/>')
                text(start+value*width+5, yy+13, f'{100*value:.2f}' if metric=='accuracy' else f'{value:.3f}', 13)
text(40, 645, 'Test: 52 images / 1,040 candidates. Held-out: 1 specimen / 104 images / 2,080 candidates.', 16)
text(40, 671, 'Raw thresholds: train. Local thresholds: val. zero/mTRE: best through 200; BCE: last 350.', 16)
text(40, 697, 'Different checkpoints and calibration splits; not an isolated BCE ablation or a claim of clinical generalization.', 15)
parts.append('</g></svg>')
(ROOT/"assets/results.svg").write_text('\n'.join(parts)+'\n', encoding="utf8")
print('Rendered assets/results.svg from 12 aggregate rows')
