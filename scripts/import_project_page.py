#!/usr/bin/env python3
"""Import the public presentation from a local Devyn preview export."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PAGE_URL = 'https://jonyeeshen.github.io/Devyn/'
MEDIA = {
    'devyn-selected-teaser-v4': 'devyn-highlight-reel',
    'natural-myoleg-stairs': 'myoleg-stairs-ours',
    'v4-myoleg-hilly-01': 'myoleg-hilly-ours',
    'v3-myoleg-rough-02-side-clean': 'myoleg-rough-ours',
    'v3-myoleg-flat-01-side-clean': 'myoleg-flat-ours',
    'v4-h1-run-ours': 'h1-run-ours',
    'v4-h1-run-sac': 'h1-run-sac',
    'v3-humanoid-h065-mean-e0': 'h1-seated-balance-ours',
    'natural-OstrichRun-v1-nosyn-i2-e0': 'ostrich-run-ours',
    'OstrichRun-v1-sac-s0-e0': 'ostrich-run-sac',
    'v4-g1-balance-selected': 'g1-balance-board-ours',
    'v3-hand-keyturn-continuous': 'myohand-keyturn-ours',
    'v3-hand-reorient-continuous': 'myohand-reorient-ours',
    'v3-hand-reach-continuous': 'myohand-reach-ours',
    'v3-hand-pen-continuous': 'myohand-pen-ours',
    'v3-hand-hold-continuous': 'myohand-hold-ours',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Preview dist directory')
    args = parser.parse_args()
    source = args.source.resolve()
    dest = ROOT / 'docs'
    dest.mkdir(exist_ok=True)
    (dest / 'assets').mkdir(exist_ok=True)

    def read(name):
        return json.loads((source / 'assets' / name).read_text())

    def save(name, value):
        (dest / 'assets' / name).write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n')

    for filename in ['index.html', 'style.css', 'app.js', 'math.js']:
        text = (source / filename).read_text()
        for old, new in MEDIA.items():
            text = text.replace(old, new)
        text = text.replace('devyn-teaser-v4.json', 'teaser-chapters.json').replace('pnosyn', 'ours')
        if filename == 'index.html':
            text = text.replace('<video controls muted loop playsinline preload="metadata"', '<video controls muted loop playsinline preload="none"')
            metadata = (
                f'<link rel="canonical" href="{PAGE_URL}">\n'
                '<meta property="og:type" content="website">\n'
                '<meta property="og:title" content="Devyn — Developing Synergies">\n'
                '<meta property="og:description" content="High-dimensional robotic reinforcement learning with developing synergies. NeurIPS 2026.">\n'
                f'<meta property="og:url" content="{PAGE_URL}">\n'
                f'<meta property="og:image" content="{PAGE_URL}assets/myoleg-stairs-ours.jpg">\n'
                '<meta name="twitter:card" content="summary_large_image">\n'
            )
            text = text.replace('</head>', metadata+'</head>')
        (dest / filename).write_text(text)

    # Only published videos/posters; never import episode receipts or review assets.
    for old, new in MEDIA.items():
        for extension in ['.mp4', '.jpg']:
            src = source / 'assets' / (old+extension)
            if src.exists():
                shutil.copy2(src, dest / 'assets' / (new+extension))
    shutil.copy2(source / 'assets/mark.svg', dest / 'assets/mark.svg')
    shutil.copytree(source / 'vendor/katex', dest / 'vendor/katex', dirs_exist_ok=True)
    (dest / '.nojekyll').touch()

    results = read('results.json')
    results = {'training': {('ours' if k == 'pnosyn' else k): v for k, v in results['training'].items()}, 'test': results['test']}
    save('results.json', results)

    curves = read('main-curves.json')
    tasks = []
    for task in curves['tasks']:
        value = {k: task[k] for k in ['task', 'title', 'configured_plot_budget', 'steps']}
        value['methods'] = {k: {field: values[field] for field in ['n_seeds', 'mean', 'std']} for k, values in task['methods'].items()}
        tasks.append(value)
    save('main-curves.json', {'metric': 'Training episode return', 'smoothing': 'Trailing 100 completed episodes per seed', 'aggregation': 'Mean and sample standard deviation across five seeds', 'tasks': tasks})

    evolution = read('synergy-evolution.json')
    runs = []
    for run in evolution['runs']:
        value = {k: run[k] for k in ['task', 'title', 'rows', 'columns', 'color_scales']}
        value['frames'] = [
            {**{k: f[k] for k in ['step', 'matrix', 'elementwise_proxy', 'hoyer_sparsity']},
             'action_scale': f['tau']}
            for f in run['frames']
        ]
        runs.append(value)
    save('synergy-evolution.json', {
        'default_view': 'proxy',
        'display_transform': 'Elementwise tanh(omega * U_tilde); initialization uses U0.',
        'matrix_axes': evolution['matrix_axes'],
        'hoyer_sparsity_definition': evolution['hoyer_sparsity_definition'],
        'runs': runs,
    })

    transfer = read('transfer-result.json')
    for bar in transfer['bars']:
        bar['seed_values'] = [{k: row[k] for k in ['seed', 'value']} for row in bar['seed_values']]
    save('transfer-result.json', transfer)

    teaser = read('devyn-teaser-v4.json')
    teaser['segments'] = [{k: segment[k] for k in ['start', 'end', 'title']} for segment in teaser['segments']]
    save('teaser-chapters.json', teaser)
    print(f'Imported public page into {dest}')


if __name__ == '__main__':
    main()
