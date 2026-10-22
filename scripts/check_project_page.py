#!/usr/bin/env python3
"""Check the static project's dependencies and GitHub Pages subpath compatibility."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1] / 'docs'
errors = []


def local_reference(value, parent=ROOT):
    if not value or value.startswith(('#', 'data:')) or urlsplit(value).scheme:
        return
    if value.startswith('/'):
        errors.append(f'Root-relative URL breaks /Devyn/: {value}')
        return
    path = (parent / urlsplit(value).path).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        errors.append(f'Missing or outside-site reference: {value}')


class References(HTMLParser):
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ['src', 'href', 'poster']:
                local_reference(value)


References().feed((ROOT / 'index.html').read_text())
for css in ROOT.rglob('*.css'):
    for reference in re.findall(r'url\([\"\']?([^\)\"\']+)', css.read_text()):
        local_reference(reference, css.parent)
app = (ROOT / 'app.js').read_text()
for path in re.findall(r"fetch\(['\"]([^'\"]+)", app):
    local_reference(path)
for stem in re.findall(r"(?:ours|sac):'([^']+)'", app):
    if re.fullmatch(r'[a-z0-9-]+', stem):
        for extension in ['mp4', 'jpg']:
            local_reference(f'assets/{stem}.{extension}')

for path in ROOT.rglob('*'):
    if path.is_symlink():
        errors.append(f'Pages artifact cannot include symlinks: {path}')
    if path.is_file() and path.stat().st_size >= 100*1024*1024:
        errors.append(f'File exceeds GitHub normal-file limit: {path}')
    if path.is_file() and (path.suffix in ['.zip', '.npz', '.pkl'] or path.name.startswith(('review.', 'curation-'))):
        errors.append(f'Non-public research asset: {path}')
    if path.suffix in ['.html', '.js', '.json', '.css']:
        text = path.read_text()
        for marker in ['/home/isi/', 'saved_models_energy/', 'source_checkpoint', 'configuration_path', 'devyn_project_page/evidence', 'chatgpt.site']:
            if marker in text:
                errors.append(f'Internal preview metadata in {path.name}: {marker}')
        if path.suffix == '.json':
            json.loads(text)

chapters = json.loads((ROOT/'assets/teaser-chapters.json').read_text())
assert chapters['segments'][0]['start'] == 0
assert chapters['segments'][-1]['end'] == chapters['duration']
assert all(a['end'] == b['start'] for a, b in zip(chapters['segments'], chapters['segments'][1:]))
assert len(json.loads((ROOT/'assets/results.json').read_text())['test']) == 6
if errors:
    raise SystemExit('\n'.join(errors))
files = [p for p in ROOT.rglob('*') if p.is_file()]
print(f'Project page OK: {len(files)} files, {sum(p.stat().st_size for p in files)/1e6:.1f} MB; all references resolve under /Devyn/.')
