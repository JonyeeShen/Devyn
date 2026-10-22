# Devyn project page

Public URL: **https://jonyeeshen.github.io/Devyn/**

The website is plain HTML, CSS and JavaScript in [`docs/`](../docs). GitHub Pages
serves the published files; no separate server, Node build, or paid domain is required.

## Update the website

1. Edit `docs/index.html`, `docs/style.css`, `docs/typography.css`, `docs/app.js`, or the media/data under `docs/assets/`.
2. Run `python3 scripts/check_project_page.py` and `node --check docs/app.js`.
3. Commit and push to `main`. The **Deploy project page** workflow publishes changes automatically.

Deployment status is shown in the repository's **Actions** tab. Settings → Pages
uses **GitHub Actions** as the publishing source. A failed deployment leaves the
previous successfully deployed page available.

For a simple local preview, run `python3 -m http.server 8000 --directory docs` and
open `http://localhost:8000/`. The public site lives below `/Devyn/`, so asset URLs
are relative. GitHub Pages supports video byte ranges for scrubbing.

## Import an updated research preview

The importer takes the public presentation and a specific selection of assets:

```bash
python3 scripts/import_project_page.py --source /path/to/preview/dist
python3 scripts/check_project_page.py
```

It preserves numerical values while removing local source paths and private
curation metadata. The review library, checkpoint files, per-episode receipts and
research logs are not part of the public site. Published filenames use readable
task names. If the preview introduces new media, update the importer's `MEDIA`
mapping before importing.

The original research workspace remains the archive for full source provenance.
The public `docs/` files can be maintained directly after import.

## Assets and dependencies

- Videos are real simulation rollouts and play at their original speed.
- The teaser has a chapter index in `docs/assets/teaser-chapters.json`.
- Plot data and synergy matrices are static JSON files under `docs/assets/`.
- KaTeX is vendored under `docs/vendor/katex/`, including its license.
- Source Sans 3 is self-hosted under `docs/vendor/source-sans-3/`, with its SIL Open Font License and upstream provenance. KaTeX retains its own math fonts.
- `docs/typography.css` owns the page's type scale; `style.css` handles layout and color. Keep new component text on the shared type tokens.

GitHub Pages documentation: https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages
