# Publishing these docs

These docs are built with [MkDocs](https://www.mkdocs.org/) and the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme —
Markdown in, a static site out. The recommended host is **GitHub Pages**, which
is free and deploys automatically from CI.

## Build & preview locally

```bash
pip install "ragmill[docs]"     # mkdocs-material + extensions
mkdocs serve                    # live-reload preview at http://127.0.0.1:8000
```

Edit any file under `docs/`, save, and the browser refreshes. Build the static
site (into `site/`) without serving:

```bash
mkdocs build --strict           # --strict fails on broken links / warnings
```

## Deploy to GitHub Pages

### Option A — automatic on every push (recommended)

A workflow at `.github/workflows/docs.yml` builds and deploys the site whenever
you push to `main`:

```yaml
name: docs
on:
  push:
    branches: [main]
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "mkdocs-material>=9.5"
      - run: mkdocs build --strict
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

One-time setup in the repo:

1. Push the workflow, `mkdocs.yml`, and `docs/` to `main`.
2. On GitHub: **Settings → Pages → Build and deployment → Source → GitHub Actions**.
3. The site goes live at `https://<user>.github.io/<repo>/` — here,
   `https://abdullahbinaqeel.github.io/RAGMill/` (matches `site_url` in
   `mkdocs.yml`).

### Option B — one-off manual deploy

MkDocs can push a built site straight to the `gh-pages` branch:

```bash
mkdocs gh-deploy
```

Then set **Settings → Pages → Source** to the `gh-pages` branch. (Prefer Option A
so docs stay in sync automatically.)

## Alternative hosts

- **Read the Docs** — connect the repo, add a `.readthedocs.yaml`, and it builds
  MkDocs for you with versioning. Good if you want per-release docs.
- **Netlify / Vercel / Cloudflare Pages** — set the build command to
  `mkdocs build` and the publish directory to `site`.

## Link it from the project

Once live, point people to it:

- Add the URL to the repository's **About** panel on GitHub.
- Add `Documentation = "https://abdullahbinaqeel.github.io/RAGMill/"` under
  `[project.urls]` in `pyproject.toml` — PyPI then shows a **Documentation** link
  on the package page.
- Add a docs badge to `README.md`:

  ```markdown
  [![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://abdullahbinaqeel.github.io/RAGMill/)
  ```
