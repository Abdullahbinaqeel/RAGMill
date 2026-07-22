# Releasing RAGMill

This is the authoritative procedure for cutting a new release of RAGMill and
publishing it to **PyPI** and **GitHub**. It is written so that any maintainer
can follow it start to finish.

RAGMill publishes to PyPI via **trusted publishing (OIDC)** — there is no API
token stored anywhere. A GitHub Release triggers
[`.github/workflows/publish.yml`](.github/workflows/publish.yml), which builds
the sdist + wheel, runs `twine check`, and uploads to PyPI.

---

## Versioning

RAGMill follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

- **PATCH** — backwards-compatible bug fixes.
- **MINOR** — backwards-compatible new features.
- **MAJOR** — breaking changes (reserved for the road to 1.0).

The version lives in **one place**: `version` in `pyproject.toml`. Keep the
`version="..."` strings in `server.py` and `config_ui.py` in sync with it.

---

## One-time setup (first release only)

1. **Create the PyPI project & trusted publisher.**
   On <https://pypi.org>, register a pending publisher for the project `ragmill`
   under *Your projects → Publishing*:
   - Owner: `Abdullahbinaqeel`
   - Repository: `RAGMill`
   - Workflow name: `publish.yml`
   - Environment: `pypi`
2. **Create the `pypi` GitHub Environment.**
   Repo → *Settings → Environments → New environment* → name it `pypi`.
   (Optionally add required reviewers so releases are gated.)
3. No secrets or tokens are needed — OIDC handles authentication.

> Tip: to rehearse without touching real PyPI, configure the same on
> <https://test.pypi.org> and add a TestPyPI publish step.

---

## Release checklist

Run through this list for every release.

### 1. Pre-flight (on a clean `main`)

```bash
git checkout main && git pull
pip install -e ".[dev]"

pytest --cov=src/ragmill --cov-report=term-missing --cov-fail-under=75
black --check src/ tests/
mypy src/ragmill/ --ignore-missing-imports --no-strict-optional
mkdocs build --strict          # only if docs changed
```

All four must pass.

### 2. Bump the version

- Update `version` in `pyproject.toml`.
- Update the matching `version="..."` in `src/ragmill/server.py` and
  `src/ragmill/config_ui.py`.

### 3. Update the changelog

- Add a new section to [`CHANGELOG.md`](CHANGELOG.md) for the version, dated,
  under `### Added` / `### Changed` / `### Fixed` / `### Removed`.

### 4. Build & verify locally

```bash
rm -rf dist/
python -m build                # builds sdist + wheel
twine check dist/*             # validates metadata + README rendering
python -m pip install dist/ragmill-*.whl   # smoke-test in a fresh venv
python -c "import ragmill; print(ragmill.RAGEngine)"
```

Confirm the wheel and sdist contents are correct:

```bash
unzip -l dist/ragmill-*.whl        # should include all modules + static dirs + entry point
tar -tzf dist/ragmill-*.tar.gz     # sdist should NOT contain review/scratch markdown
```

### 5. Commit, tag, and push

```bash
git add pyproject.toml CHANGELOG.md src/ragmill/server.py src/ragmill/config_ui.py
git commit -m "chore(release): vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

### 6. Publish the GitHub Release

- GitHub → *Releases → Draft a new release*.
- Choose the tag `vX.Y.Z`, title `vX.Y.Z`.
- Paste the CHANGELOG entry as the release notes.
- Click **Publish release**.

This triggers `publish.yml`, which builds and uploads to PyPI automatically.

### 7. Post-release verification

```bash
# Wait ~1 min for PyPI to index, then:
pip install --upgrade ragmill
pip show ragmill               # confirm the new version
```

- Check the [PyPI project page](https://pypi.org/project/ragmill/) renders the
  README correctly.
- Confirm the [CI](https://github.com/Abdullahbinaqeel/RAGMill/actions) and
  publish workflows are green.
- If docs changed, confirm the
  [docs site](https://abdullahbinaqeel.github.io/RAGMill/) updated (deployed by
  `docs.yml`).

---

## If something goes wrong

- **PyPI rejects an already-uploaded version.** Versions are immutable — you
  cannot re-upload `X.Y.Z`. Bump to `X.Y.(Z+1)` and release again.
- **A broken release is live.** Yank it on PyPI (*Manage → Yank*) so it stops
  being installed by default, then publish a fixed version.
- **The publish workflow fails.** Re-run it from the Actions tab, or use the
  `workflow_dispatch` trigger after fixing the cause. Trusted publishing requires
  the run to originate from the `pypi` environment on this repo.
