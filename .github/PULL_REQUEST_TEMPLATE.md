<!--
Thanks for contributing to RAGMill! Please fill out the sections below.
Keep changes focused and use Conventional Commits in your commit messages.
-->

## Summary

<!-- What does this PR do, and why? -->

## Related issue

<!-- e.g. Closes #123 -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation
- [ ] Refactor / chore

## Checklist

- [ ] `pytest` passes and coverage stays ≥ 75%
- [ ] `black --check src/ tests/` is clean
- [ ] `mypy src/ragmill/ --ignore-missing-imports --no-strict-optional` is clean
- [ ] Added/updated tests (bug fixes include a regression test)
- [ ] New dependencies are optional (added to the right extra, imported lazily)
- [ ] Updated `CHANGELOG.md` if user-facing
- [ ] Updated docs (`docs/`) if behavior or API changed; `mkdocs build --strict` is clean
