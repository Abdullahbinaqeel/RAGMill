# Security Policy

## Supported versions

RAGMill is pre-1.0. Security fixes are applied to the latest released minor
version only.

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, use GitHub's private vulnerability reporting:

1. Go to the
   [Security tab](https://github.com/Abdullahbinaqeel/RAGMill/security/advisories/new)
   of the repository.
2. Click **Report a vulnerability** and describe the issue.

Please include, where possible:

- A description of the vulnerability and its impact.
- Steps to reproduce (a minimal proof of concept helps enormously).
- The affected version(s) and configuration.
- Any suggested remediation.

### What to expect

- **Acknowledgement** within 3 business days.
- An initial **assessment** within 7 business days.
- We will keep you informed of progress and coordinate a disclosure timeline
  with you. Please allow us a reasonable window to release a fix before any
  public disclosure.

## Deploying RAGMill securely

RAGMill can run as a network service (`ragmill serve`). The REST API is designed
to be safe by default, but the operator is responsible for the deployment. Please
observe the following:

- **Bind address.** The server binds to `127.0.0.1` by default. Only set
  `RAGMILL_HOST=0.0.0.0` when you intend to expose it, and place it behind a
  reverse proxy or gateway.
- **Authentication.** Set `RAGMILL_API_KEY` to require an `X-API-Key` header on
  all endpoints except `/health`. A network-exposed server without an API key is
  strongly discouraged.
- **Ingest path allow-list.** `RAGMILL_ALLOWED_ROOTS` (colon-separated) restricts
  the directories the `/ingest` and `/sync` endpoints may read. Without it, a
  request could ask the server to read files outside your intended corpus. Set it
  whenever the server is reachable by untrusted clients.
- **Secrets.** Never commit `.env` or API keys. `.env` is git-ignored by default.
  Provide cloud credentials (`RAGMILL_PINECONE_API_KEY`, `RAGMILL_QDRANT_API_KEY`,
  `OPENAI_API_KEY`, `GEMINI_API_KEY`) via environment variables or a secrets
  manager.
- **Untrusted documents.** Content you ingest is passed to an LLM at chat time.
  Treat ingested documents as untrusted input (prompt-injection surface) if they
  come from third parties.
- **Resource limits.** For public deployments, put request-size and rate limits
  at the proxy layer.

Thank you for helping keep RAGMill and its users safe.
