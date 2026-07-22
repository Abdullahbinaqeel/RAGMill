"""Tests for the ragmill CLI (ragmill.__main__).

Drives main() in-process by monkeypatching sys.argv (no subprocess).
serve/configure are tested by monkeypatching uvicorn.run to a no-op and
asserting the wiring — never binds a real port. chat is tested with a
scripted stdin and a mocked generator — never loads the real model.
"""

import logging
import sys

import pytest

from ragmill import __main__ as cli


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["ragmill"] + argv)
    cli.main()


def test_ingest_and_count(monkeypatch, tmp_path, caplog):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("hello world, this is a test document.")
    monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "cli.db"))

    with caplog.at_level(logging.INFO):
        _run(monkeypatch, ["ingest", str(tmp_path / "docs")])
    assert "Ingested" in caplog.text
    assert "chunks from" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        _run(monkeypatch, ["count"])
    assert caplog.text.strip().isdigit() or any(c.isdigit() for c in caplog.text)


def test_ingest_missing_directory_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "cli.db"))
    with pytest.raises(FileNotFoundError):
        _run(monkeypatch, ["ingest", str(tmp_path / "nonexistent")])


def test_search_no_results(monkeypatch, tmp_path, caplog):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "cli.db"))

    with caplog.at_level(logging.INFO):
        _run(monkeypatch, ["search", "xyznonexistentkeyword12345"])
    assert "No results found." in caplog.text


def test_export_import_roundtrip(monkeypatch, tmp_path, caplog):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("some content to export and import back.")
    monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "cli.db"))

    with caplog.at_level(logging.INFO):
        _run(monkeypatch, ["ingest", str(tmp_path / "docs")])
    caplog.clear()

    export_path = str(tmp_path / "export.jsonl")
    with caplog.at_level(logging.INFO):
        _run(monkeypatch, ["export", export_path])
    assert "Exported" in caplog.text

    monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "fresh.db"))
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _run(monkeypatch, ["import", export_path])
    assert "Imported" in caplog.text


def test_import_nonexistent_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "cli.db"))
    with pytest.raises(FileNotFoundError):
        _run(monkeypatch, ["import", str(tmp_path / "nonexistent.jsonl")])


def test_chat_repl_uses_mocked_generator(monkeypatch, tmp_path, caplog, mock_llm):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setattr(cli, "generate_answer", mock_llm)

    responses = iter(["what is this?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    with caplog.at_level(logging.INFO):
        _run(monkeypatch, ["chat"])
    assert "MOCKED ANSWER for: what is this?" in caplog.text
    assert len(mock_llm.calls) == 1


def test_chat_repl_exits_on_eof(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "cli.db"))

    def _raise_eof(prompt=""):
        raise EOFError()

    monkeypatch.setattr("builtins.input", _raise_eof)
    with caplog.at_level(logging.INFO):
        _run(monkeypatch, ["chat"])  # should not hang or raise
    assert "RAGMill chat" in caplog.text


def test_serve_wires_uvicorn_without_binding(monkeypatch):
    calls = {}

    def _fake_run(app, host, port, reload):
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port
        calls["reload"] = reload

    monkeypatch.setattr("uvicorn.run", _fake_run)
    _run(monkeypatch, ["serve"])

    assert calls["app"] == "ragmill.server:app"
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8000
    assert calls["reload"] is False


def test_configure_defaults_to_localhost_and_port_8090(monkeypatch):
    calls = {}

    def _fake_run(app, host, port):
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setattr("uvicorn.run", _fake_run)
    _run(monkeypatch, ["configure"])

    assert calls["app"] == "ragmill.config_ui:config_app"
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8090


def test_configure_respects_env_path_override(monkeypatch, tmp_path):
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    custom_path = str(tmp_path / "custom.env")

    _run(monkeypatch, ["configure", "--env-path", custom_path])

    import os

    assert os.environ["RAGMILL_ENV_PATH"] == custom_path


def test_help_and_invalid_command(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ragmill", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "ingest" in out
    assert "configure" in out

    monkeypatch.setattr(sys, "argv", ["ragmill", "invalidcommand"])
    with pytest.raises(SystemExit):
        cli.main()
