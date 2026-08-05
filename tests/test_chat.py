"""Tests for ragmill.chat — local LLM answer generation.

The real GGUF model (~1.1GB) is never downloaded in these tests; the
Llama class and the download helper are fully mocked. Set
RAGMILL_CHAT_TEST_REAL=1 to additionally run a real, slow, opt-in
end-to-end test that downloads and loads the actual model.
"""

import os

import pytest

from ragmill import chat

# ── _format_context ─────────────────────────────────────────────────────────


def test_format_context_includes_filename_and_content():
    chunks = [
        {"content": "alpha content", "metadata": {"filename": "a.txt"}},
        {"content": "beta content", "metadata": {"filename": "b.txt"}},
    ]
    formatted = chat._format_context(chunks)
    assert "(Source: a.txt)" in formatted
    assert "alpha content" in formatted
    assert "(Source: b.txt)" in formatted
    assert "beta content" in formatted


def test_format_context_defaults_filename_to_unknown():
    formatted = chat._format_context([{"content": "x", "metadata": {}}])
    assert "Source: unknown" in formatted


def test_format_context_empty_list():
    assert chat._format_context([]) == ""


# ── generate_answer (mocked Llama) ──────────────────────────────────────────


class _FakeLlama:
    def __init__(self, model_path, n_ctx, verbose):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.last_messages = None

    def create_chat_completion(self, messages, temperature, max_tokens):
        self.last_messages = messages
        return {"choices": [{"message": {"content": "a mocked grounded answer [facts.txt]"}}]}


@pytest.fixture(autouse=True)
def _clear_llm_cache():
    chat._llm_cache.clear()
    yield
    chat._llm_cache.clear()


def test_generate_answer_uses_mocked_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(
        chat, "_download_gguf", lambda repo, filename, cache_dir: tmp_path / filename
    )
    import sys
    import types

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    chunks = [{"content": "Sprocket is the mascot.", "metadata": {"filename": "facts.txt"}}]
    answer = chat.generate_answer("who is the mascot?", chunks)

    assert answer == "a mocked grounded answer [facts.txt]"


def test_generate_answer_no_chunks_still_calls_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(
        chat, "_download_gguf", lambda repo, filename, cache_dir: tmp_path / filename
    )

    import sys
    import types

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    answer = chat.generate_answer("anything?", [])
    assert answer == "a mocked grounded answer [facts.txt]"


def test_generate_answer_passes_system_and_user_messages(monkeypatch, tmp_path):
    monkeypatch.setattr(
        chat, "_download_gguf", lambda repo, filename, cache_dir: tmp_path / filename
    )

    import sys
    import types

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    chunks = [{"content": "some fact", "metadata": {"filename": "doc.txt"}}]
    chat.generate_answer("a question", chunks)

    llm = next(iter(chat._llm_cache.values()))
    roles = [m["role"] for m in llm.last_messages]
    assert roles == ["system", "user"]
    assert chat.SYSTEM_PROMPT in llm.last_messages[0]["content"]
    assert "a question" in llm.last_messages[1]["content"]
    assert "some fact" in llm.last_messages[1]["content"]


def test_generate_answer_reuses_cached_llm_across_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(
        chat, "_download_gguf", lambda repo, filename, cache_dir: tmp_path / filename
    )

    load_count = {"n": 0}

    class _CountingFakeLlama(_FakeLlama):
        def __init__(self, model_path, n_ctx, verbose):
            load_count["n"] += 1
            super().__init__(model_path, n_ctx, verbose)

    import sys
    import types

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _CountingFakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    chat.generate_answer("q1", [])
    chat.generate_answer("q2", [])
    assert load_count["n"] == 1, "the model should only be loaded once and reused"


def test_generate_answer_env_var_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGMILL_CHAT_MODEL_REPO", "some/other-repo")
    monkeypatch.setenv("RAGMILL_CHAT_MODEL_FILE", "other-file.gguf")
    monkeypatch.setenv("RAGMILL_CHAT_N_CTX", "2048")

    seen = {}

    def _fake_download(repo, filename, cache_dir):
        seen["repo"] = repo
        seen["filename"] = filename
        return tmp_path / filename

    monkeypatch.setattr(chat, "_download_gguf", _fake_download)

    import sys
    import types

    fake_module = types.ModuleType("llama_cpp")

    class _CtxCapturingFakeLlama(_FakeLlama):
        pass

    fake_module.Llama = _CtxCapturingFakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    chat.generate_answer("q", [])

    assert seen["repo"] == "some/other-repo"
    assert seen["filename"] == "other-file.gguf"
    llm = next(iter(chat._llm_cache.values()))
    assert llm.n_ctx == 2048


def test_missing_llama_cpp_raises_helpful_import_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        chat, "_download_gguf", lambda repo, filename, cache_dir: tmp_path / filename
    )

    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("No module named 'llama_cpp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    # Points at the prebuilt wheel, not `ragmill[chat]` — that is a source build
    # and is exactly what fails on Windows.
    with pytest.raises(ImportError, match="llama-cpp-python --extra-index-url"):
        chat.generate_answer("q", [])


# ── _download_gguf (network mocked) ─────────────────────────────────────────


def test_download_gguf_skips_if_already_cached(tmp_path, monkeypatch):
    cache_dir = tmp_path / "models"
    target_dir = cache_dir / "some__repo"
    target_dir.mkdir(parents=True)
    existing = target_dir / "model.gguf"
    existing.write_bytes(b"already here")

    def _boom(*args, **kwargs):
        raise AssertionError("should not attempt a download when the file is already cached")

    monkeypatch.setattr("urllib.request.urlretrieve", _boom)

    result = chat._download_gguf("some/repo", "model.gguf", cache_dir)
    assert result == existing
    assert result.read_bytes() == b"already here"


def test_download_gguf_retries_and_cleans_up_partial_on_failure(tmp_path, monkeypatch):
    cache_dir = tmp_path / "models"
    attempts = {"n": 0}

    def _flaky_urlretrieve(url, path):
        attempts["n"] += 1
        raise OSError("simulated transient DNS failure")

    monkeypatch.setattr("urllib.request.urlretrieve", _flaky_urlretrieve)

    with pytest.raises(ConnectionError):
        chat._download_gguf("some/repo", "model.gguf", cache_dir, retries=3)

    assert attempts["n"] == 3
    target_dir = cache_dir / "some__repo"
    # no partial or final file should be left behind
    assert not any(target_dir.iterdir()) if target_dir.exists() else True


def test_download_gguf_succeeds_after_transient_failure(tmp_path, monkeypatch):
    cache_dir = tmp_path / "models"
    attempts = {"n": 0}

    def _flaky_then_ok(url, path):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise OSError("simulated transient DNS failure")
        path.write_bytes(b"downloaded content")

    monkeypatch.setattr("urllib.request.urlretrieve", _flaky_then_ok)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    result = chat._download_gguf("some/repo", "model.gguf", cache_dir, retries=3)
    assert attempts["n"] == 2
    assert result.read_bytes() == b"downloaded content"


# ── Backend dispatch (RAGMILL_CHAT_BACKEND) ─────────────────────────────────


def test_default_backend_is_local(monkeypatch, tmp_path):
    monkeypatch.delenv("RAGMILL_CHAT_BACKEND", raising=False)
    monkeypatch.setattr(
        chat, "_download_gguf", lambda repo, filename, cache_dir: tmp_path / filename
    )

    import sys
    import types

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    answer = chat.generate_answer("q", [])
    assert answer == "a mocked grounded answer [facts.txt]"


def test_unknown_backend_raises_value_error(monkeypatch):
    monkeypatch.setenv("RAGMILL_CHAT_BACKEND", "not-a-real-backend")
    with pytest.raises(ValueError, match="not-a-real-backend"):
        chat.generate_answer("q", [])


# ── Gemini backend (mocked google.genai) ────────────────────────────────────


class _FakeGeminiResponse:
    def __init__(self, text):
        self.text = text


class _FakeGeminiModels:
    def __init__(self):
        self.last_call = None

    def generate_content(self, model, contents, config):
        self.last_call = {"model": model, "contents": contents, "config": config}
        return _FakeGeminiResponse("a mocked gemini answer [facts.txt]")


class _FakeGeminiClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.models = _FakeGeminiModels()


def _install_fake_genai(monkeypatch):
    import sys
    import types

    fake_types_module = types.ModuleType("google.genai.types")
    fake_types_module.GenerateContentConfig = lambda **kwargs: kwargs

    fake_genai_module = types.ModuleType("google.genai")
    fake_genai_module.Client = _FakeGeminiClient
    fake_genai_module.types = fake_types_module

    fake_google_module = types.ModuleType("google")
    fake_google_module.genai = fake_genai_module

    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types_module)


def test_gemini_backend_uses_mocked_client(monkeypatch):
    monkeypatch.setenv("RAGMILL_CHAT_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    _install_fake_genai(monkeypatch)

    chunks = [{"content": "Sprocket is the mascot.", "metadata": {"filename": "facts.txt"}}]
    answer = chat.generate_answer("who is the mascot?", chunks)

    assert answer == "a mocked gemini answer [facts.txt]"


def test_gemini_backend_env_var_model_override(monkeypatch):
    monkeypatch.setenv("RAGMILL_CHAT_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("RAGMILL_GEMINI_MODEL", "gemini-custom-model")
    _install_fake_genai(monkeypatch)

    import sys

    client_cls = sys.modules["google.genai"].Client
    original_init = client_cls.__init__
    created = {}

    def _capturing_init(self, api_key=None):
        original_init(self, api_key=api_key)
        created["client"] = self

    monkeypatch.setattr(client_cls, "__init__", _capturing_init)

    chat.generate_answer("q", [])

    assert created["client"].api_key == "fake-key"
    assert created["client"].models.last_call["model"] == "gemini-custom-model"


def test_missing_google_genai_raises_helpful_import_error(monkeypatch):
    monkeypatch.setenv("RAGMILL_CHAT_BACKEND", "gemini")

    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name in ("google", "google.genai"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    with pytest.raises(ImportError, match="ragmill\\[chat-gemini\\]"):
        chat.generate_answer("q", [])


# ── OpenAI (ChatGPT) backend (mocked openai) ────────────────────────────────


class _FakeOpenAIMessage:
    def __init__(self, content):
        self.content = content


class _FakeOpenAIChoice:
    def __init__(self, content):
        self.message = _FakeOpenAIMessage(content)


class _FakeOpenAIResponse:
    def __init__(self, content):
        self.choices = [_FakeOpenAIChoice(content)]


class _FakeOpenAICompletions:
    def __init__(self):
        self.last_call = None

    def create(self, model, messages):
        self.last_call = {"model": model, "messages": messages}
        return _FakeOpenAIResponse("a mocked openai answer [facts.txt]")


class _FakeOpenAIChat:
    def __init__(self):
        self.completions = _FakeOpenAICompletions()


class _FakeOpenAIClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chat = _FakeOpenAIChat()


def _install_fake_openai(monkeypatch):
    import sys
    import types

    fake_openai_module = types.ModuleType("openai")
    fake_openai_module.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)


def test_openai_backend_uses_mocked_client(monkeypatch):
    monkeypatch.setenv("RAGMILL_CHAT_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    _install_fake_openai(monkeypatch)

    chunks = [{"content": "Sprocket is the mascot.", "metadata": {"filename": "facts.txt"}}]
    answer = chat.generate_answer("who is the mascot?", chunks)

    assert answer == "a mocked openai answer [facts.txt]"


def test_openai_backend_passes_system_and_user_messages(monkeypatch):
    monkeypatch.setenv("RAGMILL_CHAT_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    _install_fake_openai(monkeypatch)

    import sys

    client_cls = sys.modules["openai"].OpenAI
    original_init = client_cls.__init__
    created = {}

    def _capturing_init(self, api_key=None):
        original_init(self, api_key=api_key)
        created["client"] = self

    monkeypatch.setattr(client_cls, "__init__", _capturing_init)

    chat.generate_answer(
        "a question", [{"content": "some fact", "metadata": {"filename": "doc.txt"}}]
    )

    call = created["client"].chat.completions.last_call
    roles = [m["role"] for m in call["messages"]]
    assert roles == ["system", "user"]
    assert "a question" in call["messages"][1]["content"]
    assert "some fact" in call["messages"][1]["content"]


def test_missing_openai_raises_helpful_import_error(monkeypatch):
    monkeypatch.setenv("RAGMILL_CHAT_BACKEND", "openai")

    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    with pytest.raises(ImportError, match="ragmill\\[chat-openai\\]"):
        chat.generate_answer("q", [])


# ── Optional real end-to-end tests (slow / need real credentials) ─────────


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.getenv("RAGMILL_CHAT_TEST_REAL_GEMINI") == "1"
        and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    ),
    reason="set RAGMILL_CHAT_TEST_REAL_GEMINI=1 and GEMINI_API_KEY to run a real Gemini call",
)
def test_generate_answer_real_gemini(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("RAGMILL_CHAT_BACKEND", "gemini")
    chunks = [
        {
            "content": "The mascot of RAGMill is a small mechanical owl named Sprocket.",
            "metadata": {"filename": "facts.txt"},
        }
    ]
    answer = chat.generate_answer("who is the mascot?", chunks)
    assert isinstance(answer, str) and len(answer) > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("RAGMILL_CHAT_TEST_REAL_OPENAI") == "1" and os.getenv("OPENAI_API_KEY")),
    reason="set RAGMILL_CHAT_TEST_REAL_OPENAI=1 and OPENAI_API_KEY to run a real OpenAI call",
)
def test_generate_answer_real_openai(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("RAGMILL_CHAT_BACKEND", "openai")
    chunks = [
        {
            "content": "The mascot of RAGMill is a small mechanical owl named Sprocket.",
            "metadata": {"filename": "facts.txt"},
        }
    ]
    answer = chat.generate_answer("who is the mascot?", chunks)
    assert isinstance(answer, str) and len(answer) > 0


# ── Optional real end-to-end test (slow, downloads real weights) ───────────


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RAGMILL_CHAT_TEST_REAL") != "1",
    reason="set RAGMILL_CHAT_TEST_REAL=1 to run the real (slow, ~1.1GB download) local model test",
)
def test_generate_answer_real_local_model():
    pytest.importorskip("llama_cpp")
    chat._llm_cache.clear()
    chunks = [
        {
            "content": "The mascot of RAGMill is a small mechanical owl named Sprocket.",
            "metadata": {"filename": "facts.txt"},
        }
    ]
    answer = chat.generate_answer("who is the mascot?", chunks)
    assert isinstance(answer, str) and len(answer) > 0


# ── Missing local model: guide to the install, do not derail to a backend ────


class TestLocalBackendMissingFlow:
    """A missing optional install is a routine state, not a crash.

    The local backend is the default, so this is what a new user hits first.
    The message must hand them the one command that works (a prebuilt wheel),
    not steer them to Gemini/OpenAI — they asked for local chat — and not print
    a traceback for something that is simply not installed yet.
    """

    def test_message_leads_with_the_install_command(self):
        msg = chat.LOCAL_BACKEND_MISSING
        # `ragmill setup-chat` is the easy path, so it comes first; the raw pip
        # command follows for anyone who would rather run it themselves.
        assert "ragmill setup-chat" in msg
        assert chat.LLAMA_INSTALL_COMMAND in msg
        assert msg.index("ragmill setup-chat") < msg.index(chat.LLAMA_INSTALL_COMMAND)

    def test_manual_command_in_the_message_forces_a_wheel(self):
        """The message must not hand out the command that compiles from source."""
        assert "--only-binary llama-cpp-python" in chat.LLAMA_INSTALL_COMMAND

    def test_message_does_not_push_a_different_backend(self):
        msg = chat.LOCAL_BACKEND_MISSING.lower()
        for other in ("gemini", "openai", "chat-gemini", "chat-openai"):
            assert other not in msg, f"local-model error mentions {other}"

    def test_check_is_silent_when_llama_cpp_is_importable(self, monkeypatch):
        import sys
        import types

        monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
        chat.check_backend_available()  # must not raise

    def test_check_raises_before_any_work_when_missing(self, monkeypatch):
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        monkeypatch.delenv("RAGMILL_CHAT_BACKEND", raising=False)

        with pytest.raises(ImportError) as excinfo:
            chat.check_backend_available()

        assert chat.LLAMA_INSTALL_COMMAND in str(excinfo.value)

    def test_check_skips_hosted_backends(self, monkeypatch):
        """A Gemini user must not be told to install a local model."""
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        monkeypatch.setenv("RAGMILL_CHAT_BACKEND", "gemini")
        chat.check_backend_available()  # must not raise

    def test_cli_exits_cleanly_instead_of_raising(self, monkeypatch, tmp_path):
        """No traceback, exit code 1, message on the way out."""
        from ragmill import __main__ as cli

        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        monkeypatch.delenv("RAGMILL_CHAT_BACKEND", raising=False)
        monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "c.db"))

        class Args:
            top_k = 5

        with pytest.raises(SystemExit) as excinfo:
            cli.cmd_chat(Args())

        assert chat.LLAMA_INSTALL_COMMAND in str(excinfo.value)
        assert "Traceback" not in str(excinfo.value)
