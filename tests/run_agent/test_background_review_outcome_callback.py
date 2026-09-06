"""Outcome callback tests for `_run_review_in_thread`.

Guards that every successfully completed background review emits a callback —
both when actions are returned and when there are none — and that a review
that raises inside `run_conversation` never produces a success callback.

All dependencies are monkeypatched: no real thread, network, filesystem or
provider. The real `_run_review_in_thread` function is exercised.
"""
from __future__ import annotations

from agent import background_review as bg_review
from tools import terminal_tool as tt


class _Agent:
    def __init__(self):
        self.model = "fake-model"
        self.platform = "telegram"
        self.provider = "openai"
        self.base_url = ""
        self.api_key = ""
        self.api_mode = ""
        self.session_id = "test-session"
        self._credential_pool = None
        self.request_overrides = {}
        self.max_tokens = 1000
        self.acp_command = None
        self.acp_args = []
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self.reasoning_config = None
        self._cached_system_prompt = "cached-prompt"
        import datetime as _dt
        self.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
        self._memory_store = object()
        self._memory_enabled = True
        self._user_profile_enabled = False
        self.memory_notifications = "on"
        self.background_review_callback = None
        self._safe_print = lambda *_a, **_k: None
        self._emit_auxiliary_failure = lambda *_a, **_k: None


class _FakeReviewAgent:
    def __init__(self, **kwargs):
        self._session_messages = []

    def run_conversation(self, **kwargs):
        pass

    def shutdown_memory_provider(self):
        pass

    def close(self):
        pass


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install(monkeypatch, *, summarize_result, run_raises=None):
    """Wire every dependency of `_run_review_in_thread` to stubs."""
    monkeypatch.setattr(
        bg_review, "thread_scoped_silence", lambda: _NullContext()
    )
    monkeypatch.setattr(
        bg_review, "_resolve_review_runtime",
        lambda agent, task_cfg=None: {"routed": False, "model": "fake-model",
                       "provider": "openai", "api_mode": "", "base_url": "",
                       "api_key": "", "credential_pool": None,
                       "request_overrides": {}, "max_tokens": 1000,
                       "command": None, "args": []},
    )
    import run_agent as run_agent_module
    monkeypatch.setattr(run_agent_module, "AIAgent", _FakeReviewAgent)
    monkeypatch.setattr(tt, "set_approval_callback", lambda *_a, **_k: None)

    def _digest(msgs, **kwargs):
        return msgs
    monkeypatch.setattr(bg_review, "_digest_history", _digest)

    import model_tools
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kwargs: [],
    )

    import hermes_cli.plugins as plugins
    monkeypatch.setattr(plugins, "set_thread_tool_whitelist", lambda *_a, **_k: None)
    monkeypatch.setattr(plugins, "clear_thread_tool_whitelist", lambda *_a, **_k: None)

    import tools.skill_manager_guards as smg
    monkeypatch.setattr(
        smg, "_reset_background_review_read_marks", lambda: None,
    )

    events = {}

    def _run_conversation(self, **kw):
        if run_raises is not None:
            raise run_raises
        events["_session_messages"] = []

    monkeypatch.setattr(_FakeReviewAgent, "run_conversation", _run_conversation)

    def _summarize(review_messages, prior_snapshot, notification_mode="on"):
        return summarize_result
    monkeypatch.setattr(
        bg_review, "summarize_background_review_actions", _summarize,
    )
    return events


def test_callback_with_actions(monkeypatch):
    _install(monkeypatch, summarize_result=["memory updated"])
    agent = _Agent()
    printed = []
    failures = []
    cb = []
    agent._safe_print = lambda *a, **_k: printed.append(" ".join(str(x) for x in a))
    agent._emit_auxiliary_failure = lambda *_a, **_k: failures.append(_a)
    agent.background_review_callback = lambda msg: cb.append(msg)

    bg_review._run_review_in_thread(
        agent, [{"role": "user", "content": "hi"}], "review"
    )

    assert cb == ["💾 Self-improvement review: memory updated"]
    assert printed == ["  💾 Self-improvement review: memory updated"]
    assert failures == []


def test_callback_with_no_actions(monkeypatch):
    _install(monkeypatch, summarize_result=[])
    agent = _Agent()
    printed = []
    failures = []
    cb = []
    agent._safe_print = lambda *a, **_k: printed.append(" ".join(str(x) for x in a))
    agent._emit_auxiliary_failure = lambda *_a, **_k: failures.append(_a)
    agent.background_review_callback = lambda msg: cb.append(msg)

    bg_review._run_review_in_thread(
        agent, [{"role": "user", "content": "hi"}], "review"
    )

    assert cb == ["💾 Self-improvement review: completata — nessuna modifica necessaria"]
    assert printed == [
        "  💾 Self-improvement review: completata — nessuna modifica necessaria"
    ]
    assert failures == []


def test_error_never_calls_callback(monkeypatch):
    _install(monkeypatch, summarize_result=[], run_raises=RuntimeError("probe"))
    agent = _Agent()
    printed = []
    failures = []
    cb = []
    agent._safe_print = lambda *a, **_k: printed.append(" ".join(str(x) for x in a))
    agent._emit_auxiliary_failure = lambda *a, **_k: failures.append(a)
    agent.background_review_callback = lambda msg: cb.append(msg)

    bg_review._run_review_in_thread(
        agent, [{"role": "user", "content": "hi"}], "review"
    )

    assert cb == []
    assert len(failures) == 1
    assert failures[0][0] == "background review"
    assert printed == []
