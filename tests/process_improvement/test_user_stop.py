"""Explicit user shutdown cannot be silently reversed by an old scheduler."""
import pytest

from scripts.process_improvement import controller_control


@pytest.mark.parametrize('action',['start','restart'])
def test_persisted_user_stop_rejects_automatic_restart(tmp_path,monkeypatch,action):
    monkeypatch.setattr(controller_control,'ROOT',tmp_path)
    path=tmp_path/'controller/user_stop.json';path.parent.mkdir()
    path.write_text('{"status":"STOP_REQUESTED_BY_USER"}')
    with pytest.raises(RuntimeError,match='User stopped this study'):
        controller_control.require_restart_authorization(action)


@pytest.mark.parametrize('action',['stop','status'])
def test_stop_and_status_remain_available_after_shutdown(tmp_path,monkeypatch,action):
    monkeypatch.setattr(controller_control,'ROOT',tmp_path)
    path=tmp_path/'controller/user_stop.json';path.parent.mkdir()
    path.write_text('{}')
    controller_control.require_restart_authorization(action)


def test_fresh_authorized_study_can_start(tmp_path,monkeypatch):
    monkeypatch.setattr(controller_control,'ROOT',tmp_path)
    controller_control.require_restart_authorization('start')
