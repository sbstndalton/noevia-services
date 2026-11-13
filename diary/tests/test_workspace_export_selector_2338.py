"""#2338: an empty WORKSPACE_EXPORT_IMPL (compose passthrough) means the default, not an error. Synthetic values only."""
from types import SimpleNamespace

import pytest

import agent.app as appmod
from agent import rust_forward
from agent import workspace_export_native as native


@pytest.mark.parametrize('raw', [None, '', '   ', '\t'])
def test_unset_empty_or_blank_selects_python(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv('WORKSPACE_EXPORT_IMPL', raising=False)
    else:
        monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', raw)
    assert native.impl_value() == 'python'
    assert native.selected() is False


@pytest.mark.parametrize('raw, expected', [('python', False), ('rust', True)])
def test_named_values_are_unchanged(monkeypatch, raw, expected):
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', raw)
    assert native.impl_value() == raw
    assert native.selected() is expected


@pytest.mark.parametrize('raw', ['native', 'RUST', ' rust'])
def test_other_values_still_raise(monkeypatch, raw):
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', raw)
    with pytest.raises(ValueError):
        native.selected()


def test_forwarder_sends_the_normalised_selector_to_rust(monkeypatch):
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', '')
    # A stand-in state: the app's own state carries whatever client stubs earlier tests installed.
    llm = SimpleNamespace(base_url='http://synthetic.invalid', chat_model='', fallback_model=None,
                          setting_key='llm.chat_model', fallback_setting_key=None)
    state = SimpleNamespace(auth_token='t', llm_main=llm, llm_aux=llm)
    cfg = rust_forward.server_config(appmod._base_cfg, state, 8011, 't', 'f')
    assert cfg['workspace_export']['impl'] == 'python'
