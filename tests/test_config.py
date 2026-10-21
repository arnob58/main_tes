import pytest

from tessera.config import Config, load_config


def test_defaults_match_spec():
    c = Config()
    assert c.context_window == 4096
    assert c.context_recovery_threshold == 0.90
    assert c.active_message_limit == 5
    assert c.embedding_model == "sentence-transformers/paraphrase-MiniLM-L3-v2"
    assert c.pipelines["complex"] == ["interpret", "develop", "evaluate", "respond"]


def test_toml_then_env_override(tmp_path):
    p = tmp_path / "t.toml"
    p.write_text('context_window = 8192\nmodel_path = "m.gguf"\n')
    c = load_config(p, env={"TESSERA_CONTEXT_WINDOW": "2048", "TESSERA_RESET_KV_BETWEEN_CALLS": "yes"})
    assert c.model_path == "m.gguf"
    assert c.context_window == 2048          # env beats file
    assert c.reset_kv_between_calls is True


def test_unknown_toml_key_rejected(tmp_path):
    p = tmp_path / "t.toml"
    p.write_text("contxt_window = 1\n")
    with pytest.raises(ValueError, match="contxt_window"):
        load_config(p, env={})


def test_missing_file_is_fine():
    assert load_config("does-not-exist.toml", env={}) == Config()
