import pytest

from nanovllm_pp.server.__main__ import main, parser


def test_defaults_are_single_model_localhost_paged():
    args = parser().parse_args(["./Qwen3-0.6B"])
    assert (args.backend, args.device, args.host) == ("paged", "cuda", "127.0.0.1")
    assert args.enable_chunked_prefill
    assert not args.enable_prefix_caching


@pytest.mark.parametrize("options", [
    ["--backend", "hf", "--enable-prefix-caching"],
    ["--backend", "paged", "--load-in-4bit"],
    ["--port", "0"],
    ["--load-in-4bit", "--load-in-8bit"],
])
def test_invalid_cli_exits_before_model_load(options):
    with pytest.raises(SystemExit) as error:
        main(["nonexistent-model", *options])
    assert error.value.code == 2


def test_launcher_passes_options_to_engine_and_server(monkeypatch):
    pytest.importorskip("fastapi")
    import nanovllm_pp
    from nanovllm_pp.server import app

    observed = {}
    sentinel = object()

    def engine(model, **kwargs):
        observed.update(model=model, config=kwargs)
        return sentinel

    def serve(instance, sampling, **kwargs):
        assert instance is sentinel
        assert sampling is nanovllm_pp.SamplingParams
        observed.update(server=kwargs)

    monkeypatch.setattr(nanovllm_pp, "LLM", engine)
    monkeypatch.setattr(app, "start_server", serve)
    main(["local-model", "--enable-prefix-caching", "--port", "8080"])
    assert observed["model"] == "local-model"
    assert observed["config"]["enable_prefix_caching"]
    assert observed["server"] == {"host": "127.0.0.1", "port": 8080}
