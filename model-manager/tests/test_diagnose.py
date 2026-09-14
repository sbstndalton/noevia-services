from app.diagnose import analyse


def spawn(name, port):
    return f"1.00 I srv  load: spawning server instance with name={name} on port {port}"


def exited(name, status):
    return f"2.00 I srv  run: instance name={name} exited with status {status}"


def test_out_of_memory_is_named_with_evidence():
    log = "\n".join([spawn("big", 5001), "[5001] 0.1 I load_model: loading",
                     "[5001] 0.2 E ggml_vulkan: Device memory allocation of size 9000000000 failed.",
                     "[5001] 0.3 E llama_init: failed to initialize", exited("big", 1)])
    [d] = analyse(log)
    assert d.model == "big" and d.cause == "gpu-memory" and d.status == 1
    assert any("allocation" in e for e in d.evidence)
    assert "context size" in d.advice


def test_requested_stops_are_not_failures():
    log = "\n".join([spawn("m", 5002), "[5002] llama_server: model loaded",
                     "3.0 I srv  unload: stopping model instance name=m",
                     "4.0 W srv  run: force-killing model instance name=m after timeout", exited("m", 1)])
    assert analyse(log) == []


def test_a_later_successful_load_clears_the_failure_but_a_crash_after_loading_does_not():
    crash = "\n".join([spawn("m", 5003), "[5003] llama_server: model loaded",
                       "[5003] E cudaMalloc failed: out of memory", exited("m", 134)])
    assert analyse(crash)[0].cause == "gpu-memory"
    recovered = crash + "\n" + "\n".join([spawn("m", 5004), "[5004] llama_server: model loaded"])
    assert analyse(recovered) == []


def test_other_causes_and_unknown():
    cases = {
        "missing-file": "[6000] gguf_init_from_file: failed to open GGUF file '/models/x.gguf' (No such file or directory)",
        "corrupt-file": "[6000] gguf_init_from_file_impl: invalid magic characters: 'abcd'",
        "architecture": "[6000] llama_model_load: error loading model: unknown model architecture: 'qwen9'",
        "bad-option": "[6000] error: invalid argument: --flash-attn maybe",
        "kv-cache": "[6000] llama_kv_cache: failed to allocate buffer for kv cache",
    }
    for cause, line in cases.items():
        [d] = analyse("\n".join([spawn("x", 6000), line, exited("x", 1)]))
        assert d.cause == cause, (cause, d.cause)
    [d] = analyse("\n".join([spawn("y", 6001), "[6001] something odd happened", exited("y", 2)]))
    assert d.cause == "unknown" and d.evidence == ["something odd happened"]
