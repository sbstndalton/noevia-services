import os
import struct
import sys
import tempfile
from pathlib import Path

# Settings are read at import, so the synthetic directories must exist first.
ROOT = Path(tempfile.mkdtemp(prefix="model-manager-test-"))
(ROOT / "models" / "tiny").mkdir(parents=True)
(ROOT / "data").mkdir()
os.environ.update(MODELS_DIR=str(ROOT / "models"), MODELS_INI_PATH=str(ROOT / "models" / "models.ini"),
                  DATA_DIR=str(ROOT / "data"), LLAMA_CONTAINERS="", HOST_RAM_RESERVE_GB="4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _gguf(kv: dict) -> bytes:
    def s(x):
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<Q", 0), struct.pack("<Q", len(kv))]
    for k, v in kv.items():
        out.append(s(k))
        if isinstance(v, str):
            out += [struct.pack("<I", 8), s(v)]
        else:
            out += [struct.pack("<I", 4), struct.pack("<I", v)]
    return b"".join(out)


(ROOT / "models" / "tiny" / "tiny-Q4_K_M.gguf").write_bytes(_gguf({
    "general.architecture": "llama", "llama.context_length": 8192, "llama.embedding_length": 256,
    "llama.block_count": 4, "llama.attention.head_count": 4, "llama.attention.head_count_kv": 2,
    "tokenizer.chat_template": "{{ messages }}"}) + b"\0" * 4096)
INI = "version = 1\n\n[tiny]\nmodel = /models/tiny/tiny-Q4_K_M.gguf\nctx-size = 4096\n"
(ROOT / "models" / "models.ini").write_text(INI)
