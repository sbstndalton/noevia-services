"""Explicit provisioning only. Never called by the serving process."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = '1c5edc17a7acd8701df6fc341c0d179f1c62c982'
snapshot_download('convaiinnovations/laya', revision=REVISION, local_dir='/model', max_workers=1,
                  allow_patterns=['model.safetensors', 'rl_agent_config.json', 'encoder/config.json', 'tokenizer/*'])
# Apply the SDK's tokenizer compatibility normalization before the read-only runtime mount.
p = Path('/model/tokenizer/tokenizer_config.json')
cfg = json.loads(p.read_text())
if cfg.get('tokenizer_class') in (None, 'TokenizersBackend'):
    cfg['tokenizer_class'] = 'PreTrainedTokenizerFast'
    cfg.pop('backend', None)
    cfg.pop('is_local', None)
if isinstance(cfg.get('extra_special_tokens'), list):
    cfg['extra_special_tokens'] = {f'extra_{i}': v for i, v in enumerate(cfg['extra_special_tokens'])}
p.write_text(json.dumps(cfg, indent=2))
Path('/model/NOEVIA_REVISION').write_text(REVISION+'\n')
print('Pinned English checkpoint downloaded; no inference performed.')
