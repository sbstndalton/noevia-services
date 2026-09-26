from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    models_dir: Path = Path("/models")
    # Host folder mounted at models_dir, shown to the user as where downloads land.
    models_host_path: str = ""
    # Extra download destinations: comma-separated folder names directly under models_dir
    # (e.g. another share mounted at /models/archive). Mount points there are offered too.
    model_download_targets: str = ""
    # Shared secret the web proxy sends. When set, every route except /api/v1/health needs it:
    # this service holds the Docker socket, so network reachability alone must not be enough.
    model_loader_token: str = ""
    models_ini_path: Path = Path("/models/models.ini")
    data_dir: Path = Path("/data")
    llama_containers: str = ""  # empty = auto-discover any ghcr.io/ggml-org/llama.cpp:* container
    gpu_vram: str = ""  # optional container_name:vram_gib overrides — auto-probed via nvidia-smi/rocm-smi if empty
    bind_port: int = 8090
    # Fallback internal port used to probe a llama.cpp container when Docker metadata has none
    # (issue #341: an internal-network-only container publishes no port, so
    # NetworkSettings.Ports is {} and Config.ExposedPorts is null). Only used when the
    # container's own command/env doesn't say otherwise. 8080 is llama.cpp server's default;
    # set to 0 to disable the fallback entirely (probe reports probe_error="port unknown").
    llama_default_port: int = 8080
    max_concurrent_downloads: int = 2
    # RAM the OS, page cache and everything else on the box need. Subtracted before deciding
    # whether a model fits in system memory, because total RAM is never all yours: sizing
    # against it produces plans that swap or get OOM-killed. Raise it on a busy host.
    host_ram_reserve_gb: float = 32.0

    @field_validator("model_loader_token")
    @classmethod
    def validate_model_loader_token(cls, value: str) -> str:
        if not value:
            return ""
        token = value.strip()
        if len(token) < 32:
            raise ValueError("MODEL_LOADER_TOKEN must contain at least 32 non-whitespace characters")
        return token

    @property
    def llama_container_names(self) -> list[str]:
        return [n.strip() for n in self.llama_containers.split(",") if n.strip()]

    @property
    def gpu_vram_map(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for pair in self.gpu_vram.split(","):
            pair = pair.strip()
            if ":" not in pair:
                continue
            name, val = pair.split(":", 1)
            try:
                result[name.strip()] = int(val.strip())
            except ValueError:
                continue
        return result


settings = Settings()
