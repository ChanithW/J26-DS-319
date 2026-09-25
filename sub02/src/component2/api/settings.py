"""Service configuration from environment variables (see .env.example)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings; names map to upper-case environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    component2_bundle_path: str = "model_bundle_v3.joblib"
    database_url: str = "sqlite:///./runs.db"
    cors_origins: str = "http://localhost:5173"
    max_flares_per_request: int = 500
    max_points_per_flare: int = 5000
    enable_lightkurve_fetch: bool = False
    sample_data_dir: str = "data/tess_raw"

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
