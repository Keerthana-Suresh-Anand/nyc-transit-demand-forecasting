"""MLflow registry helpers: production promotion and resolution via version aliases.

Aliases replace the deprecated stage API. Versions promoted before the alias
migration carry only the legacy "Production" stage, so resolution falls back to
it; the alias takes over at each model's next promotion.
"""
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from src.utils.config import PRODUCTION_ALIAS


def resolve_production_version(client: MlflowClient, model_name: str) -> ModelVersion | None:
    """The model version currently serving production, or None if never promoted.

    Tries the production alias first, then the legacy stage (highest such version).
    """
    try:
        return client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except MlflowException:
        pass
    legacy = [v for v in client.search_model_versions(f"name='{model_name}'")
              if v.current_stage == "Production"]
    return max(legacy, key=lambda v: int(v.version)) if legacy else None


def promote_to_production(client: MlflowClient, model_name: str, version: int | str) -> None:
    """Point the production alias at ``version``. Reassignment is atomic — the
    previously aliased version simply loses the alias (no archiving step)."""
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, str(version))


def production_model_uri(client: MlflowClient, model_name: str) -> str:
    """``models:/`` URI pinned to the resolved production version number, so loading
    works for alias-promoted and legacy stage-promoted versions alike."""
    mv = resolve_production_version(client, model_name)
    if mv is None:
        raise MlflowException(f"No production version found for '{model_name}'")
    return f"models:/{model_name}/{mv.version}"
