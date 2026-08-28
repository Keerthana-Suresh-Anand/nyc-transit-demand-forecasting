"""Tests for registry helpers: alias-first resolution with legacy-stage fallback."""
from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException

from src.utils.config import PRODUCTION_ALIAS
from src.utils.registry import (
    production_model_uri,
    promote_to_production,
    resolve_production_version,
)


def _version(version: str, stage: str = "None") -> MagicMock:
    v = MagicMock()
    v.version = version
    v.current_stage = stage
    return v


def _client(alias_version=None, search_versions=()):
    client = MagicMock()
    if alias_version is None:
        client.get_model_version_by_alias.side_effect = MlflowException("alias not found")
    else:
        client.get_model_version_by_alias.return_value = alias_version
    client.search_model_versions.return_value = list(search_versions)
    return client


class TestResolveProductionVersion:
    def test_alias_wins_when_set(self):
        aliased = _version("3")
        client = _client(alias_version=aliased, search_versions=[_version("4", "Production")])
        assert resolve_production_version(client, "m") is aliased
        client.search_model_versions.assert_not_called()

    def test_falls_back_to_legacy_stage(self):
        legacy = _version("2", "Production")
        client = _client(search_versions=[_version("1", "Archived"), legacy, _version("3")])
        assert resolve_production_version(client, "m") is legacy

    def test_legacy_fallback_picks_highest_version(self):
        v1, v5 = _version("1", "Production"), _version("5", "Production")
        client = _client(search_versions=[v1, v5])
        assert resolve_production_version(client, "m") is v5

    def test_none_when_never_promoted(self):
        client = _client(search_versions=[_version("1"), _version("2")])
        assert resolve_production_version(client, "m") is None


class TestPromoteToProduction:
    def test_sets_alias_with_string_version(self):
        client = MagicMock()
        promote_to_production(client, "m", 7)
        client.set_registered_model_alias.assert_called_once_with("m", PRODUCTION_ALIAS, "7")


class TestProductionModelUri:
    def test_uri_pins_resolved_version_number(self):
        client = _client(alias_version=_version("4"))
        assert production_model_uri(client, "m") == "models:/m/4"

    def test_raises_when_no_production_version(self):
        client = _client(search_versions=[])
        with pytest.raises(MlflowException, match="No production version"):
            production_model_uri(client, "m")
