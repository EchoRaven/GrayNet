from types import SimpleNamespace

import pytest

from graynet.agents.registry import DriverConfig, available_drivers, create_driver


def test_plugin_driver_is_lazy_and_core_execution_needs_no_vendor_import(monkeypatch):
    loaded = []
    driver = SimpleNamespace(
        preflight=lambda: None, diagnose=lambda incident: None, trace=[], metadata={}
    )
    entry = SimpleNamespace(
        name="second-driver", load=lambda: loaded.append(True) or (lambda context, config: driver)
    )
    monkeypatch.setattr("graynet.agents.registry.entry_points", lambda **kwargs: [entry])
    assert available_drivers() == ("codex", "second-driver") and loaded == []
    assert create_driver("second-driver", None, DriverConfig("model", "low")) is driver
    assert loaded == [True]


def test_plugin_cannot_override_a_registered_driver(monkeypatch):
    monkeypatch.setattr(
        "graynet.agents.registry.entry_points", lambda **kwargs: [SimpleNamespace(name="codex")]
    )
    with pytest.raises(ValueError, match="duplicate"):
        available_drivers()


def test_invalid_driver_contract_fails_before_network_deployment(monkeypatch):
    monkeypatch.setattr(
        "graynet.agents.registry.entry_points",
        lambda **kwargs: [SimpleNamespace(name="bad", load=lambda: lambda *args: object())],
    )
    with pytest.raises(ValueError, match="implement"):
        create_driver("bad", None, DriverConfig(None, None))


@pytest.mark.parametrize("value", [0, -1, True, 1.1])
def test_driver_budgets_are_positive_integers(value):
    with pytest.raises(ValueError):
        DriverConfig(None, None, value)
