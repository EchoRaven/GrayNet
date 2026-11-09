from pathlib import Path

import pytest

from graynet.scenarios.definitions import CaseFormatError, load_case

FIXTURE = Path(__file__).parent / "fixtures" / "sustained_congestion"


def test_load_case() -> None:
    case = load_case(FIXTURE)
    assert case.case_id == "sustained-congestion-01"
    assert case.ground_truth.component == "r1-r2"
    assert case.validation.threshold == 0.20
    assert "slower than expected" in case.incident
    assert case.incident_path == FIXTURE.resolve() / "incident.md"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("schema_version: 1", "schema_version: 2", "schema_version"),
        ("schema_version: 1", "schema_version: true", "schema_version"),
        ("id: sustained-congestion-01", "id: ../../outside", "lowercase slug"),
        ("anomaly: true", "anomaly: false", "healthy control"),
        ("effect: degraded", "effect: stable", "must agree"),
        ("threshold: 0.20", "threshold: 0", "positive"),
        ("min_samples: 5", "min_samples: 2", "at least 3"),
        ("topology: topology.clab.yml", "topology: missing.yml", "does not exist"),
        ("topology: topology.clab.yml", "topology: ../../outside.yml", "inside the case directory"),
    ],
)
def test_reject_inconsistent_case(tmp_path: Path, old: str, new: str, message: str) -> None:
    source = (FIXTURE / "scenario.yaml").read_text(encoding="utf-8")
    (tmp_path / "scenario.yaml").write_text(source.replace(old, new), encoding="utf-8")
    (tmp_path / "incident.md").write_text(
        (FIXTURE / "incident.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "topology.clab.yml").touch()
    with pytest.raises(CaseFormatError, match=message):
        load_case(tmp_path)


def test_reject_unknown_fields(tmp_path: Path) -> None:
    source = (FIXTURE / "scenario.yaml").read_text(encoding="utf-8")
    (tmp_path / "scenario.yaml").write_text(source + "unknown: value\n", encoding="utf-8")
    (tmp_path / "incident.md").write_text("Investigate.", encoding="utf-8")
    (tmp_path / "topology.clab.yml").touch()
    with pytest.raises(CaseFormatError, match="unknown scenario field"):
        load_case(tmp_path)


def standalone_case(directory: Path, extra: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "scenario.yaml").write_text((FIXTURE / "scenario.yaml").read_text() + extra)
    (directory / "topology.clab.yml").touch()
    return directory


@pytest.mark.parametrize("profile", ["two_path", "partial_path"])
def test_each_portfolio_uses_shared_public_inputs(profile):
    root = Path(__file__).parents[1]
    environment = root / "environments" / profile
    expected = environment / "incident.md"
    cases = [load_case(path) for path in sorted((root / "cases").iterdir()) if path.is_dir()]
    cases = [case for case in cases if case.network == profile]
    assert len(cases) == {"two_path": 36, "partial_path": 9}[profile]
    assert {case.incident_path for case in cases} == {expected.resolve()}
    assert {case.incident for case in cases} == {expected.read_text().strip()}
    assert {case.environment_path for case in cases} == {environment / "environment.md"}
    assert len({case.environment for case in cases}) == 1
    assert all(case.ground_truth.diagnosis for case in cases)


def test_explicit_case_report_override(tmp_path):
    directory = standalone_case(tmp_path, "incident: report.md\n")
    (directory / "report.md").write_text("Some transfers are slow.")
    case = load_case(directory)
    assert case.incident == "Some transfers are slow."
    assert case.incident_path == directory / "report.md"


def test_network_case_requires_explicit_local_override(tmp_path):
    directory = standalone_case(tmp_path, "network: two_path\n")
    (directory / "incident.md").write_text("Different report.")
    with pytest.raises(CaseFormatError, match="explicit incident"):
        load_case(directory)
    manifest = directory / "scenario.yaml"
    manifest.write_text(manifest.read_text() + "incident: incident.md\n")
    assert load_case(directory).incident == "Different report."


@pytest.mark.parametrize("report", ["missing.md", "../outside.md", "/absolute.md"])
def test_incident_reference_rejects_missing_absolute_or_escaped_files(tmp_path, report):
    directory = standalone_case(tmp_path / "case", f"incident: {report}\n")
    (tmp_path / "outside.md").write_text("Not an allowed source.")
    with pytest.raises(CaseFormatError, match="incident"):
        load_case(directory)


def test_incident_symlink_cannot_escape_case_directory(tmp_path):
    directory = standalone_case(tmp_path / "case")
    outside = tmp_path / "secret.md"
    outside.write_text("Private.")
    (directory / "incident.md").symlink_to(outside)
    with pytest.raises(CaseFormatError, match="incident must be inside"):
        load_case(directory)


@pytest.mark.parametrize("content", [None, "", "  \n"])
def test_standalone_incident_must_exist_and_be_nonempty(tmp_path, content):
    directory = standalone_case(tmp_path)
    if content is not None:
        (directory / "incident.md").write_text(content)
    with pytest.raises(CaseFormatError, match="incident report"):
        load_case(directory)


def test_benign_fault_is_a_healthy_case_with_an_active_fault() -> None:
    case = load_case(Path(__file__).parents[1] / "cases/decoy_backup_link_loss")
    assert case.benign_fault and not case.ground_truth.anomaly
    assert case.fault["kind"] == "link_loss" and case.validation.effect == "stable"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("benign_fault: true", "benign_fault: 1", "must be a boolean"),
        ("benign_fault: true\n", "", "healthy control"),
    ],
)
def test_reject_inconsistent_benign_fault(tmp_path: Path, old: str, new: str, message: str) -> None:
    root = Path(__file__).parents[1]
    case = root / "cases" / "decoy_backup_link_loss"
    text = (case / "scenario.yaml").read_text().replace(old, new)
    copy = root / "cases" / f".test-{tmp_path.name}"
    copy.mkdir()
    try:
        (copy / "scenario.yaml").write_text(text)
        with pytest.raises(CaseFormatError, match=message):
            load_case(copy)
    finally:
        (copy / "scenario.yaml").unlink()
        copy.rmdir()
