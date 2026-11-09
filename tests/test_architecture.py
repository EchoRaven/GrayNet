import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "graynet"


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


def test_root_is_a_small_composition_boundary_not_a_flat_framework():
    assert {path.name for path in PACKAGE.glob("*.py")} == {
        "__init__.py",
        "__main__.py",
        "cli.py",
        "paths.py",
    }


def test_execution_is_independent_of_concrete_agent_drivers():
    for path in (PACKAGE / "execution").glob("*.py"):
        assert "graynet.agents.codex" not in set(imports(path))


def test_tools_do_not_import_the_containerlab_implementation_or_private_faults():
    for path in (PACKAGE / "tools").glob("*.py"):
        found = set(imports(path))
        assert "graynet.environments.containerlab" not in found
        assert "graynet.environments.faults" not in found
        assert "graynet.evaluation.judge" not in found


def test_session_policy_does_not_depend_on_agents_or_evaluation():
    for path in (PACKAGE / "runtime").glob("*.py"):
        assert not any(
            module.startswith(("graynet.agents", "graynet.evaluation", "graynet.execution"))
            for module in imports(path)
        )
