"""Repository conventions enforced on the test suite (see AGENTS.md design checklist)."""

import ast
from pathlib import Path

TESTS = Path(__file__).parent
ROOT = TESTS.parent
SKILL_COPIES = (
    ROOT / "skill" / "SKILL.md",
    ROOT / "src" / "runtime" / "skills" / "oracle" / "SKILL.md",
    ROOT / "src" / "oracle_rag" / "skills" / "oracle" / "SKILL.md",
)
PATCH_METHODS = {"setattr", "setitem", "patch"}

# Known private patches that predate the rule. Remove entries as they are replaced
# by injected fakes; do not add new ones.
ALLOWED_PRIVATE_PATCHES = {
    ("test_service.py", "service.service._stop_state"),
    ("test_service.py", "service.service._spawn_detached"),
    ("test_service.py", "service.service._windows_process_alive"),
    ("test_service.py", "service.service._windows_spawn_options"),
    ("test_supervisor.py", "service.service._escalate_stop"),
    ("test_supervisor.py", "service.service._wait_for_exit"),
    ("test_chunks.py", "documents.chunks._chunk_end"),
}


def _is_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def private_name_uses(path: Path) -> list[tuple[str, int]]:
    """Return (target, line) for private imports and string patch targets in a test file."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            found += [
                (f"{node.module}.{alias.name}", node.lineno)
                for alias in node.names
                if _is_private(alias.name)
            ]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            first = node.args[0] if node.args else None
            if (
                node.func.attr in PATCH_METHODS
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and _is_private(first.value.rsplit(".", 1)[-1])
            ):
                found.append((first.value, node.lineno))
    return found


def test_tests_do_not_use_private_names_from_src():
    violations = [
        f"{path.name}:{line} uses {target}"
        for path in sorted(TESTS.glob("test_*.py"))
        if path.name != Path(__file__).name
        for target, line in private_name_uses(path)
        if (path.name, target) not in ALLOWED_PRIVATE_PATCHES
        and not target.startswith("test_")  # helpers shared between test files
    ]
    assert not violations, (
        "Inject a fake port instead of importing or patching private names:\n"
        + "\n".join(violations)
    )


def test_detector_flags_private_import_and_string_patch(tmp_path):
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "from pkg.mod import _hidden, public\n"
        "def test_x(monkeypatch):\n"
        "    monkeypatch.setattr('pkg.mod._other', None)\n"
        "    monkeypatch.setattr('pkg.mod.public', None)\n"
    )
    assert private_name_uses(sample) == [("pkg.mod._hidden", 1), ("pkg.mod._other", 3)]


def test_bundled_skill_copies_stay_identical():
    contents = {path.relative_to(ROOT): path.read_bytes() for path in SKILL_COPIES}
    canonical = contents[Path("skill/SKILL.md")]
    mismatched = [str(path) for path, data in contents.items() if data != canonical]
    assert not mismatched, f"Sync skill/SKILL.md to bundled copies: {', '.join(mismatched)}"

