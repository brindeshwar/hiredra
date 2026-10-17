"""CI safety guards (plan unit G1, decisions F8/D29): static checks that keep hard rule 1 true by construction.

    G-a  Only the fill driver may import Playwright inside hiredra/autofill/ and hiredra/gate/.
    G-b  Nothing in hiredra/autofill/, hiredra/gate/ or hiredra/outreach/ sends POST/PUT through an HTTP client.
    G-c  Nothing in hiredra/ imports a mail-sending or messaging client.

The checks parse the code (ast), so comments and strings never trip them. A guarded directory that
doesn't exist yet passes trivially; the guard starts biting the day that module appears.
Run:  uv run python scripts/ci_guards.py   (exit code 1 = at least one violation)
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PLAYWRIGHT_DIRS = ("hiredra/autofill", "hiredra/gate")
PLAYWRIGHT_ALLOWED = ("hiredra/autofill/driver.py",)
NO_WRITE_HTTP_DIRS = ("hiredra/autofill", "hiredra/gate", "hiredra/outreach")
WRITE_METHODS = {"post", "put"}
MESSAGING_MODULES = {
    "smtplib",
    "aiosmtplib",
    "email.mime",
    "yagmail",
    "sendgrid",
    "mailgun",
    "mailjet_rest",
    "postmarker",
    "twilio",
    "slack_sdk",
    "slack",
    "telegram",
    "discord",
    "pywhatkit",
    "whatsapp",
}


@dataclass(frozen=True)
class Violation:
    guard: str
    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.guard}: {self.path}:{self.line}: {self.detail}"


def _py_files(root: Path, rel_dir: str):
    base = root / rel_dir
    if not base.is_dir():
        return
    yield from sorted(base.rglob("*.py"))


def _imported_modules(tree: ast.AST):
    """(module_name, line) for every import, plus __import__/importlib.import_module with a constant name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, node.lineno
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name in ("__import__", "import_module") and isinstance(node.args[0].value, str):
                yield node.args[0].value, node.lineno


def _matches(module: str, banned: str) -> bool:
    return module == banned or module.startswith(banned + ".")


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def guard_a(root: Path) -> list[Violation]:
    out = []
    for rel_dir in PLAYWRIGHT_DIRS:
        for path in _py_files(root, rel_dir):
            rel = path.relative_to(root).as_posix()
            if rel in PLAYWRIGHT_ALLOWED:
                continue
            for module, line in _imported_modules(_parse(path)):
                if _matches(module, "playwright"):
                    out.append(Violation("G-a", rel, line, f"imports '{module}'; only {PLAYWRIGHT_ALLOWED[0]} may"))
    return out


def _is_write_method(value: ast.AST) -> bool:
    return (
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value.strip().lower() in WRITE_METHODS
    )


def guard_b(root: Path) -> list[Violation]:
    out = []
    for rel_dir in NO_WRITE_HTTP_DIRS:
        for path in _py_files(root, rel_dir):
            rel = path.relative_to(root).as_posix()
            for node in ast.walk(_parse(path)):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr in WRITE_METHODS:
                    out.append(Violation("G-b", rel, node.lineno, f"calls .{f.attr}(...)"))
                for kw in node.keywords:
                    if kw.arg == "method" and _is_write_method(kw.value):
                        out.append(Violation("G-b", rel, node.lineno, f"method={kw.value.value!r}"))
                name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
                if name in ("request", "Request", "stream") and node.args and _is_write_method(node.args[0]):
                    out.append(Violation("G-b", rel, node.lineno, f"{name}({node.args[0].value!r}, ...)"))
                if name in ("Request", "urlopen") and any(kw.arg == "data" for kw in node.keywords):
                    out.append(Violation("G-b", rel, node.lineno, f"urllib {name}(..., data=...) sends a body"))
    return out


def guard_c(root: Path) -> list[Violation]:
    out = []
    for path in _py_files(root, "hiredra"):
        rel = path.relative_to(root).as_posix()
        for module, line in _imported_modules(_parse(path)):
            for banned in MESSAGING_MODULES:
                if _matches(module, banned):
                    out.append(Violation("G-c", rel, line, f"imports '{module}' (mail/messaging client)"))
    return out


def run(root: Path = REPO) -> list[Violation]:
    return guard_a(root) + guard_b(root) + guard_c(root)


def main() -> int:
    violations = run()
    for v in violations:
        print(v)
    print(f"ci_guards: {len(violations)} violation(s) (G-a playwright imports, G-b POST/PUT, G-c mail/messaging)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
