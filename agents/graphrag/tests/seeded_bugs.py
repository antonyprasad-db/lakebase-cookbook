"""Seeded bugs: does test_differential.py notice when the compiler is wrong?

Each bug changes one compiler method, in memory and in a fresh process (the source files are never
edited), and the randomized differential tests are run against it. A bug the tests do not catch
exits non-zero.

    cd agents/graphrag
    export PGRAPH_DSN="postgresql://..."   # same throwaway database as the tests
    uv run --with pytest python tests/seeded_bugs.py

Prints, for each bug, how many of the randomized tests (one per random graph) give a wrong answer.
Only a failed assertion counts as catching a bug. A bug run with wrong answers is caught even if
some tests also errored (the errors are reported). A run with skips, with a number of tests other
than CASES, or whose failures are all errors is not a measurement and stops the script; bugs
already found to escape are still reported. Every BUGS entry is checked against the compiler
before anything runs, and the clean compiler runs first and must pass every test with no errors.

Exit status: 0 when every bug is caught, 1 when a bug gives no wrong answer, 2 when nothing could
be measured or trusted: a stale BUGS entry, no PGRAPH_DSN, a run with skips or a number of tests
other than CASES, a bug run whose failures are all errors, a clean compiler with any wrong answer
or error, a pytest run that broke, or any unexpected exception. The reason is printed.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 900  # per run of the 300 tests; a bug that makes a walk explode should not hang this
# The randomized test the counts come from; other tests in the file are not run.
TEST = "test_compiled_sql_matches_reference"
# Random graphs per run, pinned so a PGRAPH_CASES set for quick test runs cannot shrink the sample.
CASES = "300"

# name -> (compiler method, exact text, replacement). Each text must occur once in its method;
# a text of None replaces the method with a no-op instead.
BUGS = {
    "no relationship uniqueness": ("uniqueness", None, None),
    "nulls sort first ascending": (
        "sort_keys",
        "WHEN 'number' THEN 4 ELSE 5 END",
        "WHEN 'number' THEN 4 ELSE -1 END",
    ),
    "an edge may repeat inside a variable length path": (
        "rel",
        ") CYCLE key_src, key_dst, rel SET reused",
        ") CYCLE hops SET reused",
    ),
    "undirected treated as outgoing": (
        "rel",
        'if r.direction == "both"',
        'if r.direction == "never"',
    ),
}


class Unusable(Exception):
    """A run that measured nothing: no result, a skip, an error, or too few tests."""


def stop(reason: str) -> NoReturn:
    """Nothing was measured, which is not the same as a bug escaping: say why and exit 2."""
    print(reason, file=sys.stderr, flush=True)
    sys.exit(2)


def check(bug: str):
    """The method a BUGS entry patches, its source and first line, or stop if the entry is stale."""
    from pgcypher import compiler

    if bug not in BUGS:
        stop(f"unknown bug {bug!r}; the bugs are {list(BUGS)}")
    method, old, new = BUGS[bug]
    if not hasattr(compiler._Compiler, method):
        stop(f"{bug}: _Compiler has no method {method}; update BUGS to match the compiler")
    lines, first = inspect.getsourcelines(getattr(compiler._Compiler, method))
    source = textwrap.dedent("".join(lines))
    if old is not None:
        if source.count(old) != 1:
            stop(
                f"{bug}: expected the text once in _Compiler.{method}, found {source.count(old)};"
                " update BUGS to match the compiler"
            )
        if new == old:
            stop(f"{bug}: the replacement is the same as the text, so it seeds nothing")
        try:
            compile(source.replace(old, new), compiler.__file__, "exec")
        except SyntaxError as exc:
            stop(f"{bug}: the patched _Compiler.{method} does not compile: {exc}")
    return compiler, method, source, first


def apply(bug: str) -> None:
    """Rebind one _Compiler method to a no-op, or to a copy of its source with the text replaced."""
    compiler, method, source, first = check(bug)
    _, old, new = BUGS[bug]
    if old is None:
        setattr(compiler._Compiler, method, lambda *args, **kwargs: None)
        return
    # Padded so a traceback from the patched method points at its own lines in compiler.py.
    # getsourcelines includes decorators, so a @staticmethod comes back as one.
    code = compile("\n" * (first - 1) + source.replace(old, new), compiler.__file__, "exec")
    scope: dict = {}
    exec(code, vars(compiler), scope)
    setattr(compiler._Compiler, method, scope[method])


def child(bug: str | None) -> None:
    import pytest

    sys.path.insert(0, str(ROOT))
    if bug:
        apply(bug)
    outcome: dict[str, str] = {}  # one per test, whichever of setup, call or teardown decides it
    first_error: list[str] = []

    class Count:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item, call):
            report = (yield).get_result()
            seen = outcome.get(report.nodeid)
            if report.failed:
                # the exception type, not the message: a bare assert has no "AssertionError" text
                wrong = (
                    report.when == "call"
                    and call.excinfo is not None
                    and call.excinfo.errisinstance(AssertionError)
                )
                if seen != "wrong":
                    outcome[report.nodeid] = "wrong" if wrong else "errors"
                if not wrong and not first_error:
                    first_error.append(report.longreprtext[-2500:])
            elif report.skipped:
                outcome.setdefault(report.nodeid, "skipped")
            elif report.when == "call":
                outcome.setdefault(report.nodeid, "passed")

    # --tb=short so an error carries its file, line and query, not just the exception message
    code = pytest.main(
        [f"{ROOT / 'tests/test_differential.py'}::{TEST}", "-q", "-p", "no:cacheprovider"]
        + ["--no-header"]
        + ["--tb=short", "-rN"],
        plugins=[Count()],
    )
    if code not in (0, 1):  # 0 all passed, 1 some failed; anything else is a broken run
        stop(f"pytest exited {code}, so nothing was measured")
    counts = {
        k: sum(v == k for v in outcome.values()) for k in ("wrong", "errors", "skipped", "passed")
    }
    print(json.dumps({**counts, "first_error": first_error[0] if first_error else None}))


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2] if len(sys.argv) > 2 else None)
        return 0

    sys.path.insert(0, str(ROOT))
    for name in BUGS:  # a stale entry stops the script now, not after four full runs
        check(name)
    # PGRAPH_CASES pinned; PYTEST_ADDOPTS dropped, since -x, -k or -n would shrink the sample
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    env["PGRAPH_CASES"] = CASES

    def run(bug: str | None) -> dict:
        label = bug or "clean compiler"
        print(f"running {label} ...", file=sys.stderr, flush=True)
        try:
            out = subprocess.run(
                [sys.executable, __file__, "--child", *([bug] if bug else [])],
                capture_output=True,
                text=True,
                cwd=ROOT,
                env=env,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise Unusable(
                f"{label}: no result within {TIMEOUT_S} s. A query may still be running on the"
                " server and the graph schema may still hold test rows: cancel the query"
                " (pg_cancel_backend) and DROP SCHEMA graph CASCADE in the test database before"
                " running again."
            ) from None
        lines = out.stdout.strip().splitlines()
        if out.returncode or not lines:
            raise Unusable(f"{label} run failed:\n{out.stdout[-2000:]}{out.stderr[-2000:]}")
        r = json.loads(lines[-1])
        total = r["wrong"] + r["errors"] + r["skipped"] + r["passed"]
        extra = "".join(f", plus {r[k]} {k}" for k in ("errors", "skipped") if r[k])
        print(
            f"{label}: {r['wrong']} of {total} randomized tests give a wrong answer{extra}",
            flush=True,
        )
        if r["skipped"]:
            raise Unusable(
                f"{label}: the randomized tests skipped; set PGRAPH_DSN as for the tests"
            )
        if total != int(CASES):
            raise Unusable(
                f"{label}: ran {total} randomized tests, not {CASES}, so the counts do not compare"
            )
        if r["errors"] and (bug is None or not r["wrong"]):
            raise Unusable(
                f"{label}: {r['errors']} tests errored, so this run is not a clean measurement."
                f" First error:\n{r['first_error']}"
            )
        return r

    try:
        clean = run(None)
    except Unusable as exc:
        stop(str(exc))
    if clean["wrong"]:
        stop("the clean compiler gives wrong answers; fix that before seeding bugs")
    escaped: list[str] = []
    unusable = None
    for name in BUGS:
        try:
            if run(name)["wrong"] == 0:
                escaped.append(name)
        except Unusable as exc:
            unusable = str(exc)
            break
    if escaped:
        print(f"not caught: {', '.join(escaped)}", file=sys.stderr, flush=True)
    if unusable:
        print(f"stopped before the remaining bugs: {unusable}", file=sys.stderr, flush=True)
        if not escaped:
            sys.exit(2)
    return 1 if escaped else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # not SystemExit: exit 1 must mean only that a bug escaped
        stop(f"unexpected error, so nothing was measured: {exc!r}")
