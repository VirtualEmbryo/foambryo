"""The ambiguous top-level `benchmarks` name must not be importable.

Both this repository and `delaunay-watershed-3d` used to ship a top-level package
literally named `benchmarks`. A script that put both repository roots on `sys.path`
(e.g. a research script that resolves the sibling repository by path rather than by
its installed package) could then have `import benchmarks` /
`from benchmarks.metrics import ...` silently resolve to whichever repository came
first on `sys.path` -- see `benchmarks/dw3d_variant_comparison.py`'s (now
`foambryo_benchmarks/dw3d_variant_comparison.py`) `load_dw3d_benchmark_metrics`
docstring for this collision described first-hand, and `BENCHMARKS.md`'s "package
renamed" section for the fix.

This repository's package is now `foambryo_benchmarks`. This test fails (loudly, at
collection or otherwise) if the ambiguous name `benchmarks` can ever supply the
dangerous attribute again -- `benchmarks.metrics` -- whether from a real reintroduced
package, or from a stray sibling checkout on `sys.path`.

A later check found that `import benchmarks` is not always the right check: this
repository's own working tree, right after the package rename, still had a gitignored
`benchmarks/__pycache__/` left on disk from before the `git mv`, with `benchmarks/`'s
`__init__.py` gone. Python happily imports that as an implicit namespace package (PEP
420) -- `import benchmarks` succeeds, `benchmarks.__file__` is `None` -- which made the
original assert-it-raises version of this test fail on a real, freshly-migrated
checkout it was specifically meant to protect, while the actual hazard (a second
`benchmarks.metrics` shadowing this repository's) was already gone. See
`CHANGELOG.md`'s note on that rename for the checkout-hygiene fix (delete the stray
directory) this test no longer needs in order to pass.
"""

from __future__ import annotations

import pytest


def test_bare_benchmarks_package_cannot_supply_metrics():
    """`benchmarks.metrics` must not be importable, in either form `benchmarks` can take.

    Either `import benchmarks` raises outright (the clean-checkout case), or it resolves
    to an implicit namespace package -- no `__file__`, no source files, so it cannot
    supply a `.metrics` submodule (the stale-`__pycache__` case, see module docstring).
    A *real* colliding package -- one with a `__file__` -- is the actual hazard this test
    exists to catch, and fails it.
    """
    try:
        import benchmarks
    except ModuleNotFoundError:
        return

    real_file = getattr(benchmarks, "__file__", None)
    assert real_file is None, (
        f"A real 'benchmarks' package exists at {real_file!r} -- "
        "the ambiguous top-level name has been reintroduced."
    )
    with pytest.raises(ModuleNotFoundError):
        import benchmarks.metrics
