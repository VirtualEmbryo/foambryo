"""Registry of `dw3d` mesh-reconstruction variants, resolved from git rather than hardcoded.

Built for the attribution benchmark (meshing error versus inference error). `dw3d`'s default
configuration has changed repeatedly during development (`dithered` -> `deterministic` ->
`offset_included` (a former shipped default) -> `offset_excluded` -> `offset_excluded_linkcheck`,
which **later made the new shipped default** via `get_link_checked_mesh_reconstruction_algorithm`),
and any recorded snapshot of `delaunay-watershed-3d`'s state goes stale as soon as that
independently developed repository moves. So this module never hardcodes
"the current dw3d variant" — it discovers whatever `get_*_mesh_reconstruction_algorithm`
factories the imported `dw3d` package actually exposes, and the caller picks names from that
discovery rather than from a list frozen at the time this file was written.

**The variant-registry hazard, found and fixed by the robustness work.** A curated entry keyed
to `get_default_mesh_reconstruction_algorithm` silently changes *meaning* every time `dw3d`
flips its default — which is exactly what happened when the default moved from
`offset_included` to `offset_excluded_linkcheck`
**after** `foambryo_benchmarks/baseline/attribution.json` was committed with a row named
`"default"` that meant `offset_included`. Reading that row today, unchanged, would silently mean
something else. The fix is structural, not a one-off relabelling: every curated variant here
is pinned to an **explicit** getter name (`get_junction_protected_...`, `get_link_checked_...`, ...),
never to `get_default_...`, and `resolve_variant` refuses (`DW3DImportError`) to resolve *any*
variant, curated or not, through a getter whose name contains `"default"` — so this class of
defect cannot recur silently even if a future edit reintroduces a `"default"` entry by
accident. The pre-fix `"default"`-labelled rows in the committed baseline were re-labelled
`"a3_a5"` (verified bit-for-bit against a live re-run) since
that is what they always were; no row in the grid resolves through the moving getter.

**How `dw3d` is resolved.** `foambryo`'s own environment (`.venv`) has no declared dependency
on `dw3d` (checked: absent from `pyproject.toml` and `uv.lock`) and its installed copy can be a
stale snapshot that predates the reconstruction-pipeline renames (no `get_dithered_*`,
no `get_deterministic_*`, no `get_link_checked_*`). The attribution benchmark therefore
requires `dw3d` to be importable from the sibling checkout, e.g.:

    PYTHONPATH="../delaunay-watershed-3d/src" uv run python -m foambryo_benchmarks.run_attribution ...

`import_dw3d` below fails loudly (not a stale silent import) if the installed copy lacks the
variants this module needs, so a forgotten `PYTHONPATH` cannot silently benchmark the wrong
`dw3d`.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

#: Variants curated for the default attribution run, and why each is in the grid (not
#: exhaustive - see `discover_variants` for everything `dw3d` exposes). Kept as *names*,
#: resolved to callables lazily by `import_dw3d`, so importing this module never requires
#: `dw3d` itself.
DEFAULT_VARIANT_NAMES: tuple[str, ...] = (
    # The published configuration (dw3d 0.3.6) - the comparability anchor.
    "dithered",
    # Deterministic tie-breaking, an earlier default.
    "deterministic",
    # Boundary layer + junction protection (a former shipped default), pinned to its OWN explicit
    # getter (get_junction_protected_mesh_reconstruction_algorithm) rather than to
    # get_default_mesh_reconstruction_algorithm. Measured to carry a +15.7% signed median
    # interface-area error (47 ground-truth cases, min_distance = 3). This entry used to be named
    # "default" and resolve through the default getter; a later default flip moved the default out
    # from under it, which is the hazard this module's guard now forbids by construction (see module
    # docstring).
    "offset_included",
    # The link-condition-checked offset exclusion. Clears every offset-exclusion
    # acceptance criterion and **is** the shipped default - reachable here by its own
    # explicit name (get_link_checked_...), never through get_default_... Included because
    # it is the direct test of the prediction that a smaller interface-area bias improves
    # tension inference. Measured on the 42 equilibrium cases: ForceBalanceKKT total error
    # 0.344 here (+1.07% area bias) against 0.715 on offset_included (+15.7%).
    "offset_excluded_linkcheck",
)

#: Map from the short name used in this module (and in every attribution record) to the
#: exact `dw3d` export name, so the JSON output does not depend on dw3d's naming convention
#: leaking through. Deliberately has NO entry mapping to `get_default_mesh_reconstruction_
#: algorithm` - every curated variant is pinned to an explicit, non-moving getter (see
#: module docstring, "the variant-registry hazard"). `resolve_variant` enforces this even
#: for uncurated names. The `v0_3` / `a1b` / `a3_a5` entries are legacy short names, kept so
#: rows committed under them (and any saved command using them) keep resolving.
_EXPORT_NAME_BY_SHORT_NAME = {
    "dithered": "get_dithered_mesh_reconstruction_algorithm",
    "deterministic": "get_deterministic_mesh_reconstruction_algorithm",
    "offset_included": "get_junction_protected_mesh_reconstruction_algorithm",
    "boundary_layer": "get_boundary_layer_mesh_reconstruction_algorithm",
    "junction_protected": "get_junction_protected_mesh_reconstruction_algorithm",
    "cubic_score": "get_junction_protected_mesh_reconstruction_algorithm",
    "offset_excluded": "get_junction_protected_mesh_reconstruction_algorithm",
    "offset_excluded_linkcheck": "get_link_checked_mesh_reconstruction_algorithm",
}

#: Keyword arguments a short name adds to its getter. dw3d 0.5.0 deprecates `offset_included`,
#: `cubic_score` and `offset_excluded` as exact synonyms of these keyword calls on the
#: junction-protected getter (removed in 0.6.0), so they are spelled that way here; the mesh is
#: bit-identical either way (dw3d's own tests pin it).
_KEYWORDS_BY_SHORT_NAME: dict[str, dict] = {
    "cubic_score": {"spline_order": 3},
    "offset_excluded": {"exclude_offsets_from_surface": True},
}

#: Inverse of `_EXPORT_NAME_BY_SHORT_NAME`, so `discover_variants` reports the *curated*
#: short name (e.g. `"offset_excluded_linkcheck"`) rather than dw3d's own, more cryptic
#: export-derived one (`"link_checked"`) wherever this module has an opinion on naming.
#: Built before the legacy aliases below are added, and only from entries that add no keyword
#: and do not share their getter with another curated name, so a canonical short name always
#: wins this direction.
_SHORT_NAME_BY_EXPORT_NAME = {
    "get_dithered_mesh_reconstruction_algorithm": "dithered",
    "get_deterministic_mesh_reconstruction_algorithm": "deterministic",
    "get_boundary_layer_mesh_reconstruction_algorithm": "boundary_layer",
    "get_junction_protected_mesh_reconstruction_algorithm": "junction_protected",
    "get_link_checked_mesh_reconstruction_algorithm": "offset_excluded_linkcheck",
}

#: Legacy short names, still accepted (forward direction only) so old committed baseline
#: rows and saved commands using them keep resolving to the same getter as their canonical
#: replacement.
_EXPORT_NAME_BY_SHORT_NAME.update(
    {
        "v0_3": "get_dithered_mesh_reconstruction_algorithm",
        "a1b": "get_deterministic_mesh_reconstruction_algorithm",
        "a3_a5": "get_junction_protected_mesh_reconstruction_algorithm",
    },
)

_FACTORY_NAME_PATTERN = re.compile(r"^get_(.+)_mesh_reconstruction_algorithm$")


class DW3DImportError(RuntimeError):
    """Raised when the imported `dw3d` package is missing a variant this module needs."""


def import_dw3d() -> object:
    """Import `dw3d`, failing loudly rather than silently using a stale install.

    Returns:
        object: The imported `dw3d` module.

    Raises:
        DW3DImportError: If `dw3d` cannot be imported, or is importable but predates the
            reconstruction-pipeline renames (e.g. the stale copy that ships with a bare
            `foambryo` install) - recognised by the absence of
            `get_dithered_mesh_reconstruction_algorithm`, which every current `dw3d`
            checkout exposes.
    """
    try:
        dw3d = importlib.import_module("dw3d")
    except ImportError as exc:
        message = (
            "dw3d is not importable. The attribution benchmark needs the current delaunay-watershed-3d "
            "checkout on PYTHONPATH, e.g. "
            'PYTHONPATH="../delaunay-watershed-3d/src" uv run python -m foambryo_benchmarks.run_attribution'
        )
        raise DW3DImportError(message) from exc
    if not hasattr(dw3d, "get_dithered_mesh_reconstruction_algorithm"):
        message = (
            f"imported dw3d from {getattr(dw3d, '__file__', '?')}, which has no "
            "get_dithered_mesh_reconstruction_algorithm - this is an older snapshot "
            "(the one bundled with a bare foambryo install), not the current "
            "delaunay-watershed-3d checkout. Put the sibling repo's src/ on PYTHONPATH."
        )
        raise DW3DImportError(message)
    return dw3d


def discover_variants(dw3d: object) -> dict[str, str]:
    """List every `get_*_mesh_reconstruction_algorithm` factory the imported `dw3d` exposes.

    Discovered from the module's own attributes rather than a list frozen in this file, so
    a variant added to `dw3d` after this module was written is still reachable by name
    without editing `foambryo`.

    Args:
        dw3d (object): The imported `dw3d` module (see `import_dw3d`).

    Returns:
        dict[str, str]: `{short_name: export_name}`, `short_name` being the export name
            with the `get_` prefix and `_mesh_reconstruction_algorithm` suffix stripped
            (e.g. `"dithered"` for `get_dithered_mesh_reconstruction_algorithm`).
    """
    found = {}
    for export_name in dir(dw3d):
        match = _FACTORY_NAME_PATTERN.match(export_name)
        if match and callable(getattr(dw3d, export_name)):
            short_name = _SHORT_NAME_BY_EXPORT_NAME.get(export_name, match.group(1))
            found[short_name] = export_name
    return found


def assert_not_a_default_getter(short_name: str, export_name: str) -> None:
    """Fail loudly if `export_name` is a getter whose name contains `"default"`.

    The variant-registry hazard (module docstring): `dw3d.get_default_mesh_reconstruction_
    algorithm` has already changed meaning once (`offset_included` -> `offset_excluded_
    linkcheck`) without any public signature changing, so a registry entry pinned to it
    silently changes what a committed baseline row means the moment a newer `dw3d` is
    imported. This is a structural guard, not a one-off check for today's registry: it
    rejects *any* short name (curated or not, e.g. a caller passing `--variants default`
    directly) that would resolve through such a getter, so the defect this guard fixed
    cannot reappear by a future edit re-adding a `"default"` entry to
    `_EXPORT_NAME_BY_SHORT_NAME` or by a caller bypassing the registry.

    Args:
        short_name (str): The variant name requested (for the error message).
        export_name (str): The `dw3d` export name it would resolve to.

    Raises:
        DW3DImportError: If `"default"` appears in `export_name`.
    """
    if "default" in export_name:
        message = (
            f"variant {short_name!r} would resolve through {export_name!r}, whose name "
            "contains 'default'. dw3d's default has already changed meaning once "
            "(`offset_included` -> `offset_excluded_linkcheck`) with no public signature "
            "change, so any registry entry that resolves through a 'default' getter "
            "silently changes meaning whenever dw3d's default changes next. Pin an "
            "explicit, non-moving getter name instead (see "
            "foambryo_benchmarks/dw3d_variants.py's module docstring)."
        )
        raise DW3DImportError(message)


def resolve_variant(dw3d: object, short_name: str, *, min_distance: int = 3) -> Callable:
    """Resolve a variant's short name to a ready-to-call `MeshReconstructionAlgorithm`.

    Args:
        dw3d (object): The imported `dw3d` module (see `import_dw3d`).
        short_name (str): A key of `discover_variants(dw3d)`, e.g. `"dithered"`.
        min_distance (int, optional): Passed to the factory. Defaults to 3, the spacing
            the recorded attribution runs used.

    Returns:
        Callable: The constructed `MeshReconstructionAlgorithm`, i.e. the return value of
            `dw3d.get_<short_name>_mesh_reconstruction_algorithm(min_distance)`.

    Raises:
        DW3DImportError: If `short_name` is not exposed by this `dw3d` (e.g. asking for
            `"offset_excluded_linkcheck"` on a `dw3d` checkout that predates it), or if it
            would resolve through a getter whose name contains `"default"` (see
            `assert_not_a_default_getter` and the module docstring's "variant-registry
            hazard").
    """
    available = discover_variants(dw3d)
    export_name = _EXPORT_NAME_BY_SHORT_NAME.get(short_name, f"get_{short_name}_mesh_reconstruction_algorithm")
    assert_not_a_default_getter(short_name, export_name)
    # Checked by export name, not by short-name membership in `available`: a legacy short
    # name (e.g. `"a3_a5"`) resolves to a canonical export (`get_junction_protected_...`) that
    # `discover_variants` reports under its own canonical short name, not the legacy one.
    if export_name not in available.values():
        message = (
            f"dw3d variant {short_name!r} (expected export {export_name!r}) is not exposed "
            f"by the imported dw3d (available: {sorted(available)}). Either this dw3d "
            "checkout predates that variant, or the short name is wrong."
        )
        raise DW3DImportError(message)
    factory = getattr(dw3d, export_name)
    keywords = _KEYWORDS_BY_SHORT_NAME.get(short_name, {})
    return factory(min_distance, print_info=False, **keywords)


def reconstruct(
    algorithm: object,
    mask: NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Run one `MeshReconstructionAlgorithm` on a segmentation mask.

    Thin wrapper kept for a single, greppable call site and to pin the return-value
    contract this module relies on (`(points, triangles, labels)`,
    labels are the two materials each triangle separates, `0` = exterior).

    Args:
        algorithm (object): A `MeshReconstructionAlgorithm`, e.g. from `resolve_variant`.
        mask (NDArray[np.integer]): Instance segmentation label image.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
            `(points, triangles, labels)`.
    """
    points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    return points, triangles, labels
