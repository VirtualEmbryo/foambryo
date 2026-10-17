"""Tests for `foambryo_benchmarks.dw3d_variants` (the robustness work's fix for the variant-registry hazard).

`dw3d.get_default_mesh_reconstruction_algorithm` has already changed meaning once
(`offset_included` -> `offset_excluded_linkcheck`) with no public signature change, which
silently changed the meaning of any committed row keyed to a curated `"default"` entry.
These tests pin the structural fix: no curated variant resolves through a
`"default"`-named getter, and the guard rejects one even if it did not run through the
discovery machinery.
"""

import pytest

from foambryo_benchmarks import dw3d_variants as dv


class _FakeDW3D:
    """Minimal stand-in exposing a few `get_*_mesh_reconstruction_algorithm` factories.

    Lets these tests exercise `resolve_variant`'s guard and `discover_variants` without
    requiring the sibling `delaunay-watershed-3d` checkout on `PYTHONPATH`.
    """

    def get_dithered_mesh_reconstruction_algorithm(self, min_distance, **kwargs):  # noqa: ANN003, ARG002
        return ("dithered", min_distance)

    def get_junction_protected_mesh_reconstruction_algorithm(self, min_distance, **kwargs):  # noqa: ANN003
        return ("junction_protected", min_distance, kwargs)

    def get_default_mesh_reconstruction_algorithm(self, min_distance, **kwargs):  # noqa: ANN003, ARG002
        return ("default", min_distance)


def test_no_curated_variant_resolves_through_a_default_named_getter():
    """The structural invariant this guard adds: scan every curated entry, not just one."""
    for short_name in dv.DEFAULT_VARIANT_NAMES:
        export_name = dv._EXPORT_NAME_BY_SHORT_NAME.get(
            short_name,
            f"get_{short_name}_mesh_reconstruction_algorithm",
        )
        assert "default" not in export_name, f"{short_name!r} resolves through {export_name!r}"


def test_assert_not_a_default_getter_raises_on_a_default_named_export():
    with pytest.raises(dv.DW3DImportError, match="contains 'default'"):
        dv.assert_not_a_default_getter("default", "get_default_mesh_reconstruction_algorithm")


def test_assert_not_a_default_getter_passes_on_an_explicit_export():
    dv.assert_not_a_default_getter("offset_included", "get_junction_protected_mesh_reconstruction_algorithm")


def test_resolve_variant_refuses_the_default_getter_even_though_it_is_discoverable():
    """`"default"` is discoverable, but `resolve_variant` must refuse it anyway.

    Discovery alone (`dw3d.discover_variants`) would have let this hazard back in; the guard
    that matters is the one at resolution time.
    """
    fake_dw3d = _FakeDW3D()
    assert "default" in dv.discover_variants(fake_dw3d)  # discoverable...
    with pytest.raises(dv.DW3DImportError, match="contains 'default'"):
        dv.resolve_variant(fake_dw3d, "default")  # ...but refused at resolution time


def test_resolve_variant_offset_included_resolves_through_the_junction_protected_getter():
    """`get_offset_included_...`, an exact synonym of this getter, was removed from dw3d in 0.7.0."""
    fake_dw3d = _FakeDW3D()
    result = dv.resolve_variant(fake_dw3d, "offset_included", min_distance=3)
    assert result == ("junction_protected", 3, {"print_info": False})


@pytest.mark.parametrize(
    ("short_name", "keywords"),
    [("cubic_score", {"spline_order": 3}), ("offset_excluded", {"exclude_offsets_from_surface": True})],
)
def test_the_deprecated_synonyms_resolve_to_their_keyword_calls(short_name, keywords):
    fake_dw3d = _FakeDW3D()
    result = dv.resolve_variant(fake_dw3d, short_name, min_distance=3)
    assert result == ("junction_protected", 3, {"print_info": False, **keywords})


def test_resolve_variant_accepts_the_legacy_short_name():
    """The legacy short name `a3_a5` must still resolve to the same getter as `offset_included`."""
    fake_dw3d = _FakeDW3D()
    result = dv.resolve_variant(fake_dw3d, "a3_a5", min_distance=3)
    assert result == ("junction_protected", 3, {"print_info": False})
