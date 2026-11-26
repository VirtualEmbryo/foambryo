"""foambryo's package-level functions take 2-D meshes and masks: the dispatch, `PlanarMesh`, and the refusals.

Pinned: (1) `dimension_of` tells a 3-D mesh from every accepted 2-D one, and refuses anything else by name; (2) the
package-level functions given a 2-D mesh return bitwise what the private 2-D implementation returns (the arc reading,
the per-component solution, the Laplace pressures, the global fit), whatever the 2-D form (`PlanarMesh`, dw2d's
`DCEL_Data`, a triple, a pair); (3) an option of one dimension given a non-default value in the other is refused;
functions with no 2-D counterpart refuse a 2-D mesh; (4) the mask, array and file functions return a `PlanarMesh` for
2-D input, with the image border and the pixel size; (5) `foambryo.two_dimensional` (never released) is gone, its
2-D-only names are at the top level; (6) the global fit refuses a label image that is not the mesh's.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import foambryo
from foambryo import PressureComputationMethod, TensionComputationMethod, _two_dimensional


def three_cells() -> np.ndarray:
    """A 60 x 60 label image: three cells meeting at two junctions, in a medium."""
    labels = np.zeros((60, 60), dtype=np.int64)
    labels[10:50, 10:30] = 1
    labels[10:30, 30:50] = 2
    labels[30:50, 30:50] = 3
    return labels


@pytest.fixture(scope="module")
def planar() -> foambryo.PlanarMesh:
    return foambryo.dcel_mesh_from_segmentation_mask(three_cells(), min_distance=3)


def _same_map(a: dict, b: dict) -> bool:
    return list(a) == list(b) and np.array_equal(np.array(list(a.values())), np.array(list(b.values())), equal_nan=True)


def test_the_mask_function_returns_a_planar_mesh_with_the_image_border(planar):
    import dw2d

    algorithm = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    points, segments, labels = algorithm.construct_mesh_from_segmentation_mask(three_cells())
    assert isinstance(planar, foambryo.PlanarMesh)
    assert np.array_equal(planar.points, points) and np.array_equal(planar.e, np.hstack((segments, labels)))
    assert np.array_equal(planar.image_border_vertices, algorithm.image_border_vertices)
    assert planar.image_shape == (60, 60) and planar.cells == (1, 2, 3) and planar.cut_cells == ()


def test_the_2d_default_spacing_is_four_pixels():
    default = foambryo.dcel_mesh_from_segmentation_mask(three_cells())
    explicit = foambryo.dcel_mesh_from_segmentation_mask(three_cells(), min_distance=4)
    assert np.array_equal(default.points, explicit.points)
    assert foambryo.io.default_min_distance(2) == 4 and foambryo.io.default_min_distance(3) == 3


def test_dimension_of_every_accepted_form(planar):
    dcel = foambryo.dcel_mesh_from_multimaterial_mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float), np.array([[0, 1, 2], [0, 3, 1]]),
        np.array([[0, 1], [0, 1]]))
    assert foambryo.dimension_of(dcel) == 3
    for form in (planar, (planar.points, planar.segments, planar.labels), (planar.points, planar.e)):
        assert foambryo.dimension_of(form) == 2
    with pytest.raises(TypeError, match="PlanarMesh"):
        foambryo.dimension_of("a mesh")


@pytest.mark.parametrize("form", ["planar", "triple", "pair"])
def test_the_package_level_inference_is_the_2d_implementation_bitwise(planar, form):
    mesh = {"planar": planar, "triple": (planar.points, planar.segments, planar.labels),
            "pair": (planar.points, planar.e)}[form]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _same_map(foambryo.infer_tensions(mesh), _two_dimensional.infer_tensions(planar))
        assert _same_map(foambryo.infer_tensions(mesh, angle_reading="chord"),
                         _two_dimensional.infer_tensions(planar, angle_reading="chord"))
        solution = foambryo.infer_tensions_by_junction_component(mesh)
        expected = _two_dimensional.infer_tensions_by_junction_component(planar)
        assert _same_map(solution.dict_tensions, expected.dict_tensions)
        assert _same_map(foambryo.infer_pressures(mesh, solution), _two_dimensional.infer_pressures(planar, expected))
        tensions, pressures = foambryo.infer_forces(mesh)
        assert _same_map(tensions, expected.dict_tensions)
        assert _same_map(pressures, _two_dimensional.infer_pressures(planar, expected))


def test_infer_forces_and_the_front_end_call_agree_in_2d(planar):
    tensions, pressures, diagnostics = foambryo.infer_forces(planar, base_pressure=0.5, diagnostics=True)
    assert isinstance(diagnostics, foambryo.SolveDiagnostics) and len(diagnostics.records) == 2
    outcome = foambryo.infer_tensions_then_pressures(planar, base_pressure=0.5)
    assert outcome.status == "complete"
    assert _same_map(outcome.tensions, tensions) and _same_map(outcome.pressures, pressures)
    assert pressures[0] == 0.5
    tensions_only = foambryo.infer_tensions_then_pressures(planar, with_pressures=False)
    assert tensions_only.pressures is None and _same_map(tensions_only.tensions, tensions)


def test_the_dw2d_dcel_is_read_as_the_planar_mesh(planar):
    import dw2d

    algorithm = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    algorithm.construct_mesh_from_segmentation_mask(three_cells())
    dcel = algorithm.last_constructed_dcel
    assert foambryo.dimension_of(dcel) == 2
    assert _same_map(foambryo.infer_tensions(dcel), foambryo.infer_tensions(planar))


def test_options_of_one_dimension_are_refused_in_the_other(planar):
    with pytest.raises(ValueError, match="refusal applies to 3-D meshes only"):
        foambryo.infer_tensions(planar, refusal=foambryo.WedgeRefusal(foambryo.WedgeRefusalPolicy.DropLine))
    with pytest.raises(ValueError, match="not offered in 2D"):
        foambryo.infer_pressures(planar, foambryo.infer_tensions_by_junction_component(planar),
                                 PressureComputationMethod.Variational)
    with pytest.raises(ValueError, match="not offered in 2D"):
        foambryo.infer_tensions(planar, mode=TensionComputationMethod.Lami)
    mesh3 = _a_3d_mesh()
    with pytest.raises(ValueError, match="2-D meshes only"):
        foambryo.infer_tensions(mesh3, exclude_image_border=False)
    with pytest.raises(ValueError, match="2-D meshes only"):
        foambryo.infer_pressures(mesh3, {}, excluded_vertices=[0])


@pytest.mark.parametrize("function", ["infer_force_balance", "infer_tensions_young_dupre_local",
                                      "infer_tensions_with_uncertainty", "compute_inference_diagnostics",
                                      "infer_tensions_and_residuals", "infer_pressures_and_residuals"])
def test_functions_without_a_2d_counterpart_refuse_a_2d_mesh(planar, function):
    arguments = (planar, {}) if function == "infer_pressures_and_residuals" else (planar,)
    with pytest.raises(ValueError, match="no 2-D counterpart"):
        getattr(foambryo, function)(*arguments)


def test_the_seed_ensemble_refuses_a_2d_mask():
    with pytest.raises(ValueError, match="no 2-D counterpart"):
        foambryo.infer_tensions_over_seeds(three_cells(), [0, 1])


def test_arrays_and_files_give_a_planar_mesh(planar, tmp_path):
    from_arrays = foambryo.dcel_mesh_from_multimaterial_mesh(planar.points, planar.segments, planar.labels)
    assert isinstance(from_arrays, foambryo.PlanarMesh)
    with pytest.raises(ValueError, match="3-D meshes only"):
        foambryo.dcel_mesh_from_multimaterial_mesh(planar.points, planar.segments, planar.labels,
                                                   source_spacing=(1.0, 1.0, 1.0))
    reconstruction = foambryo.reconstruct_mask(three_cells(), (0.5, 0.5), min_distance=3)
    assert reconstruction.dimension == 2 and reconstruction.summary().startswith(f"{len(planar.points)} points")
    reconstruction.write_rec(tmp_path / "mesh.rec")
    read = foambryo.dcel_mesh_from_file(tmp_path / "mesh.rec")
    assert isinstance(read, foambryo.PlanarMesh)
    assert np.array_equal(read.points, planar.points * 0.5) and read.spacing == 0.5 and read.unit == "um"
    assert np.array_equal(read.image_border_vertices, planar.image_border_vertices)
    assert read.image_shape == (60, 60)
    # Laplace pressures per unit length: half the pixel size, twice the pressure.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pixels = foambryo.infer_forces(planar)
        _, physical = foambryo.infer_forces(read)
    assert np.allclose([physical[k] for k in pixels], [2 * pixels[k] for k in pixels], rtol=1e-9)


def test_the_2d_module_is_private_and_its_names_are_at_the_top_level():
    with pytest.raises(ImportError):
        from foambryo import two_dimensional  # noqa: F401
    for name in ("PlanarMesh", "infer_forces_by_equilibrium_tiling", "EquilibriumTilingReport",
                 "EQUILIBRIUM_TILING_MAX_EVALUATIONS", "interface_arcs", "InterfaceArc", "junction_table",
                 "image_border_vertices", "TwoDimensionalReport", "TwoDimensionalPressureReport",
                 "MAX_LENS_CONDITION", "IllConditionedLensWarning", "NoJunctionError", "ANGLE_READINGS_2D",
                 "DEFAULT_ANGLE_READING_2D"):
        assert name in foambryo.__all__ and getattr(foambryo, name) is not None
    assert foambryo.ANGLE_READINGS_2D == ("arc", "chord", "fitted_tangent")
    assert foambryo.DEFAULT_ANGLE_READING_2D == "arc"
    assert not hasattr(_two_dimensional, "infer_by_arc_tiling")


def test_the_global_fit_takes_the_mesh_first_and_checks_the_label_image(planar):
    labels = three_cells()
    solution, pressures, report = foambryo.infer_forces_by_equilibrium_tiling(planar, labels, diagnostics=True)
    assert isinstance(report, foambryo.EquilibriumTilingReport)
    assert set(solution.dict_tensions) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert pressures[0] == 0.0
    with pytest.raises(ValueError, match="shape"):
        foambryo.infer_forces_by_equilibrium_tiling(planar, labels[:, :50])
    with pytest.raises(ValueError, match="absent from labels"):
        foambryo.infer_forces_by_equilibrium_tiling(planar, np.where(labels == 3, 0, labels))
    with pytest.raises(ValueError, match="2-D label image"):
        foambryo.infer_forces_by_equilibrium_tiling(planar, labels[None])


def test_planar_mesh_checks_its_arrays():
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        foambryo.PlanarMesh(np.zeros((3, 3)), [[0, 1]], [[0, 1]])
    with pytest.raises(ValueError, match="outside"):
        foambryo.PlanarMesh(np.zeros((3, 2)), [[0, 5]], [[0, 1]])
    with pytest.raises(ValueError, match="smaller first"):
        foambryo.PlanarMesh(np.zeros((3, 2)), [[0, 1]], [[2, 1]])


def _a_3d_mesh() -> foambryo.DcelData:
    return foambryo.dcel_mesh_from_multimaterial_mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
        np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]), np.array([[0, 1]] * 4))
