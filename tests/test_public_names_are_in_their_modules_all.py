"""Every public name a front end uses is listed in its module's `__all__`.

napari-foambryo and the command line use names of these modules (found by reading napari-foambryo's syntax trees);
`foambryo.io` had an `__all__` without `default_min_distance`, and the other modules had none. Each of them now lists
every public name it defines, so that its `__all__` is complete: a module's own public functions, classes and constants,
not the names it imports (`numpy`, ...).
"""

from __future__ import annotations

import ast
import importlib
import inspect

import pytest

#: The names used outside their module, by module.
USED_OUTSIDE = {
    "foambryo.io": ["default_min_distance"],
    "foambryo.pressure_inference": ["default_pressure_method"],
    "foambryo.force_inference": ["InferenceOutcome"],
    "foambryo.second_order_angle_reading": ["ANGLE_READINGS", "CHORD_READING"],
    "foambryo.seed_ensembles": ["COMBINE_OPTIONS"],
    "foambryo.dcel": ["DcelData", "separate_faces_dict_keep_idx"],
}

#: The modules that had no `__all__`: theirs is every public name they define.
COMPLETE = [name for name in USED_OUTSIDE if name != "foambryo.io"]


def _public_names_defined_in(module: object) -> set[str]:
    names = set()
    for node in ast.parse(inspect.getsource(module)).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")} - {"__all__"}


@pytest.mark.parametrize(("module_name", "names"), sorted(USED_OUTSIDE.items()))
def test_the_names_used_outside_are_in_the_modules_all(module_name, names):
    module = importlib.import_module(module_name)
    for name in names:
        assert name in module.__all__, f"{module_name}.{name}"
    for name in module.__all__:
        assert hasattr(module, name), f"{module_name}.__all__ names {name}, which it does not have"
    assert len(set(module.__all__)) == len(module.__all__)


@pytest.mark.parametrize("module_name", COMPLETE)
def test_an_all_lists_every_public_name_the_module_defines(module_name):
    module = importlib.import_module(module_name)
    assert set(module.__all__) == _public_names_defined_in(module)
    namespace: dict = {}
    exec(f"from {module_name} import *", namespace)  # noqa: S102 - the star import is what __all__ decides
    assert set(namespace) - {"__builtins__"} == set(module.__all__)
