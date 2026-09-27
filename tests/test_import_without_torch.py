"""The sparse-assembly work — `import foambryo` and inference must not require `torch`.

torch became an optional `[autodiff]` extra in the sparse-assembly work (it backs only the gradient-check test).
We prove the runtime path is torch-free by running a subprocess in which any `import torch`
raises, then importing foambryo and running the full default inference (YoungDupre tensions
+ Variational pressures, plus the Variational tension solve). The subprocess also asserts
that `torch` never entered `sys.modules`.

This is a subprocess rather than an in-process monkeypatch because `torch` is already
imported by other tests in the same session; only a fresh interpreter can prove a clean
torch-free import.
"""

import subprocess
import sys
import textwrap

from tests.conftest import require_substitute_mesh

_SCRIPT = textwrap.dedent(
    """
    import sys
    from importlib.abc import MetaPathFinder

    class _BlockTorch(MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError("torch is blocked for this test")
            return None

    sys.meta_path.insert(0, _BlockTorch())

    import foambryo
    from foambryo import (
        TensionComputationMethod,
        dcel_mesh_from_file,
        infer_forces,
    )
    from foambryo.tension_inference import infer_tensions_and_residuals

    mesh = dcel_mesh_from_file(sys.argv[1])
    # Full published default: YoungDupre tensions + Variational pressures.
    tensions, pressures = infer_forces(mesh)
    assert len(tensions) > 0 and len(pressures) > 0
    # The variational tension solve (the one that used the derivatives) must also run.
    _, var_tensions, _ = infer_tensions_and_residuals(mesh, 1, TensionComputationMethod.Variational)
    assert len(var_tensions) > 0

    assert "torch" not in sys.modules, "torch was imported on the runtime path"
    print("OK-NO-TORCH")
    """,
)


def test_import_and_inference_without_torch():
    mesh_path = require_substitute_mesh()
    result = subprocess.run(  # noqa: S603 — fixed script + trusted paths, a test subprocess
        [sys.executable, "-c", _SCRIPT, str(mesh_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "OK-NO-TORCH" in result.stdout, result.stdout
