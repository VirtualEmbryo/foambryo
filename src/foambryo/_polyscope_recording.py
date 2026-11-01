"""The viewer's scene prepared off the window's thread: Polyscope's calls recorded, then replayed on the window's.

Polyscope draws on the thread that runs its window, and every Polyscope call must be made there. The viewer's
drawing code mixes the two kinds of work: computing what to draw (the mesh's graph, its colours, curvatures, stress
tensors, the exploded copy; most of the time on a large tissue) and handing the arrays to Polyscope (fast). So that
the window keeps drawing frames while a scene is computed, the drawing code runs on a worker thread against a
`Recorder`, which takes the place of the `polyscope` module there and records every call with its arguments (the
final arrays); `Recorder.replay` then makes the same calls, in the same order, on the window's thread.

`polyscope_target` is the `ps` the drawing modules use: on a thread that has a recorder (`recording`), it is the
recorder; everywhere else, it is Polyscope itself, looked up at each call (so tests that replace a Polyscope
function still replace it). Replaying a recording makes exactly the calls a direct drawing makes, with the same
arrays (tested): nothing drawn changes, only the thread the arrays are computed on.

Either way, `polyscope_target` hands Polyscope (or the recorder) the positions, triangles and vectors in the drawn
frame, (x, y, z) = the mesh's columns (2, 1, 0) with each triangle's orientation kept (`foambryo._display_frame`):
the drawing code works in the mesh's own coordinates, and the window shows the tissue, not its mirror image.
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Any

import polyscope as _polyscope

from foambryo._display_frame import in_drawn_frame

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Recorder", "polyscope_target", "recording"]

_LOCAL = threading.local()

#: Calls that act on the window rather than on the scene: not recorded (the window's thread makes them itself).
_NOT_RECORDED = frozenset({"init", "show"})

#: The structure kinds `register_<kind>`, `has_<kind>` and `get_<kind>` name.
_KINDS = ("surface_mesh", "curve_network", "point_cloud")


class _PolyscopeTarget:
    """`ps` in the drawing modules: this thread's recorder while one is set, else the `polyscope` module.

    Its positions, triangles and vectors go through the drawn frame (`foambryo._display_frame.in_drawn_frame`).
    """

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - whatever Polyscope or the recorder holds
        recorder = getattr(_LOCAL, "recorder", None)
        attribute = getattr(_polyscope, name) if recorder is None else recorder.attribute(name)
        return in_drawn_frame(name, attribute)


polyscope_target = _PolyscopeTarget()


class RecordedStructure:
    """A structure registered while recording: the method calls made on it are recorded in order."""

    def __init__(self, recorder: Recorder, index: int, kind: str, name: str) -> None:
        self._recorder, self._index, self.kind, self.name = recorder, index, kind, name

    def get_name(self) -> str:
        """The structure's name, as Polyscope's `get_name`."""
        return self.name

    def __getattr__(self, method: str) -> Any:  # noqa: ANN401 - a recording function
        if method.startswith("_"):
            raise AttributeError(method)

        def record(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            self._recorder.calls.append((self._index, method, args, kwargs))

        return record


class Recorder:
    """Polyscope's calls made by drawing code, in order, with their arguments, to be replayed (`replay`).

    Attributes:
        calls: `(target, name, args, kwargs)`: `target` is `None` for a module function, else the index of the
            call that registered the structure the method is called on.
        structures: the structures registered, by `(kind, name)`.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int | None, str, tuple, dict]] = []
        self.structures: dict[tuple[str, str], RecordedStructure] = {}

    def attribute(self, name: str) -> Any:  # noqa: ANN401 - a recording function, or a Polyscope class
        """What `ps.<name>` is while recording."""
        if name in _NOT_RECORDED:
            return lambda *_args, **_kwargs: None
        if name[:1].isupper():  # a class or an enum (type hints, isinstance): Polyscope's own
            return getattr(_polyscope, name)
        for kind in _KINDS:
            if name == f"register_{kind}":
                return lambda structure_name, *args, _kind=kind, **kwargs: self._register(
                    _kind, structure_name, args, kwargs)
            if name == f"has_{kind}":
                return lambda structure_name, _kind=kind: (_kind, structure_name) in self.structures
            if name == f"get_{kind}":
                return lambda structure_name, _kind=kind: self.structures[(_kind, structure_name)]
        if name == "remove_all_structures":
            self.structures.clear()

        def record(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            self.calls.append((None, name, args, kwargs))

        return record

    def _register(self, kind: str, name: str, args: tuple, kwargs: dict) -> RecordedStructure:
        index = len(self.calls)
        self.calls.append((None, f"register_{kind}", (name, *args), kwargs))
        structure = RecordedStructure(self, index, kind, name)
        self.structures[(kind, name)] = structure
        return structure

    def recorded_calls(self, structure_name: str, method: str) -> list[tuple[tuple, dict]]:
        """The arguments of every call of `method` on the structures named `structure_name`, in order."""
        registered = {index for index, (_, name, args, _) in enumerate(self.calls)
                      if name.startswith("register_") and args[0] == structure_name}
        return [(args, kwargs) for target, name, args, kwargs in self.calls
                if target in registered and name == method]

    def replay(self) -> None:
        """Make the recorded calls on Polyscope, in order. Call it on the window's thread.

        Polyscope is initialised first (the recorder drops the drawing code's `init` calls; `init` does nothing once
        Polyscope runs), so that a first scene can be replayed in a fresh process.
        """
        _polyscope.init()
        structures: dict[int, Any] = {}
        for index, (target, name, args, kwargs) in enumerate(self.calls):
            if target is None:
                result = getattr(_polyscope, name)(*args, **kwargs)
                if name.startswith("register_"):
                    structures[index] = result
            else:
                getattr(structures[target], name)(*args, **kwargs)


@contextlib.contextmanager
def recording(recorder: Recorder) -> Iterator[Recorder]:
    """Within this block, on this thread, `polyscope_target` records into `recorder`."""
    previous = getattr(_LOCAL, "recorder", None)
    _LOCAL.recorder = recorder
    try:
        yield recorder
    finally:
        _LOCAL.recorder = previous
