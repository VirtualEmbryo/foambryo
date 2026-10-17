"""Viewer defaults that the `foambryo` command needs without the `viewing` extra.

Kept apart from `foambryo._viewer_figures`, which imports matplotlib, so that building the command's parser (which
shows these defaults in `foambryo view --help`) works after a plain `pip install foambryo`.
"""

from __future__ import annotations

#: The scattered (exploded) view's coefficient when none is given: each cell moved by 0.2 x (its centroid - the
#: tissue's centroid), `DcelData.compute_scattered_arrays(coeff)`. The viewer functions' default, and `foambryo view
#: --scattered`'s.
DEFAULT_SCATTER_COEFF = 0.2
