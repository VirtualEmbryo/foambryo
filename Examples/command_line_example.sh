#!/bin/sh
# The foambryo command, from a mask to the figures, on a synthetic mask whose answer is known.
#
#     pip install "foambryo[viewing]"
#     sh command_line_example.sh
#
# Everything is written into ./foambryo-command-example/. The mask is a doublet at equilibrium: two
# equal spheres of radius R = 30 voxels whose centres are R apart, split by the plane between them,
# so that the three surfaces meet at 120 degrees. All three tensions are then equal (1, with the
# default mean tension of 1) and each cell's pressure is 2 / R = 0.133 per um on voxels of 0.5 um
# (R = 15 um). The inferred values come within about 10 % of these: the gap is the cost of
# meshing a voxel mask at this size.
#
# The steps:
#   1. foambryo mesh             the mask's mesh, in um (a .npz that records its voxel size; then a
#                                stack with voxels twice as long along z, whose size is given with --spacing)
#   2. foambryo infer            tensions.csv, pressures.csv, run.json, and solves.csv with --diagnostics
#   3. foambryo infer --no-pressures   the tensions only
#   4. foambryo view             the colour bars without a window (--colour-bars, --no-show), and the
#                                view itself as a figure (--figure; needs Polyscope and a display)
#   5. foambryo infer-over-seeds the tensions over eight reconstructions, with their spread
set -e
mkdir -p foambryo-command-example
cd foambryo-command-example

python - <<'EOF'
import numpy as np

n, radius = 102, 30
z, y, x = np.indices((n, n, n))
c = (n - 1) / 2
left = (z - c) ** 2 + (y - c) ** 2 + (x - (c - radius / 2)) ** 2 <= radius**2
right = (z - c) ** 2 + (y - c) ** 2 + (x - (c + radius / 2)) ** 2 <= radius**2
mask = np.where(left | right, np.where(x < c, 1, 2), 0).astype(np.uint16)
np.savez("doublet.npz", mask=mask, spacing=np.array([0.5, 0.5, 0.5]), spacing_unit=np.array("um"))
np.save("doublet_every_second_z_slice.npy", mask[::2])  # no voxel size in the file
EOF

# 1. The .npz records its spacing, so --spacing is not needed. The .npy records none: --spacing is
#    required (there is no default), and a stack with longer voxels along z is resampled to cubic
#    voxels before meshing (a message says so).
foambryo mesh doublet.npz -o doublet.rec --overwrite
foambryo mesh doublet_every_second_z_slice.npy --spacing 1.0 0.5 0.5 --spacing-unit um \
    -o doublet_from_the_stack.rec --overwrite

# 2. Tensions and pressures, with the defaults of the library (bit for bit those of infer_forces),
#    and one row per linear solve in solves.csv.
foambryo infer doublet.rec -o results --diagnostics basic --overwrite

# 3. The tensions only (no pressures.csv; run.json says the pressures were not computed).
foambryo infer doublet_from_the_stack.rec -o results-tensions-only --no-pressures --overwrite

# 4. The colour bars need neither a display nor Polyscope. The figure (the view with its colour bars
#    and length scale bar) needs both; without a display, this step says so and the script goes on.
foambryo view doublet.rec --results results --colour-bars results/colour_bars.png --no-show
foambryo view doublet.rec --results results --figure results/figure.png --no-show \
    || echo "no figure: 'foambryo view --figure' needs Polyscope and a display"

# 5. The same mask reconstructed at the dither seeds 0 to 7, and the tensions averaged over them:
#    tensions.csv gains the spread over the seeds, and tensions_per_seed.csv holds each seed's own
#    values. The spread is a lower bound on the error, not an error bar.
foambryo infer-over-seeds doublet.npz -o results-over-seeds --seeds 8 --overwrite

cat results/tensions.csv
cat results/pressures.csv
cat results/solves.csv
cat results-tensions-only/tensions.csv
cat results-over-seeds/tensions.csv

# To look at the tissue in the window (needs a display):
#     foambryo view doublet.rec --results results
# or start from the mask, and reconstruct and infer in the window:
#     foambryo view doublet.npz
