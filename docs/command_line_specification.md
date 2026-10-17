# The `foambryo` command — behavioural specification, version 1

Version 1, for foambryo 0.7.0 (October 2026; § 11 lists the changes since the first version). The implementation follows this text; where the two disagree, the text is the intent and the code is a defect. § 9 lists possible extensions.

This specification describes **behaviour only**: commands, options, defaults, files, columns, exit codes and messages. Every number the command writes is a number the library returns for the same options: the command adds file handling and records, and **no default of its own**.

## 1. Installation and invocation

```text
pip install foambryo                  the command, .npy / .npz masks
pip install "foambryo[cli]"           adds tifffile, for .tif / .tiff masks
pip install "foambryo[viewing]"       adds Polyscope and matplotlib, for `foambryo view`
```

* One executable, `foambryo`, with four sub-commands: `mesh`, `infer`, `infer-over-seeds`, `view`. There is no `foambryo-mesh`, `foambryo-infer`, … : names with a hyphen are not installed, so they cannot collide with other tools of that name on the `PATH`.
* `python -m foambryo.cli …` is the same program, for a machine where the scripts directory is not on the `PATH`.
* `foambryo --version` prints `foambryo <version> (delaunay-watershed-3d <version>)` and exits 0.
* `foambryo --help` and `foambryo <sub-command> --help` print the help text and exit 0. The help text is part of this interface.
* `foambryo` with no sub-command prints the usage and exits 2.

## 2. The pipeline

```text
mask (.npy/.npz/.tif) --foambryo mesh--> mesh (.rec, spacing recorded) --foambryo infer--> OUTDIR/
                                                                                            tensions.csv
                                                                                            pressures.csv
                                                                                            run.json
                                                                                            solves.csv (opt.)
                                                                                            spectra.npz (opt.)
mesh + OUTDIR --foambryo view--> Polyscope window, and/or a colour-bar image, and/or a figure
mask --foambryo infer-over-seeds--> OUTDIR/ tensions.csv (with the seed spread), tensions_per_seed.csv, run.json
```

Each step can also start from files made otherwise: `infer` reads any mesh file foambryo's library reads, and `view` any mesh with or without results.

## 3. `foambryo mesh`

```text
foambryo mesh MASK -o MESH.rec [--spacing DZ DY DX] [--spacing-unit UNIT]
                               [--min-distance N] [--overwrite]
```

**What it does.** Reads a segmentation mask and reconstructs its multimaterial mesh **in physical units** with delaunay-watershed-3d's default reconstruction, through that library's entry point for masks with a voxel size (the same call as `dw3d.get_default_mesh_reconstruction_algorithm(min_distance).construct_mesh_in_physical_units(mask, spacing, spacing_unit=UNIT, axis_order="zyx")`), then writes the `.rec` file that reconstruction exports itself, whose metadata appendix records the spacing. The same route is `foambryo.dcel_mesh_from_segmentation_mask(mask, spacing=...)` in Python.

* **Cubic voxels** (the three spacings equal, to a relative 1e-6): the mask is meshed as it is and the vertices are multiplied by the spacing.
* **Voxels longer along one axis** (a z step larger than the pixel size): the mask is first **resampled to cubic voxels** of the shortest spacing by shape-based interpolation (each label's signed distance interpolated between slices), then meshed, and the vertices are multiplied by the shortest spacing. The reconstruction then handles `k` times the voxels, `k` being the longest spacing over the shortest, and `--min-distance` counts cubic voxels.
* **Two different short axes** (three different spacings, or two long and one short) are refused.

**Why resample.** Meshing the native voxels and scaling the mesh afterwards samples the surfaces coarsely along the long axis: the junction triangles come out about `k` times taller and tilted, and every contact angle read from them is off. Measured on 12 simulated tissues of 90 and 165 cells voxelised at a real cell size and down-sampled along z by `k`, one meshing seed, as the median over tissues of each mesh's median line-mean contact-angle error minus that of the same tissue meshed at full resolution: resampling first costs +0.83, +0.71 and +1.17 degrees at `k` = 5/3, 2 and 3; meshing the native voxels costs +1.38, +1.97 and +4.10 degrees (`BENCHMARKS.md`). The tensions follow. **Nothing was measured above `k` = 3** at that cell size, so the command warns there (§ 3.4) and builds the mesh anyway. The default angle reading stays: the quadric reading was worse on every route at that cell size.

### 3.1 The mask

| Suffix | Read with | Requirement |
|---|---|---|
| `.npy` | `numpy.load` (no pickles) | one array |
| `.npz` | `numpy.load` (no pickles) | the array named `mask`, or else the file's only array other than `spacing` / `spacing_unit`; optionally `spacing` (3 numbers) and `spacing_unit` (a string) |
| `.tif`, `.tiff` | `tifffile.imread` (`foambryo[cli]`) | one 3-D image; **the TIFF's resolution tags are not read** (§ 9, *TIFF metadata*) |

The array must be **three-dimensional, of an integer type, with no negative value**. Label 0 is the medium (exterior); every other label is a cell. **Axis 0 is z, axis 1 is y, axis 2 is x**: the order in which numpy and tifffile return a z-stack. The array is passed to the reconstruction unchanged (same values, same type).

### 3.2 Options

| Option | Default | Meaning |
|---|---|---|
| `MASK` | required | the mask file |
| `-o`, `--output MESH.rec` | required | the mesh file to write; the suffix must be `.rec` |
| `--spacing DZ DY DX` | from the `.npz`, else **required** | voxel size along axes 0, 1, 2, each finite and > 0; one axis may be longer than the other two, which must be equal. Give `--spacing 1 1 1` to keep voxel units |
| `--spacing-unit UNIT` | from the `.npz`, else `um` | the unit of the spacing: one word of ASCII letters, digits and `_ . - / ^` (it is written into the file as text; `µm` is refused, write `um`) |
| `--min-distance N` | `3`, the library's | minimum distance between the reconstruction's points, in cubic voxels of the shortest spacing, an integer ≥ 1 |
| `--overwrite` | off | replace `MESH.rec` if it exists; without it an existing file is an error |

When both `--spacing` and the `.npz` give a spacing and they differ, **the command line wins** and a warning names both values. The same holds for the unit.

There is deliberately **no default spacing**: microscopy voxels are rarely cubic, and a mesh read in the wrong proportions gives wrong angles, tensions and pressures without any other sign.

### 3.3 The mesh file

* Vertex coordinates are **physical**, in the unit of the spacing: the reconstruction runs on cubic voxels of the shortest spacing `h` (the mask itself when its voxels are cubic, its resampling otherwise), and `x_physical = x_cubic_voxel × h`; the centre of the mask's first voxel is the origin. With cubic voxels the coordinates are bit for bit those of the plain reconstruction multiplied by the spacing, and with `--spacing 1 1 1` the plain reconstruction's.
* **The file is byte for byte delaunay-watershed-3d's own `.rec` export** of that reconstruction (`save_to_rec_mesh`).
* Triangles and labels are the reconstruction's. When a cell of the mask reaches a face of the image box, the triangles that close it there (crop faces) are written as the last block of rows and recorded in the appendix, exactly as delaunay-watershed-3d's own `.rec` export does: every face of the image box is a crop (the library's default).
* Text `.rec`, with this appendix (keys of delaunay-watershed-3d's `.rec` appendix, version 1):

  ```text
  # rec-appendix 1
  # written_by = <delaunay-watershed-3d version>
  # coordinate_frame = physical
  # axis_order = zyx
  # spacing = <DZ> <DY> <DX>
  # source_shape = <n0> <n1> <n2>              the mask's own shape, before resampling
  # resampling = shape
  # resampled_spacing = <h>                     the shortest spacing
  # spacing_unit = <UNIT>
  # physical_image_faces = none                 only when a cell reaches the image box
  # crop_faces = <first row> <count>            only when a cell reaches the image box
  ```

  `spacing` is the mask's own voxel size: the coordinates are already scaled by it, so a reader must not scale them again. foambryo's `dcel_mesh_from_file` reads it as the mesh's `source_spacing`, which changes no number. (The first version of this specification also wrote `crop_offset = 0 0 0`; delaunay-watershed-3d's export does not, and the key is optional.)

### 3.4 Messages

* Once the spacing is accepted, before the reconstruction, one line on standard output reports what it will meet (`foambryo.segmentation_masks.segmentation_mask_report`, the report the napari plugin shows; the mask is not changed): `mask, for information (nothing is changed): <C> cells; <B> touch the image box (cut cells); <S> in several pieces; <H> pieces of medium enclosed by the tissue; <N> smaller than 5% of the median cell (<V> voxels)`. Pieces are connected through shared voxel faces. It is information, not a warning: neither command repairs a segmentation, and pieces and small labels are reconstructed as they are.
* On success, one line on standard output: `wrote MESH.rec: <P> points, <T> triangles, <C> cells (<K> cut by the image boundary)`.
* **Anisotropic voxels**: whenever the voxels are not cubic, a warning (`AnisotropicVoxelSpacingWarning`) says that the mask is resampled to cubic voxels, of which size, how many times the voxels that is, and that `--min-distance` counts cubic voxels (§ 10).
* **Above `k` = 3**, delaunay-watershed-3d's own `UnmeasuredAnisotropyWarning` is relayed as well: the accuracy was measured up to a ratio of 3 only. The mesh is built and the exit code is 0 (§ 9, *Anisotropic voxels*).
* Warnings raised by the reconstruction (for example the coarse-spacing warning at `--min-distance 5` or more) are relayed (§ 7).

## 4. `foambryo infer`

```text
foambryo infer MESH -o OUTDIR [--tension-method METHOD] [--pressure-method METHOD | --no-pressures]
                              [--mean-tension T] [--base-pressure P]
                              [--solver auto|dense|sparse] [--diagnostics none|basic|spectrum]
                              [--overwrite]
```

**What it does.** Reads a mesh with `foambryo.dcel_mesh_from_file(MESH)` and infers the tensions then the pressures, exactly as `foambryo.infer_forces` does with the same options, then writes the results and a run record. With `--no-pressures` it infers the tensions only: `infer_forces`' first call, `foambryo.infer_tensions`, with the same options.

### 4.1 Options

| Option | Default (the library's) | Library argument |
|---|---|---|
| `MESH` | required | any file `dcel_mesh_from_file` reads: `.rec` / `.arec` from delaunay-watershed-3d, or a file meshio reads with `label1` / `label2` cell data (`.vtk`, `.vtu`, …) |
| `-o`, `--output OUTDIR` | required | created if missing (with its parents) |
| `--tension-method` | `young-dupre` | `mode_tension`; one of `young-dupre`, `projection-young-dupre`, `equilibrium`, `cotan`, `inv-cotan`, `lami`, `inv-lami`, `log-lami`, `variational`, `force-balance`, `force-balance-soft-row`, `force-balance-map`, `young-dupre-local` (the order of `--help`) |
| `--pressure-method` | `variational` | `mode_pressure`; one of `variational`, `laplace`, `weighted-laplace` |
| `--mean-tension` | `1` | `mean_tension`, a finite number |
| `--base-pressure` | `0` | `base_pressure`, a finite number (the medium's pressure) |
| `--no-pressures` | off | the tensions only (`infer_tensions(mesh, mean_tension=…, mode=…, solver=…, diagnostics=…)`): no pressure solve, no `pressures.csv`; `--pressure-method` or `--base-pressure` given with it is refused (exit 2), even at its default value, since it would not be used |
| `--solver` | `auto` | `solver` |
| `--diagnostics` | `none` | `diagnostics`: `none` → `False`, `basic` → `True`, `spectrum` → `"spectrum"` |
| `--overwrite` | off | see § 4.3 |

The method names are the library's enumeration members (`TensionComputationMethod`, `PressureComputationMethod`) written in lower case with hyphens between words.

### 4.2 Output files

**`tensions.csv`**: one row per interface of the tension map, sorted by `label_1` then `label_2`.

```text
label_1,label_2,tension
0,1,0.8731...
1,2,1.0442...
```

* `label_1 < label_2` are the two materials of the interface (0 is the medium).
* `tension` is written with the shortest decimal representation that reads back as the same 64-bit float (Python's `repr`), so the file holds **exactly** the library's numbers. An undetermined tension is `nan`.
* An interface made only of crop faces carries no tension and has no row, as in the library's map.

**`pressures.csv`**: one row per material, sorted by `label`, including the medium (0), whose pressure is `--base-pressure`.

```text
label,pressure
0,0.0
1,0.1234...
```

* The same number format. A pressure the library does not report (a cell cut by the crop, or a cell no equation determines) is `nan`.
* **Written only when the pressures were computed**: not with `--no-pressures`, nor when the library refuses them (§ 4.4).

**`run.json`**: the run record, schema 1 (§ 5).

**`solves.csv`**: only with `--diagnostics basic` or `spectrum`. One row per linear solve, in the order they ran, with the columns

```text
index,system,solver,n_rows,n_unknowns,rank,residual_norm,condition_number,condition_number_source,seconds,lsqr_istop,lsqr_iterations,spectrum_seconds,spectrum_note,n_singular_values
0,"tensions YoungDupre, junction-graph component 0",dense lstsq,57,30,30,1.43…e-15,8.21…,…
```

* `index` is the solve's position in `run.json`'s `solves` and in `spectra.npz`'s names; the other columns are the scalars of the library's `SolveRecord` (`foambryo.linear_solves`), the same values as `run.json`'s `solves`: which system, the solver that produced the solution, the rows (equations) and unknowns of the system solved, its numerical rank (dense `lstsq` only), the residual norm, the condition number and how it was obtained (computed or estimated), the wall time, `lsqr`'s stop code and iterations (sparse only), the time spent on the spectrum and why a requested spectrum is missing or partial, and how many singular values `spectra.npz` holds for it.
* A value the solve does not have is an empty field; floats are in the exact format of `tensions.csv`. The run's warnings are not per solve: they are in `run.json`'s `warnings`.
* Written and read by `foambryo.results.write_solve_table` / `read_solve_table`, which the napari plugin uses too.

**`spectra.npz`**: only with `--diagnostics spectrum`. One array per solve that recorded singular values, named `singular_values_<i>`, `<i>` being the solve's position in `run.json`'s `solves` list; the values in decreasing order. On a dense solve these are all the singular values; on a sparse solve the largest and the smallest few, concatenated. The layout is the one `foambryo.save_solve_diagnostics` writes.

Units: the tensions are in the units of `--mean-tension` (relative tensions with the default 1). The pressures are in those units divided by the mesh's length unit: the `spacing_unit` of a mesh written by `foambryo mesh`, voxels for a mesh in voxel coordinates, and unknown otherwise. `run.json` states which.

### 4.3 Existing outputs

Without `--overwrite`, `infer` refuses to start if `OUTDIR` already holds any of `tensions.csv`, `pressures.csv`, `run.json`, `spectra.npz` or `solves.csv` (`foambryo.results.INFER_OUTPUTS`). With `--overwrite`, those five files (and only those) are deleted before the run, so that no file of an earlier run survives beside the new ones. Any other file in `OUTDIR` is left alone.

### 4.4 When the library refuses

On a mesh without crop faces, the pressure solve refuses tensions that are not all finite (an interface with no triple line has the tension `nan`; the library raises `UndeterminedTensionError`). Then `infer` writes `tensions.csv` and `run.json` (with `"status": "pressures refused"` and the library's message), writes no `pressures.csv`, prints `foambryo infer: error: pressures refused: <message> (the tensions were written)` and exits 4. Any other refusal by the library (a `ValueError`, including an option combination the library rejects, such as `--solver sparse` with a method that has no sparse path, or a malformed triple line) writes `run.json` with `"status": "failed"`, prints `foambryo infer: error: failed: <message> (...)` and exits 4. On a cropped mesh the library accepts undetermined tensions and reports the cut cells' pressures as `nan` (with a warning): that run is `"complete"`.

## 4b. `foambryo infer-over-seeds`

```text
foambryo infer-over-seeds MASK -o OUTDIR [--spacing DZ DY DX] [--spacing-unit UNIT] [--min-distance N]
                                         [--seeds N] [--combine tensions|angles] [--angle-reading READING]
                                         [--tension-method METHOD] [--max-workers K] [--overwrite]
```

**What it does.** Reconstructs the mask at the dither seeds 0..N−1, each as `foambryo mesh` would (delaunay-watershed-3d's default reconstruction, resampled to cubic voxels when one axis is longer), infers the tensions of each mesh, and combines them per interface, exactly as `foambryo.infer_tensions_over_seeds(mask, seeds=range(N), spacing=..., ...)` does. It writes `tensions.csv`, `tensions_per_seed.csv` and `run.json`; **no pressures**, because no single mesh carries them. The spread over seeds is reported as a **lower bound on the error, not an error bar**. N is fixed in advance: seeds are never chosen by their result.

| Option | Default (the library's) | Library argument |
|---|---|---|
| `MASK` | required | read as in § 3.1 |
| `-o`, `--output OUTDIR` | required | created if missing |
| `--spacing DZ DY DX`, `--spacing-unit UNIT` | as for `mesh` (§ 3.2): no default spacing | `spacing` |
| `--min-distance` | `3` | `min_distance` (cubic voxels) |
| `--seeds N` | `8` (the library's seed list, 0..7) | `seeds=range(N)` |
| `--combine` | `tensions` | `combine`: `tensions` averages the seeds' tensions; `angles` (**experimental**, `young-dupre` only) averages their contact angles and infers once, and warns `SeedEnsembleExperimentalWarning` |
| `--angle-reading` | `chord` | `angle_reading`; one of `chord`, `quadric`, `oriented_chord`, `oriented_quadric` |
| `--tension-method` | `young-dupre` | `mode` |
| `--max-workers K` | `1` | `max_workers`: K seeds at once, in separate processes, each holding one reconstruction; the result does not depend on it |
| `--overwrite` | off | deletes `tensions.csv`, `tensions_per_seed.csv` and `run.json` (only those) before the run |

**`tensions.csv`**: one row per interface found by at least one seed, sorted by `label_1` then `label_2`, with the columns

```text
label_1,label_2,tension,tension_seed_sd,n_seeds_found,found_by_fewer_than_half_the_seeds
0,1,0.8731...,0.0412...,8,false
```

* `tension`: the ensemble tension (the mean over the seeds that found the interface, or with `angles` the one solve on the averaged angles); `tension_seed_sd`: the standard deviation of the seeds' own tensions (`nan` with fewer than two); both in the exact float format of § 4.2.
* `n_seeds_found`: how many seeds found the interface with a finite tension.
* `found_by_fewer_than_half_the_seeds`: `true` or `false`. On synthetic tissues with known geometry most such interfaces do not exist in the true tissue; they are kept, flagged, not removed.

**`tensions_per_seed.csv`**: every seed's own tensions, as `infer_tensions` gives them on that seed's mesh, one row per seed and interface, in the seeds' order and, within a seed, sorted by `label_1` then `label_2`; same number format.

```text
seed,label_1,label_2,tension
0,0,1,0.8692...
0,1,2,1.0511...
1,0,1,0.8774...
```

**`run.json`**, schema 1 (§ 5), with `command` = `"foambryo infer-over-seeds"` and these keys in place of the mesh-related ones:
* `input`: `path`, `sha256` and `bytes` of the mask file, its `shape`, the `spacing` used, its `spacing_unit`, and `spacing_from` (`"command line"` or the file's name);
* `options`: `seeds` (N), `combine`, `angle_reading`, `tension_method`, `min_distance`, `max_workers`;
* `seeds`: the seed list, `[0, 1, …, N−1]`;
* `reconstructions`: one object per seed, with `seed`, `points`, `triangles`, `crop_faces` and `seconds`;
* `results`: `tensions`, `undetermined_tensions` and `found_by_fewer_than_half_the_seeds` (counts);
* `units`: `tension` (relative, each seed gauged to a mean of 1) and `length` (the spacing's unit).

`library_call`, `status` (`complete` or `failed`), `error`, `warnings` and `outputs` are as in § 5.

## 5. `run.json`, schema 1

A JSON object, UTF-8, indented. Non-finite numbers are written as the strings `"nan"`, `"inf"`, `"-inf"` (JSON has no such numbers). **Keys may be added within schema 1; a reader must ignore keys it does not know.** Removing or renaming a key, or changing a key's meaning, makes a new schema number and is announced in the CHANGELOG.

| Key | Type | Content |
|---|---|---|
| `schema` | integer | `1` |
| `command` | string | `"foambryo infer"` |
| `arguments` | list of strings | the arguments after `foambryo`, as given |
| `date` | string | start of the run, ISO 8601 in UTC (`2026-10-10T10:12:00+00:00`) |
| `elapsed_seconds` | number | wall time from reading the mesh to writing the last result |
| `versions` | object | `python`, `foambryo`, `delaunay-watershed-3d`, `numpy`, `scipy` (installed distribution versions) |
| `input` | object | `path` (as given), `sha256` (of the file's bytes), `bytes`, `metadata` (the `.rec` appendix as delaunay-watershed-3d reads it, its version under `appendix_version`; `null` for a file with no appendix or not a `.rec`) |
| `options` | object | `tension_method`, `pressure_method`, `mean_tension`, `base_pressure`, `solver`, `diagnostics`: every option, in the command's spelling, defaults included. With `--no-pressures`, also `no_pressures: true`, and `pressure_method` and `base_pressure` are `null` (not used); without it there is no `no_pressures` key |
| `library_call` | string | the equivalent Python call, e.g. `infer_forces(mesh, mean_tension=1.0, base_pressure=0.0, mode_tension=TensionComputationMethod.YoungDupre, mode_pressure=PressureComputationMethod.Variational, solver='auto', diagnostics=False)`; with `--no-pressures`, `infer_tensions(mesh, mean_tension=1.0, mode=TensionComputationMethod.YoungDupre, solver='auto', diagnostics=False)` |
| `mesh` | object | `points`, `triangles`, `materials` (medium included), `crop_faces` (count), `cut_cells` (list of labels) |
| `results` | object | `tensions`, `undetermined_tensions`, `pressures`, `undetermined_pressures` (counts; the pressure counts are `null` when no pressure was computed, with `--no-pressures` among other cases) |
| `units` | object | `tension` (`relative, gauged to a mean tension of <T>`), `pressure` (`tension per <unit>`, the unit being the appendix's `spacing_unit` for physical coordinates, `voxel` for voxel coordinates, else `mesh length unit (not recorded in the mesh file)`) |
| `status` | string | `"complete"`, `"pressures refused"` or `"failed"` |
| `error` | string or `null` | the library's message when `status` is not `"complete"` |
| `warnings` | list | one object per distinct warning: `category` (the class name), `message`, `count` |
| `solves` | list or `null` | with `--diagnostics basic` or `spectrum`: one object per linear solve, in the order they ran (the scalars of the library's `SolveRecord`: `system`, `solver`, `n_rows`, `n_unknowns`, `rank`, `residual_norm`, `condition_number`, `condition_number_source`, `seconds`, …; the same values as `solves.csv`); `null` with `none` |
| `outputs` | list of strings | the files written in `OUTDIR`, `run.json` included |

## 6. `foambryo view`

```text
foambryo view MESH [--results OUTDIR] [--colour-bars FILE] [--figure FILE] [--no-show]
                   [--scale-bar LENGTH | --no-scale-bar] [--scattered [COEFF]]
                   [--tension-method METHOD] [--pressure-method METHOD | --no-pressures] [--mean-tension T]
                   [--base-pressure P] [--solver auto|dense|sparse]
foambryo view MASK [--spacing DZ DY DX] [--spacing-unit UNIT] [--min-distance N] [--scattered [COEFF]]
                   [--scale-bar LENGTH | --no-scale-bar] [the inference options above]
```

| Option | Default | Meaning |
|---|---|---|
| `MESH` | required | read with `dcel_mesh_from_file` |
| `--results OUTDIR` | none | read what `foambryo infer` or `foambryo infer-over-seeds` wrote in `OUTDIR` (§ 6.2). Without `--results`, the tensions then the pressures are inferred **exactly as `foambryo infer` does with the same options** (the inference options below); if the library refuses the pressures, the tensions alone are shown, with a warning; if it refuses the tensions, exit 4 |
| `--colour-bars FILE` | none | write the colour bars of what the viewer shows (§ 6.1) to `FILE`; the format follows the suffix (`.png`, `.pdf`, `.svg`, …). Needs matplotlib. Written before any window opens |
| `--figure FILE` | none | write the window's first view with its colour bars and its scale bar (§ 6.4) to `FILE` (`.png`, `.pdf`, `.svg`, …). Needs Polyscope and a display. Prints `wrote FILE` |
| `--no-show` | off | do not keep a viewer window open. With `--colour-bars` alone it runs without a display and without Polyscope installed; with `--figure`, Polyscope renders the view once and the window closes. Without either, refused (exit 2) |
| `--scale-bar LENGTH` | a round length | the length scale bar's length, in the mesh's unit (§ 6.3) |
| `--no-scale-bar` | off | draw no length scale bar |
| `--scattered [COEFF]` | off; `COEFF` 0.2, the viewer functions' `scatter_coeff` | open the window exploded: each cell moved by `COEFF × (its centroid − the tissue's centroid)` (`DcelData.compute_scattered_arrays`); the panel changes it |
| `--no-pressures` | off | the tensions only (`infer_tensions`), as `foambryo infer --no-pressures`; refused with `--pressure-method`, `--base-pressure` or `--results` |
| `MASK`, `--spacing`, `--spacing-unit`, `--min-distance` | as for `mesh` (§ 3.2) | a mask (`.npy`, `.npz`, `.tif`, `.tiff`) is opened in the window to be reconstructed there (§ 6.5); the three options are refused with a mesh |
| `--tension-method`, `--pressure-method`, `--mean-tension`, `--base-pressure`, `--solver` | `infer`'s (§ 4.1): the library's | the options of `foambryo infer`, passed to the same library calls, when the viewer infers. **Refused with `--results`** (exit 2): the values are then read, not inferred. `--diagnostics` is not an option of `view`: the window's *Export* has its own diagnostics choice (§ 6) |

The window is `foambryo.viewing.plot_force_inference(mesh, tensions, pressures, ...)`, or `plot_tension_inference(mesh, tensions, ...)` when there are no pressures (an ensemble, `--no-pressures`, or pressures the library refused). It needs Polyscope and a display (`foambryo[viewing]`). Its panel, *foambryo*, shows:

* **which values the colours show**: the library call that produced them (`inferred by 'foambryo view': infer_forces(mesh, …)`), or `read from NAME (foambryo infer: <run.json's library_call>)`, NAME the results folder's own name, never the path to it;
* **the colour bars** of § 6.1, drawn in the window with the library's colours over the library's ranges; an outlined green swatch, *undetermined (nan)*, when a value is `nan` (or an interface has no tension), and a glass swatch, *crop face*, when the mesh has crop faces;
* with tensions and pressures, **colour the surface by** *tensions* (the default) or *pressures*. In the tension view an undetermined tension's interface is not filled and its triangles' edges are drawn in the undetermined green (*undetermined tensions' outlines*, a check box); crop faces are never filled with a value's colour and are drawn as glass, with their edges (*crop faces*, a check box hides them);
* the **background** (a choice): white (Polyscope's), light grey 0.70, dark grey 0.20, black (`BACKGROUND_COLOURS`, the napari plugin's choices too) or a custom colour; on a dark background the scale bar, the frame, their labels and the graph view's black nodes are drawn light (the mirror of their grey), on a light one dark; a background set in Polyscope's own menu is followed too. Polyscope tone-maps its background (exposure 1.1, gamma 2.2: 0.20 was shown 0.50): the window gives it `(c ** 2.2) / 1.1`, which it shows as `c`;
* the **x, y, z frame** (a check box): three segments along the mesh's columns (z, y, x), from just outside the drawn box's lowest corner (moved out by 10 % of its largest side), of the scale bar's round length, labelled `z`, `y`, `x` at their tips and with their length at the corner, in the window and in saved figures;
* *flat colours*: draws the surface unlit, so that its colours are exactly the bars' (by default the surface is shaded, as any 3-D view);
* the scale bar's check box (§ 6.3);
* **Save figure** and **Save colour bars**, each with a file name (default `foambryo_figure.png`, `foambryo_colour_bars.png`, in the working directory): PNG, PDF or SVG by the suffix. **The window never overwrites a file**: an existing name gets `_2`, `_3`, …; the panel says which file it wrote;
* **Explode**: a check box and a coefficient (0 to 1), the scattered view, followed live (each view is prepared on a background thread, the window keeps turning and draws the last one asked for); **Cut**: a scene slice plane normal to z, y or x (the mesh's axes 0, 1, 2), at a position in the mesh's unit along the drawn box, keeping the lower or the higher side (Polyscope keeps the side the plane's normal points to: measured in a real window, 2.6.1);
* a choice of tension method, pressure method or *pressures* unticked (the tensions only), and solver, and **Recompute**: the library is called again with them (and the same mean tension and base pressure), and the colours and bars follow; then the `foambryo infer MESH -o results …` line that writes the same numbers. Not for an ensemble. A *diagnostics* choice (`none`, `basic`, `spectrum`) keeps the solve records for Export;
* **Export**, to a folder (default `foambryo_export`): `foambryo infer`'s files for the values shown, written by the command's writers, `tensions.csv`, `pressures.csv` (when inferred), `solves.csv` and `spectra.npz` (with diagnostics), `run.json` (the command's record, its `input` the mesh file), and `colour_bars.png` as `--colour-bars` writes it; a mesh reconstructed in the window adds its `mesh.rec`. When the folder holds any of these files, the panel lists them and asks *Cancel* or *Replace*; another file is never touched. Values given to the window (`--results`) are exported with a `run.json` that says so (`values_given_to_the_window`). Not for an ensemble (`foambryo infer-over-seeds` writes its files).

The coloured surface draws the back of a triangle like its front: the triangles' winding follows the label order, so the outside of a tissue faces away from Polyscope's camera, and Polyscope's default shading of back faces drew every colour about 15 % darker than its bar.

### 6.1 Colour bars

`foambryo.viewing.save_colour_bars(mesh, dict_tensions, dict_pressures, path, alpha=0.05)` writes one image with one bar per quantity. The bars use **the same computation as the viewer's surface colours**, through one shared function:

* **Surface tension**: colour map `coolwarm` (blue for low, light grey for the mean, red for high), over a range **centred on the mean of the finite tensions** of the map (one value per interface; the mean is `mean_tension`, 1 by default, for every method that meets its gauge exactly) and symmetric around it, its half-width the farther of the `alpha` and `1 − alpha` quantiles from the mean (`tension_colour_range`). Tensions beyond either end are drawn with the end's colour (the viewer clips them). The bar's middle tick is the mean. (Until 0.7.0: `jet` between the two quantiles.)
* **Pressure**: colour map `magma` reversed (light for low, dark for high) and trimmed to its 10–90 % range (`PRESSURE_COLOUR_MAP_RANGE`: a fraction `x` is drawn `magma(0.1 + 0.8 x)`, so no pressure is black or near-white and none vanishes on a black or a white background; until 0.7.0 the whole map), from 0 to the `1 − alpha` quantile of the finite pressures **over the mesh's triangles**, each triangle carrying the pressure of the higher-labelled of its two materials (this is how the viewer's surface colours them). Negative pressures are drawn with the colour of 0. **This is the viewer's only pressure scale**: the surface, the graph view's cell nodes and the bar take it from one function, so a pressure has the same colour in all three. (Until this revision the graph view took its own quantile over cells: 5.383 against the surface's 5.472 on benchmark case 000.)
* A swatch labelled "undetermined" when any value is `nan`: green (`UNDETERMINED_COLOR`, #1b9e77), outside both maps; a viewer that can draws an undetermined interface as its green outline, without fill, and fills it green otherwise. A "crop face" swatch when the mesh has crop faces: see-through glass with its edges drawn (`CROP_FACE_COLOR`, `CROP_FACE_OPACITY`, `CROP_FACE_EDGE_COLOR`); crop faces carry no value.
* Each bar's two ends are labelled with their values. The function returns them, so they can be checked against the viewer's own range (the viewer prints it as "Extremas …").
* matplotlib only (its non-interactive canvas): no window, no display, no Polyscope.
* `foambryo view` prints `wrote FILE: tension <low> to <high>, pressure 0 to <high>` (6 significant digits; the pressure part only when there are pressures). No finite tension: exit 4, `no colour bar: no tension is finite: there is no colour scale to draw`.

**In the window** the same bars are drawn by the panel (ImGui rectangles, flat colours, no interpolation): the colours are those `save_colour_bars` samples, over the ranges `colour_bar_ranges` returns. Polyscope draws the surface from the colours, not from the values: its own colour maps would draw `nan` in their top colour. The graph view (cell nodes, and edges between them) is coloured over the same two ranges. The scattered view (`plot_force_inference(..., scattered=True)`) colours pressures over its own triangles (each cell's triangles carry that cell's pressure), and its pressure bar says so. *Save colour bars* in the panel writes `save_colour_bars`' image whenever the window's bars are its bars (not scattered, not an ensemble), and the panel's own bars otherwise.

### 6.2 Results of `infer` and of `infer-over-seeds`

* **`infer`'s results**: `OUTDIR/tensions.csv` and, if present, `OUTDIR/pressures.csv`. If `OUTDIR/run.json` exists, its `input.sha256` must equal `MESH`'s: results computed from another mesh are refused (exit 3).
* **`infer-over-seeds`' results**, recognised by the columns of `tensions.csv` (§ 4b): the ensemble tensions are drawn on `MESH`'s interfaces, matched by label pair (`MESH` is one reconstruction of the same mask, for example `foambryo mesh`'s). Their `run.json` names the mask, not a mesh, so the mesh cannot be checked against it: a warning says how many of the ensemble's interfaces are not on `MESH` (not drawn) and how many of `MESH`'s are not in the ensemble (drawn undetermined), and **no interface in common is refused** (exit 3). The colour scale and `--colour-bars` are taken over the interfaces drawn. The window adds the seeds' spread, `tension_seed_sd`, as a second surface quantity with its own bar (`jet`, clipped at the 5 % and 95 % quantiles), and draws the interfaces flagged `found_by_fewer_than_half_the_seeds` over the surface, as a magenta wireframe (their triangles' edges, so that their own colour stays visible). No pressure is shown: no single mesh carries an ensemble's pressures.

### 6.3 The length scale bar

When `MESH` is a `.rec` file whose appendix records physical coordinates and their unit (`coordinate_frame = physical`, `spacing_unit`; `foambryo mesh` writes both), the window draws a segment in the scene, under the tissue (along the first coordinate axis, 8 % of the tissue's largest side below it), of length 1, 2 or 5 times a power of ten, the longest at most a quarter of the tissue's largest side (or `--scale-bar LENGTH`). Its length is written beside it in the window and in saved figures (`20 µm`; the unit `um` is written `µm`). It is exact in 3-D and foreshortened on screen like any length in a perspective view. A mesh without a recorded unit has no scale bar, unless `--scale-bar` is given: it is then labelled in `voxel` for a mesh in voxel coordinates, else in `mesh units`.

### 6.4 Figures

A figure is Polyscope's own rendering of the view, without the user interface (`screenshot_to_buffer`), and under it, drawn by matplotlib: the colour bars of § 6.1, the swatches of what has no value, and a caption, `Values: …`, the panel's first line (it names the results folder, never its path). The scale bar's length is written under the segment, where Polyscope's camera projects its ends (in the lower left corner when an end is off the view). `--figure` writes the first view, framed on `MESH` (Polyscope's home view); *Save figure* writes the current one. From Python: `plot_force_inference(mesh, ..., save_figure=PATH, show=False)`.

### 6.5 From a mask

`foambryo view MASK --spacing DZ DY DX` reads the mask and its spacing exactly as `foambryo mesh` does (the same checks, messages and exit codes; the mask report is printed), then opens the window **on the mask**: its cell voxels as points at `index × spacing`, one colour per label, the medium left out (strided above three million points). Polyscope's volume grids cannot leave a value out, so the medium would hide the tissue: the points are what lets the cells be seen, and the cut slices them. The panel then offers, top to bottom once there are values:

* **Values** (as in § 6, without Recompute);
* **Mask and reconstruction**: *show the mask*, `min distance`, the estimate (time and memory, `foambryo.reconstruction_cost`: a fit to measured reconstructions, with its in-sample band, the one the napari plugin gives), **Reconstruct** and **Cancel**. The reconstruction is `foambryo.io.reconstruct_mask` (`foambryo mesh`'s call) in a child process: the window keeps drawing (no frame over 1 s on benchmark case 000), shows the elapsed time, and Cancel stops the process and keeps nothing. The mesh is then drawn half transparent over the mask;
* **Inference**: *one mesh* (`infer_tensions_then_pressures` on the reconstruction, `foambryo infer`'s call) or *seed ensemble* (`infer_tensions_over_seeds` on the mask, `foambryo infer-over-seeds`' call, with the number of seeds, workers and combination; its progress seed by seed), the tension method, the pressure method or *pressures* unticked, the solver, the diagnostics (one mesh); **Infer** and **Cancel**, also in a child process; and the command lines that give the same numbers (`foambryo mesh …` then `foambryo infer …`, or `foambryo infer-over-seeds …`). Once values are drawn the mask is hidden (its points would speckle the surface); its check box shows it again.

With a mask, `--results`, `--figure`, `--colour-bars` and `--no-show` are refused (exit 2): they need a mesh; `foambryo mesh` writes it without a window. The values are drawn once prepared in the background: the mask and the reconstruction stay on screen until then.

From Python, `foambryo.viewing.view_mask(mask, spacing, *, spacing_unit="um", min_distance=None, mean_tension, base_pressure, mode_tension, mode_pressure, solver, pressures=True, scattered=False, scatter_coeff=0.2, scale_bar=True, mask_name="mask.npz")` opens the same window (every default the library's; `min_distance=None` is delaunay-watershed-3d's); a mask or spacing the command refuses raises `ValueError`.

## 7. Messages and exit codes

* **Errors** go to standard error as one line, `foambryo <sub-command>: error: <message>`, followed by the exit code below. No Python traceback, except for an unexpected error (code 1).
* **Warnings** (the library's, and the command's own) go to standard error as `foambryo <sub-command>: warning: <Category>: <message>`, once per distinct warning, with ` (<n> times)` appended when it was raised more than once. `infer` also records them in `run.json`.
* **Results** (the lines naming each file written) go to standard output.

| Code | Meaning | Examples |
|---:|---|---|
| 0 | success | |
| 1 | unexpected error: a defect, reported with its traceback | |
| 2 | the command line is invalid or incomplete | unknown option; a value out of range; **a mask that records no spacing, without `--spacing`**; an output suffix other than `.rec`; `--no-show` without `--colour-bars` or `--figure`; an inference option of `view` with `--results` |
| 3 | a file cannot be used as asked | **an input that does not exist**; a file that cannot be read or does not hold what is expected (not 3-D integer labels, no mask array, no `label1` / `label2`); results from another mesh; an output that exists without `--overwrite`; an optional package that is not installed |
| 4 | the reconstruction or the inference refused this input | the library raised its documented refusal (§ 4.4); what was computed before the refusal is written |

The messages the acceptance tests pin:

```text
foambryo mesh: error: no such file: 'missing.npy'
foambryo infer: error: no such file: 'missing.rec'
foambryo mesh: error: mask.npy records no voxel spacing: give it with --spacing DZ DY DX (for example --spacing 1 1 1 to keep voxel units)
```

## 8. What equals what (the acceptance tests)

1. `infer`: `tensions.csv` and `pressures.csv`, read back as 64-bit floats, are **bitwise equal** to `infer_forces(dcel_mesh_from_file(MESH), …)` with the same options, for every tension method, every pressure method, and non-default `--mean-tension` / `--base-pressure` / `--solver` / `--diagnostics`.
2. `mesh`: the arrays of `MESH.rec` (points, triangles, labels, crop-face mask) have the same SHA-256 as `construct_mesh_in_physical_units`'s, in the same crop-face row order, and the file is byte for byte that reconstruction's own export; with and without a cell reaching the image box; with cubic voxels of 1 and of 0.5 and with a z step twice the pixel size. With cubic voxels, the points are also those of the plain reconstruction times the spacing. A spacing with two different short axes is refused (exit 2, or 3 from an `.npz`); above `k` = 3 both warnings are printed.
3. `run.json`: every key of § 5 is present with its type. `solves.csv` reads back equal to the same run's `solves`, field by field, floats bitwise. 3b. `infer --no-pressures`: `tensions.csv` is bitwise `infer_tensions` (and `infer_forces`' tensions) with the same options; no `pressures.csv`; a pressure option given with it is refused.
4. Exit codes and messages: a missing mask, a missing mesh, a mask without spacing.
5. Colour bars: the ends of each bar equal the range the viewer prints for the same input; the image is written with no display and with Polyscope absent.
6. No defaults of the command differ from the library's (a test reads the library's signatures).
7. `infer-over-seeds`: every column of `tensions.csv`, and every row of `tensions_per_seed.csv`, read back, equals `infer_tensions_over_seeds`' result for the same mask, seeds, spacing and options, bitwise for the floats, with cubic voxels, with a z step twice the pixel size, and with `--combine angles --angle-reading quadric`; `run.json` lists the seeds; a refused combination writes `run.json` only and exits 4.

Synthetic masks and benchmark meshes only.

## 9. Possible extensions

None of these is in schema 1; each would be a change to this specification. Requests and comments are welcome on the issue tracker.

* **Formats.** Masks are read from `.npy`, `.npz` and `.tif`, meshes from `.rec`; `mesh` could also write `.vtk`.
* **TIFF metadata.** `mesh` does not read the spacing from a TIFF (ImageJ `spacing` and the resolution tags), because these tags are often missing or wrong and a wrong spacing is silent; it could, for software whose tags are reliable.
* **Exit codes.** Codes 2, 3 and 4 tell batch runs why a command stopped; a simpler "0 or not 0" would also do.
* **Anisotropic voxels.** The mask is resampled to cubic voxels before meshing (§ 3), with a warning whenever the voxels are not cubic and a second one above `k` = 3, where the accuracy was not measured. The first warning could become optional for batch runs.
* **Options not in schema 1.** Each is a library option already: `--angle-reading` for `infer` (`infer-over-seeds` has it), `--sparse-threshold N`, a way to declare physical image faces (a coverslip), the wedge-refusal policies, and a table of tensions per junction-graph component.
* **A record for `mesh`.** `mesh` could also write a JSON record (versions, options, the mask's SHA-256); today the provenance chain starts at the mesh file (its SHA-256 in `run.json`).
* **Preprocessing.** A `foambryo preprocess` sub-command could expose delaunay-watershed-3d's mask-preprocessing functions.
* **Seed ensembles (§ 4b).** `infer-over-seeds` starts from the mask and writes no pressures. Its per-seed table, `tensions_per_seed.csv`, is a long table; one column per seed is the alternative. The default N = 8 costs eight reconstructions, each with its memory.

## 10. Every message, verbatim

`<name>` is the file's name, `<given>` the path as typed. Each error line is prefixed with `foambryo <sub-command>: error: `; each warning with `foambryo <sub-command>: warning: <Category>: `.

| Code | Sub-command | Message |
|---:|---|---|
| 2 | all | argparse's usage line, then `<option>: <reason>` (`must be a finite number, not '…'`, `must be greater than 0, not '…'`, `must be an integer of at least 1, not '…'`, `must be one ASCII word (letters, digits, _ . - / ^), not '…'`, `invalid choice: …`) |
| 2 | mesh | `<name> records no voxel spacing: give it with --spacing DZ DY DX (for example --spacing 1 1 1 to keep voxel units)` |
| 2 | mesh | `the mesh is written as .rec: -o must end in .rec, not '<name>'` |
| 2 | mesh | `--spacing: spacing (<dz>, <dy>, <dx>) has two different fine axes; resample_to_isotropic supports one coarse axis with the other two equal (a z stack)` |
| 2 | view | `--no-show needs --colour-bars: there would be nothing to do` (with neither `--colour-bars` nor `--figure`) |
| 2 | view | `--spacing applies to a mask: <name> is a mesh` (and `--spacing-unit`, `--min-distance`) |
| 2 | view | `--results needs a mesh: <name> is a mask, reconstructed in the window (run 'foambryo mesh' first to write it without a window)` (and `--figure`, `--colour-bars`); `--no-show: a mask is opened in the window to be reconstructed there (<name>)` |
| 2 | view | `--pressure-method applies to the pressures: --no-pressures infers the tensions only` (and `--base-pressure`); `--no-pressures applies when 'foambryo view' infers: with --results the values are read from OUTDIR` |
| 2, 3 | view | with a mask: the mask and spacing messages of `mesh` |
| 2 | view | `--<option> applies when 'foambryo view' infers: with --results the values are read from OUTDIR` (an inference option with `--results`) |
| 3 | all | `no such file: '<given>'` |
| 3 | mesh | `<given> exists: pass --overwrite to replace it` |
| 3 | mesh | `<name>: unsupported mask format '<suffix>'; use .npy, .npz, .tif, .tiff` |
| 3 | mesh | `reading a TIFF mask needs tifffile: pip install 'foambryo[cli]'` |
| 3 | mesh | `cannot read <name>: <reader's message>` |
| 3 | mesh | `<name> holds a <n>-D array; a mask is 3-D` |
| 3 | mesh | `<name> holds <dtype> values; a mask holds integer labels` |
| 3 | mesh | `<name> holds negative labels; labels are 0 (medium) or positive` |
| 3 | mesh | `<name> holds the arrays [...]: name the mask 'mask'` |
| 3 | mesh | `<name>: 'spacing' must be three finite numbers > 0, not [...]` / `<name>: 'spacing_unit' must be one ASCII word, not '…'` |
| 3 | mesh | `<name>: spacing (…) has two different fine axes; …` (the `.npz`'s spacing, as above) |
| 3 | mesh | `foambryo mesh needs delaunay-watershed-3d 0.7 or later (installed: <version>): pip install -U delaunay-watershed-3d` |
| 3 | infer, view | `cannot read the mesh <name>: <error type>: <reader's message>` |
| 2 | infer | `--pressure-method applies to the pressures: --no-pressures infers the tensions only` (and the same for `--base-pressure`) |
| 3 | infer | `<given> exists and is not a directory` |
| 3 | infer | `<given> already holds tensions.csv, …: pass --overwrite to replace them` |
| 3 | view | `<OUTDIR>/run.json was computed from another mesh (SHA-256 <12 hex>...; <name> is <12 hex>...)` |
| 3 | view | `cannot read <OUTDIR>/run.json: …`; `tensions.csv: expected the header label_1,label_2,tension, found …`; `… every row must have 3 values` |
| 3 | view | `no interface of <name> is in <OUTDIR>/tensions.csv: the ensemble was computed from another mask` |
| 3 | view | `the colour bars need matplotlib: pip install 'foambryo[viewing]' (…)` / `the viewer needs Polyscope: pip install 'foambryo[viewing]' (…)` |
| 4 | mesh | `the reconstruction refused <name>: <library's message>` |
| 4 | infer | `pressures refused: <library's message> (the tensions were written)` |
| 4 | infer | `failed: <library's message> (the tensions were written)` or `(nothing but run.json was written)` |
| 4 | view | `no colour bar: no tension is finite: there is no colour scale to draw` |
| 4 | view | `failed: <library's message>` (the library refused the tensions) |
| — | mesh | warning `AnisotropicVoxelSpacingWarning: the voxels are <k> times longer along axis <i> (<z, y or x>) than along the other two: the mask is resampled to cubic voxels of <h> <UNIT> by shape-based interpolation before meshing (<k> times the voxels; --min-distance counts cubic voxels)` |
| — | mesh | warning `UnmeasuredAnisotropyWarning: voxels <k> times longer along axis <i> than along the other two: the accuracy of meshing after resampling was measured up to a ratio of 3 only (on larger cells the contact-angle error kept growing above it). The mesh is built anyway.` (delaunay-watershed-3d's, above `k` = 3) |
| — | mesh | warning `UserWarning: --spacing <dz dy dx> replaces the spacing <name> records, <dz dy dx>` (and the same for `--spacing-unit`) |
| — | view | warning `UserWarning: the pressures are not shown: <library's message>` |
| — | view | warning `UserWarning: <k> interfaces of the ensemble are not on <name> (not drawn); <m> interfaces of <name> are not in the ensemble (drawn undetermined)` |
| 2, 3 | infer-over-seeds | the mask and spacing messages of `mesh` (missing spacing, unreadable mask, two different short axes) |
| 3 | infer-over-seeds | `<given> exists and is not a directory`; `<given> already holds tensions.csv, …: pass --overwrite to replace them` (the files of § 4b it holds) |
| 3 | infer-over-seeds | `foambryo infer-over-seeds needs delaunay-watershed-3d 0.7 or later (installed: <version>): pip install -U delaunay-watershed-3d` |
| 4 | infer-over-seeds | `failed: <library's message> (nothing but run.json was written)` |
| — | infer-over-seeds | warning `SeedEnsembleExperimentalWarning: combine='angles' is experimental: …` (with `--combine angles`), the `AnisotropicVoxelSpacingWarning` of `mesh`, and each seed's own warnings prefixed with `seed <s>: ` |

On success each sub-command prints, on standard output, one `wrote <path>…` line per file written.

## 11. Changes since the first version

* **14 Oct 2026: the text agrees with the command** (§ 2, § 4.1, § 4b, § 6, § 7, § 10): `solves.csv` in the pipeline, the order of the tension methods, the four values of `--angle-reading`, `--spacing` optional for an `.npz` that records it in `view MASK`, the window's diagnostics choice, `--no-pressures` in the choice of window. The warning for interfaces missing from an ensemble now says they are drawn undetermined (it said grey; they are drawn as undetermined values are, since the green outlines).

* **12 Oct 2026: `foambryo view MASK`, `--scattered`, `--no-pressures`** (§ 6, § 6.5): the window starts from a mask, reconstructs and infers it in child processes; the panel has explode, cut, the solver and *pressures*.

* **12 Oct 2026: `foambryo infer --no-pressures` and `solves.csv`** (§ 4). The tensions only, and the solve diagnostics as a table beside `run.json`; the mask report of `foambryo mesh` is worded as information (§ 3.4).

* **11 Oct 2026: `foambryo mesh` resamples anisotropic masks to cubic voxels** (§ 3), and the viewer has one pressure scale (§ 6.1).
* **11 Oct 2026: a fourth sub-command, `foambryo infer-over-seeds` (§ 4b).** Several reconstructions of one mask at fixed dither seeds, with the tensions combined over them, and the three new `tensions.csv` columns `tension_seed_sd`, `n_seeds_found` and `found_by_fewer_than_half_the_seeds`, and a second table, `tensions_per_seed.csv`, with every seed's own tensions. `run.json` gains `seeds` and `reconstructions`. Its acceptance test is § 8, item 7, its messages are in § 10, and its possible extensions are in § 9, *Seed ensembles*.
* **12 Oct 2026: the viewer's window** (§ 6): the colour bars drawn in the window, the coloured surface's back faces drawn like its front faces, a length scale bar (§ 6.3), `--figure` (§ 6.4) and *Save figure* / *Save colour bars* in the window, `infer`'s inference options for `view` (refused with `--results`), the results of `infer-over-seeds` (§ 6.2), and a choice of methods with *Recompute* in the window. A refused tension inference in `view` now exits 4 with `failed: …` (it raised a traceback, exit 1).
