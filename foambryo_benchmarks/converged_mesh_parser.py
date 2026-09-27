"""Parser for the Simba/LosTopos `multitensions.opt` optimisation-stage tension files.

Used by the converged-mesh-cohort benchmark (`foambryo_benchmarks/run_converged_cohort.py`).
Semantics are **fixed by the generator's C++ source**, not inferred, and are hand-verified
against `converged-meshes/90LumenA_.../multitensions.opt` in `tests/test_converged_mesh_parser.py`
for stages 0, 10 and 17 (17 exercises hold-last-value).

File grammar (one `multitensions.opt` per converged-mesh folder)::

    # Ncells NOptStages          <- comment line, ignored
    <n_cells> <n_stages>          <- header

    <rangeA>:<rangeB> <v1>_<v2>_..._<vk>
    ...

`n_regions = n_cells + 1` (region label 0 is the external medium; regions `1..n_cells` are
cells). Each rule sets `tension(a, b)` for every `a` in `rangeA`, `b` in `rangeB` (skipping
`a == b`, which is not an interface). A range is a comma-separated list of single integers
and/or inclusive `a-b` spans, e.g. `2-23,40,55-57`.

**Stage rule (hold-last-value):** stage `i` (0-indexed, `0 <= i < n_stages`) reads
`tokens[min(i, len(tokens) - 1)]`. Once a rule's token list is exhausted, every later stage
holds its last value — this is not a bug to guard against, it is the file format. A rule
whose token list is **longer** than `n_stages` is a file error and must raise, not be
silently truncated.

**Rules are applied in file order; a later rule overwrites an earlier one** on the same
`(a, b)` pair at the same stage. In every converged-meshes folder measured so far the region
classes named by different rules are disjoint, so this never actually fires — but the parser
implements it as specified regardless of that being unexercised by the current data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def parse_region_range(spec: str) -> list[int]:
    """Parse a range specification such as `"2-23,40,55-57"` into a sorted list of region ids.

    Args:
        spec (str): Comma-separated list of integers and/or inclusive `a-b` spans.

    Returns:
        list[int]: Sorted, de-duplicated region ids.
    """
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-")
            ids.update(range(int(start_str), int(end_str) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


@dataclass(frozen=True)
class TensionRule:
    """One parsed `<rangeA>:<rangeB> <tokens>` line."""

    region_a: tuple[int, ...]
    region_b: tuple[int, ...]
    tokens: tuple[float, ...]
    line_no: int
    raw: str

    def value_at_stage(self, stage: int) -> float:
        """Token for `stage`, holding the last value once the token list is exhausted.

        Args:
            stage (int): 0-indexed stage number.

        Returns:
            float: `tokens[min(stage, len(tokens) - 1)]`.
        """
        return self.tokens[min(stage, len(self.tokens) - 1)]


@dataclass
class MultitensionsFile:
    """A fully parsed `multitensions.opt`, plus the resolved per-stage tension maps."""

    path: Path
    n_cells: int
    n_stages: int
    rules: list[TensionRule]
    tensions_by_stage: list[dict[tuple[int, int], float]]

    @property
    def n_regions(self) -> int:
        """Number of regions, `n_cells + 1` (region 0 is the external medium)."""
        return self.n_cells + 1


def parse_multitensions(path: str | Path) -> MultitensionsFile:
    """Parse a `multitensions.opt` file into per-stage `(min_label, max_label) -> tension` maps.

    Args:
        path (str | Path): Path to `multitensions.opt`.

    Returns:
        MultitensionsFile: Parsed rules and, for convenience, the fully resolved per-stage
            tension dictionaries (`tensions_by_stage[stage][(a, b)]`, `a < b`).

    Raises:
        ValueError: If the file is empty of content, or a rule's token list is longer than
            `n_stages` (a file error per the format's own stage-count declaration, not a
            case to silently truncate).
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    content = [(i, line) for i, line in enumerate(lines) if line.strip() and not line.strip().startswith("#")]
    if not content:
        message = f"{path}: no non-comment content"
        raise ValueError(message)

    _header_no, header_line = content[0]
    n_cells_str, n_stages_str = header_line.split()
    n_cells, n_stages = int(n_cells_str), int(n_stages_str)

    rules: list[TensionRule] = []
    for line_no, line in content[1:]:
        range_part, value_part = line.split(maxsplit=1)
        range_a_str, range_b_str = range_part.split(":")
        region_a = tuple(parse_region_range(range_a_str))
        region_b = tuple(parse_region_range(range_b_str))
        tokens = tuple(float(tok) for tok in value_part.strip().split("_"))
        if len(tokens) > n_stages:
            message = (
                f"{path}:{line_no + 1}: rule '{line.strip()}' has {len(tokens)} values for "
                f"only {n_stages} stages -- a token list longer than n_stages is a file "
                f"error, not a truncation"
            )
            raise ValueError(message)
        rules.append(TensionRule(region_a, region_b, tokens, line_no + 1, line.strip()))

    tensions_by_stage: list[dict[tuple[int, int], float]] = [{} for _ in range(n_stages)]
    for rule in rules:
        for stage in range(n_stages):
            value = rule.value_at_stage(stage)
            stage_map = tensions_by_stage[stage]
            for a in rule.region_a:
                for b in rule.region_b:
                    if a == b:
                        continue
                    stage_map[(min(a, b), max(a, b))] = value

    return MultitensionsFile(
        path=path,
        n_cells=n_cells,
        n_stages=n_stages,
        rules=rules,
        tensions_by_stage=tensions_by_stage,
    )


def stage_groups(parsed: MultitensionsFile) -> list[list[int]]:
    """Group stages that resolve to an **identical** tension map (the hold-last-value groups).

    Computed from the parser's own resolved output, per the converged-mesh-cohort work's
    instruction ("compute
    the groups from the parser, do not hardcode them") -- not from any assumption about which
    rules have short token lists.

    Args:
        parsed (MultitensionsFile): A parsed `multitensions.opt`.

    Returns:
        list[list[int]]: Groups of stage indices sharing one tension map, in stage order
            (e.g. `[[0], [1], ..., [10, 11, ..., 17]]`).
    """
    groups: list[list[int]] = []
    for stage, tensions in enumerate(parsed.tensions_by_stage):
        if groups and parsed.tensions_by_stage[groups[-1][-1]] == tensions:
            groups[-1].append(stage)
        else:
            groups.append([stage])
    return groups


def validate_coverage(
    tensions: dict[tuple[int, int], float],
    mesh_interfaces: list[tuple[int, int]],
) -> dict:
    """Check that every interface actually present in a mesh has a resolved tension.

    Per the converged-mesh-cohort work's instruction: "assert every interface actually present in the .rec has a
    tension, and report any interface no rule covers" -- an assertion the parser cannot make
    on its own, since it has no independent knowledge of which region pairs are geometrically
    adjacent in a given mesh.

    Args:
        tensions (dict[tuple[int, int], float]): Resolved `(min_label, max_label) -> tension`
            map for one stage (`MultitensionsFile.tensions_by_stage[stage]`).
        mesh_interfaces (list[tuple[int, int]]): Interfaces actually present in the mesh
            (`foambryo.validation.mesh_io.list_interfaces`'s output).

    Returns:
        dict: `{"covered", "uncovered", "n_mesh_interfaces", "n_covered", "n_uncovered"}`.
    """
    uncovered = sorted(key for key in mesh_interfaces if key not in tensions)
    covered = sorted(key for key in mesh_interfaces if key in tensions)
    return {
        "covered": covered,
        "uncovered": uncovered,
        "n_mesh_interfaces": len(mesh_interfaces),
        "n_covered": len(covered),
        "n_uncovered": len(uncovered),
    }
