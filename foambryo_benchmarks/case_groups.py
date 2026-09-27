"""The `benchmarking-dataset` equilibrium split — one place, so no aggregate can mix it.

Recorded decision (H. Turlier, 2026-07-30), from the dataset audit
(`foambryo_benchmarks/run_dataset_audit.py`):

- **42 of the 47 cases are usable** as ground truth for accuracy claims, both on the inference
  methods and on the full segmentation-to-tension pipeline.
- **Five cases are not at mechanical equilibrium** — `005`, `015`, `018`, `021`, `043` — and
  are **excluded from every accuracy claim**. They stay in the repository as a
  documented *negative control*: a method that cannot tell them apart from the healthy 42 is
  telling us something, and deleting them would throw that away.
- **`004` is kept** in the equilibrium group, with its integrity flag (2 malformed triple-line
  edges of 539; `rho = 0.073`, unremarkable). Its flag travels with it via
  `INTEGRITY_FLAGGED_CASE_IDS` so it can be excluded from a specific check on purpose rather
  than by accident.

Why the five are excluded, in one line each (three independent measures from the audit):
`rho_free ~= rho` (no surface tensions balance them at all, so it is not a wrong ground-truth
tension value); 84-95 % of the squared residual sits on **interface interiors** rather than at
the triple lines, with a per-vertex amplitude 8-80x the healthy cases'; and `sigma_min` of the
column-normalised operator is 0.073-0.094 where the 42 give 0.0002-0.0116, i.e. their operator
has no near-null direction where the global-scale gauge must produce one.

Honest caveat, also from the audit and deliberately not hidden here: on this dataset the five cases'
*tension-inference error* is indistinguishable from size-matched healthy cases for all eight
working methods. The exclusion is therefore conservative — justified by the data defect being
real and reproducible, not by a demonstrated harm to inference. `ForceBalance`
minimises exactly the residual those five inflate, which is the one place it is most likely to
bite.

**Why this module exists rather than a comment in each driver.** The failure mode is silent: a
mean or a median over "all 47" reads as an accuracy number and is in fact 89 % accuracy and
11 % data defect, with no marker in the output. `require_single_group` makes that structural —
an aggregation helper that is handed a mixed list of cases raises instead of returning a
number.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: Cases the dataset-audit work found not to be at mechanical equilibrium. Excluded from accuracy claims.
NON_EQUILIBRIUM_CASE_IDS: frozenset[str] = frozenset({"005", "015", "018", "021", "043"})

#: Cases that fail the audit's integrity check but are kept (see the module docstring on `004`).
INTEGRITY_FLAGGED_CASE_IDS: frozenset[str] = frozenset({"004"})

EQUILIBRIUM = "equilibrium"
NON_EQUILIBRIUM = "non_equilibrium"

#: Group order used in every report, so tables always read "the 42, then the 5".
GROUP_ORDER: tuple[str, str] = (EQUILIBRIUM, NON_EQUILIBRIUM)

#: The `rho` and interior-residual-fraction thresholds that separated the two populations in
#: the audit. They are *descriptive of that measurement*, not a general criterion: `rho`'s floor is
#: discretisation (0.0007-0.036 on the 42), so there is no absolute cutoff between "at
#: equilibrium" and "not". Used only by `tests/test_case_groups.py`, to check that the
#: hard-coded id list above still matches the committed audit records.
B1_RHO_THRESHOLD = 0.20
B1_INTERIOR_FRACTION_THRESHOLD = 0.5


def group_of(case_id: str) -> str:
    """Return the group a case belongs to.

    Args:
        case_id (str): Dataset case id, e.g. `"005"`. Compared as a string, zero-padded as
            the dataset names its files.

    Returns:
        str: `EQUILIBRIUM` or `NON_EQUILIBRIUM`.
    """
    return NON_EQUILIBRIUM if str(case_id) in NON_EQUILIBRIUM_CASE_IDS else EQUILIBRIUM


def split_case_ids(case_ids: Iterable[str]) -> dict[str, list[str]]:
    """Partition case ids into the two groups, preserving input order within each.

    Args:
        case_ids (Iterable[str]): Case ids.

    Returns:
        dict[str, list[str]]: `{EQUILIBRIUM: [...], NON_EQUILIBRIUM: [...]}`, both keys always
            present (possibly empty).
    """
    groups: dict[str, list[str]] = {group: [] for group in GROUP_ORDER}
    for case_id in case_ids:
        groups[group_of(case_id)].append(str(case_id))
    return groups


def split_records(records: Sequence[dict], case_key: str = "case") -> dict[str, list[dict]]:
    """Partition per-case records into the two groups.

    Args:
        records (Sequence[dict]): Records each carrying a case id under `case_key`.
        case_key (str, optional): Key holding the case id. Defaults to `"case"`.

    Returns:
        dict[str, list[dict]]: `{EQUILIBRIUM: [...], NON_EQUILIBRIUM: [...]}`.
    """
    groups: dict[str, list[dict]] = {group: [] for group in GROUP_ORDER}
    for record in records:
        groups[group_of(record[case_key])].append(record)
    return groups


def require_single_group(records: Sequence[dict], case_key: str = "case") -> str:
    """Assert that `records` are all from one group, and return that group's name.

    Call this at the top of any function that reduces several cases to one number. It is what
    stops an aggregate from silently mixing the 42 usable cases with the 5 non-equilibrium
    ones, which would report a data defect as a method's accuracy.

    Args:
        records (Sequence[dict]): Per-case records.
        case_key (str, optional): Key holding the case id. Defaults to `"case"`.

    Returns:
        str: The single group the records belong to.

    Raises:
        ValueError: If `records` spans both groups, or is empty.
    """
    if not records:
        message = "cannot aggregate an empty record list: the group is undefined"
        raise ValueError(message)

    present = {group_of(record[case_key]) for record in records}
    if len(present) > 1:
        offenders = sorted(record[case_key] for record in records if group_of(record[case_key]) == NON_EQUILIBRIUM)
        message = (
            "refusing to aggregate across the equilibrium split: these cases are not at "
            f"mechanical equilibrium and must be reported separately -> {offenders}. "
            "Use foambryo_benchmarks.case_groups.split_records first (see this module's docstring "
            "for the recorded decision to exclude the five non-equilibrium cases)."
        )
        raise ValueError(message)
    return present.pop()


def group_note(group: str) -> str:
    """One-line provenance note to embed next to any per-group number in a report.

    Args:
        group (str): `EQUILIBRIUM` or `NON_EQUILIBRIUM`.

    Returns:
        str: Human-readable note.
    """
    if group == NON_EQUILIBRIUM:
        return (
            f"{len(NON_EQUILIBRIUM_CASE_IDS)} cases NOT at mechanical equilibrium "
            f"({', '.join(sorted(NON_EQUILIBRIUM_CASE_IDS))}) — negative control, "
            "excluded from every accuracy claim by recorded decision"
        )
    return (
        "cases at mechanical equilibrium — the usable dataset for accuracy claims (the five "
        "non-equilibrium cases are excluded by recorded decision); "
        f"includes {', '.join(sorted(INTEGRITY_FLAGGED_CASE_IDS))}, kept with its integrity flag"
    )
