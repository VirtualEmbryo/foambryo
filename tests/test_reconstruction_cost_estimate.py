"""The reconstruction's estimate, made in one place for every front end.

What this pins, and why: the napari plugin and `foambryo view MASK` both tell a user, before a reconstruction runs,
how long it should take and how much memory it should need. The constants used to live in each front end; now they
live here and the front ends read them. So:

* the estimate reproduces each of the seven measured reconstructions it was fitted to within its band (the band is
  the largest in-sample ratio, so this holds by construction and fails if a constant is mistyped);
* the constants are the fit's, digit for digit (the values the plugin shipped);
* an anisotropic mask counts its voxels after resampling to cubic voxels, and a tiny estimate does not read "0 s";
* a band that adds nothing is not written: a tissue that takes under a second reads "under 1 s", not "about under
  1 s (under 1 s-under 1 s)" as it did up to 0.7.0, and the seed-ensemble status of the window never reads "about
  under 1 s"; the text of every other estimate is unchanged.
"""

from __future__ import annotations

import pytest

from foambryo.reconstruction_cost import (
    RECONSTRUCTION_COST_ANCHORS,
    RECONSTRUCTION_COST_FIT,
    estimate_reconstruction_cost,
    voxels_handed_to_dw3d,
)


@pytest.mark.parametrize("anchor", RECONSTRUCTION_COST_ANCHORS, ids=[a[0] for a in RECONSTRUCTION_COST_ANCHORS])
def test_the_estimate_holds_every_anchor_within_its_band(anchor):
    _, voxels, seconds, peak_gib = anchor
    estimate = estimate_reconstruction_cost(voxels)
    assert estimate.seconds[0] * 0.999 <= seconds <= estimate.seconds[2] * 1.001
    assert estimate.peak_gib[0] * 0.999 <= peak_gib <= estimate.peak_gib[2] * 1.001


def test_the_constants_are_the_fits():
    assert RECONSTRUCTION_COST_FIT == {
        "seconds": {"prefactor": 3.6358636769611024e-08, "exponent": 1.1567209231530782,
                    "band_factor": 1.5793137370008161},
        "peak_gib": {"prefactor": 2.1422186523855385e-05, "exponent": 0.7021459857892676,
                     "band_factor": 1.4860558062942626},
    }
    estimate = estimate_reconstruction_cost(8.3e8)
    assert estimate.peak_gib[1] == 2.1422186523855385e-05 * 8.3e8 ** 0.7021459857892676


def test_voxels_after_resampling_and_the_text():
    assert voxels_handed_to_dw3d((10, 10, 10), (2.0, 1.0, 1.0)) == 2000.0
    assert voxels_handed_to_dw3d((3, 10, 10, 10), (1.0, 1.0, 1.0)) == 1000.0
    text = estimate_reconstruction_cost(8.3e8).text()
    assert "min" in text
    assert "GiB" in text
    assert "\u2013" in estimate_reconstruction_cost(8.3e8).text(dash="\u2013")
    assert "0 s" not in estimate_reconstruction_cost(1000.0).text()


@pytest.mark.parametrize("voxels", [10.0, 1e3, 1e5, 1.5e6, 1.8e6], ids=["10", "1e3", "1e5", "1.5e6", "1.8e6"])
@pytest.mark.parametrize("dash", ["-", "\u2013"], ids=["hyphen", "en dash"])
def test_a_time_under_a_second_is_said_once(voxels, dash):
    text = estimate_reconstruction_cost(voxels).text(dash=dash)
    assert text.startswith("under 1 s and "), text
    assert "about under" not in text
    assert text.count("under 1 s") == 1


def test_the_tiniest_estimate_is_one_phrase_per_quantity():
    assert estimate_reconstruction_cost(10.0).text() == "under 1 s and under 1 MiB"


def test_a_band_reaching_past_a_second_is_said_as_its_high_end():
    estimate = estimate_reconstruction_cost(2.5e6)
    assert estimate.seconds[1] < 1 <= estimate.seconds[2]
    assert estimate.text() == "up to 1 s and 681 MiB (459 MiB-1013 MiB)"


@pytest.mark.parametrize(
    ("voxels", "text"),
    [
        (3e6, "about 1 s (under 1 s-2 s) and 775 MiB (521 MiB-1 GiB)"),
        (7.72e6, "about 3 s (2 s-5 s) and 1 GiB (1012 MiB-2 GiB)"),
        (8.3e8, "about 13 min (8 min-20 min) and 39 GiB (26 GiB-58 GiB)"),
    ],
)
def test_every_other_estimate_reads_as_before(voxels, text):
    assert estimate_reconstruction_cost(voxels).text() == text


def test_a_central_time_alone_is_never_about_under_a_second():
    from foambryo.reconstruction_cost import _about_duration  # new in 0.7.0: imported here so older code collects

    assert _about_duration(0.3) == "under 1 s"
    assert _about_duration(200.0) == "about 3 min"
