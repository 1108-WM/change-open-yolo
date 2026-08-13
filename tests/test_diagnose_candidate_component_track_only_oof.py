import numpy as np

from tools.diagnose_candidate_component_track_only_oof import top_precision


def test_top_precision_uses_highest_scores_and_reports_recall():
    row = top_precision(
        np.asarray([0.1, 0.9, 0.8, 0.2]),
        np.asarray([0, 1, 0, 1]),
        0.5,
    )
    assert row["selected_count"] == 2
    assert row["positive_count"] == 1
    assert row["precision"] == 0.5
    assert row["positive_recall"] == 0.5
