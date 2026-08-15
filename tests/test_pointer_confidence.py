from scripts.analyze_pointer_confidence import binary_auc, summarize_signal


def test_confidence_summary_recognizes_perfectly_separated_scores() -> None:
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [True, True, False, False]

    summary = summarize_signal(scores, labels)

    assert summary["auc"] == 1.0
    assert summary["safe_false_answer_rate"] <= 0.02
    assert summary["safe_answer_coverage"] == 1.0


def test_auc_counts_ties_as_half_a_win() -> None:
    assert binary_auc([0.5, 0.5], [True, False]) == 0.5
