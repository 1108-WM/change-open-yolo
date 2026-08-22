from tools.train_ncs_fi1_stage_d_v3_full_rank_marginal_oof import shared


def test_stage_d_v3_keeps_frozen_model_and_uses_v3_score_contract():
    assert shared.MODEL_PARAMS == {
        "learning_rate": 0.05,
        "max_iter": 160,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "early_stopping": False,
    }
    assert shared.SCORE_FIELD == "stage_d_v3_append_score"
    assert shared.PLAN_FILE == "stage_d_v3_append_score_plan.jsonl"
