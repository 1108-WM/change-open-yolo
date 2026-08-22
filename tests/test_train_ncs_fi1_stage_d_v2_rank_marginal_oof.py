from tools.train_ncs_fi1_stage_d_v2_rank_marginal_oof import MODEL_PARAMS


def test_stage_d_v2_model_contract_is_frozen_from_v1():
    assert MODEL_PARAMS == {
        "learning_rate": 0.05,
        "max_iter": 160,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "early_stopping": False,
    }
