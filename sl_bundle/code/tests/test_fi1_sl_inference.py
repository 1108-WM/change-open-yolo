import numpy as np
import torch
from tools import fi1_sl_inference as sl
from tools import run_fi1_s_semantic_rank_residual_development200 as original


def test_selected_inference_matches_original_full_batches_and_order():
    torch.set_num_threads(4);torch.manual_seed(91)
    model=sl.core.Residual()
    # Nonzero trained-like final layer; zero-init would hide batching errors.
    with torch.no_grad():model.head.weight.normal_(0,.1)
    rng=np.random.default_rng(31);x=rng.normal(size=(17001,28)).astype(np.float32)
    mean=x.mean(0);std=x.std(0);s=rng.uniform(.01,.99,len(x));s[0]=0;s[-1]=1
    indices=np.array([17000,0,8191,8192,16383,16384,17])
    scores,delta=sl.score_selected(model,x,s,indices,mean,std)
    full=original.infer(model,x,mean,std)
    assert np.array_equal(delta,full[indices])
    assert np.array_equal(scores,sl.core.adjusted(s,full)[indices])
    assert scores[0]==1 and scores[1]==0
