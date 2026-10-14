import numpy as np
from tools.fi1_sr_score_factorization import factorize


def test_selected_max_distinct_from_full_and_zero_retained():
    m=[dict(geometry_key='g')]*3+[dict(geometry_key='h')]*2
    s=np.array([.2,.1,.02,.1,.05]);r=np.array([.1,.05,.9,0,.8])
    p0,a=factorize(m,s,r,scope=[0,1,3]);p1,b=factorize(m,s,r)
    assert np.allclose(p0[[0,1,3]],[.1,.05,0])
    assert p1[0]==.9 and p1[3]==.8
    assert a['h']['R_max']==0 and not a['h']['fallback']
    assert np.isclose(p1[1]/p1[0],s[1]/s[0])


def test_large_multiplier_not_clipped_and_max_exact():
    m=[dict(geometry_key='g')]*2
    s=np.array([.001,.0002]);r=np.array([.8,.2]);out,_=factorize(m,s,r)
    assert out[0]==.8 and np.isclose(out[1],.16)


def test_full_zero_fallback_predeclared():
    m=[dict(geometry_key='g')]*2
    out,meta=factorize(m,[.2,.1],[0,0])
    assert np.array_equal(out,[.2,.1]) and meta['g']['fallback']
