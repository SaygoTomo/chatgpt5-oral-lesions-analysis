import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0,str(SCRIPTS))
from analyze import prepare_data, icc_a1_ak, holm_adjust, cronbach_alpha
from agreement import linear_weighted_kappa
from biopsy_sensitivity import wilson


def test_i4_recode_and_aipi_total():
    d = pd.DataFrame([dict(case_id='C1',repetition=1,I1=2,I2=2,I3=2,I4=1,I5=3,I6=1,I7=3,I8=1,I9=3)])
    a,_,n=prepare_data(d,'case_id','repetition',False)
    assert n==1 and a.I4.iloc[0]==3 and a.aipi_total.iloc[0]==20
    b,_,_=prepare_data(a,'case_id','repetition',True)
    assert b.I4.iloc[0]==3


def test_out_of_range_score_not_treated_as_zero():
    d=pd.DataFrame([dict(case_id='C1',repetition=1,I1=2,I2=2,I3=2,I4=3,I5=3,I6=4,I7=3,I8=1,I9=3)])
    a,q,_=prepare_data(d,'case_id','repetition',True)
    assert len(q)==1 and np.isnan(a.diagnosis.iloc[0]) and np.isnan(a.aipi_total.iloc[0])


def test_icc_perfect_agreement():
    a,b=icc_a1_ak(np.array([[1,1,1],[2,2,2],[3,3,3]],dtype=float))
    assert a==pytest.approx(1) and b==pytest.approx(1)


def test_linear_kappa_and_holm():
    assert linear_weighted_kappa([1,2,3],[1,2,3],0,3)==pytest.approx(1)
    assert list(holm_adjust([0.01,0.02,0.5]))==pytest.approx([.03,.04,.5])


def test_wilson():
    lo,hi=wilson(103,184)
    assert 0.487<lo<0.489 and 0.629<hi<0.631


def test_cronbach_perfect_items():
    d=pd.DataFrame({'I1':[1,2,3,4],'I2':[1,2,3,4],'I3':[1,2,3,4]})
    alpha,n=cronbach_alpha(d)
    assert n==4 and alpha==pytest.approx(1)


def test_accs_component_sum_overrides_cached_total():
    from prepare_input import derive_accs_total
    assert derive_accs_total([1, 1, 1, 1, 2, 1]) == 7
    assert derive_accs_total([1, 1, 1, 1, 2, 2]) == 8
    with pytest.raises(ValueError):
        derive_accs_total([1, 1, 1, 0, 2, 1])
