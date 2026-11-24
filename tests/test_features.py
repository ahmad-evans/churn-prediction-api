"""Unit tests for feature engineering, the code shared by training and serving."""
import pandas as pd

from app.features import CATEGORICAL, NUMERIC, add_features
from tests.conftest import HIGH_RISK, LOW_RISK


def frame(*rows):
    return pd.DataFrame([{k: v for k, v in r.items() if k != "customer_id"} for r in rows])


def test_output_columns_match_model_inputs():
    out = add_features(frame(HIGH_RISK))
    assert list(out.columns) == NUMERIC + CATEGORICAL


def test_addon_count():
    out = add_features(frame(HIGH_RISK, LOW_RISK))
    assert out["NumAddons"].tolist() == [0, 4]


def test_autopay_flag():
    out = add_features(frame(HIGH_RISK, LOW_RISK))
    assert out["AutoPay"].tolist() == [0, 1]


def test_new_customer_flag():
    out = add_features(frame(HIGH_RISK, LOW_RISK))
    assert out["NewCustomer"].tolist() == [1, 0]


def test_avg_spend_handles_zero_tenure():
    row = dict(HIGH_RISK, tenure=0, TotalCharges=0)
    out = add_features(frame(row))
    assert out["AvgMonthlySpend"].iloc[0] == row["MonthlyCharges"]


def test_blank_total_charges_become_zero():
    """The raw CSV stores TotalCharges as text, with blanks for new customers."""
    row = dict(HIGH_RISK, tenure=0, TotalCharges=" ")
    out = add_features(frame(row))
    assert out["TotalCharges"].iloc[0] == 0.0


def test_does_not_modify_input():
    df = frame(HIGH_RISK)
    before = df.copy()
    add_features(df)
    pd.testing.assert_frame_equal(df, before)
