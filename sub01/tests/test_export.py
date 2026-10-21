"""CSV outputs: column names and the absence of a flare-rate column."""
from fakes import fake_analyzer
from flare_cnn.export import FLARE_COLUMNS, flares_table, star_summary


def test_flares_table_columns_and_ids():
    df = flares_table(fake_analyzer(None, 123, {}, None, None, 0.6))
    assert list(df.columns) == FLARE_COLUMNS
    assert df["flare_id"].tolist() == ["C1_123_S27_001", "C1_123_S27_002"]
    assert df["gaia_dr3_id"].iloc[0] == "Gaia DR3 5853498713190525696"     # kept as text


def test_star_summary_counts_without_rate():
    s = star_summary(fake_analyzer(None, 123, {}, None, None, 0.6))
    assert (s["n_flares"], s["n_weak"], s["n_M"], s["n_C"], s["strongest_tier"]) == (2, 1, 1, 1, "M")
    assert "flare_rate_per_day" not in s
