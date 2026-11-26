"""metrics.compute()의 규칙을 가짜 데이터로 확인하는 테스트.

실행:  python -m pytest -q
가짜 데이터를 쓰는 이유: 진짜 CSV는 값이 바뀔 수 있지만, 규칙(예: 0.5%p 넘으면 공시 비율)은 항상 지켜져야 한다.
각 테스트는 '이런 입력이면 이런 결과가 나와야 한다'는 약속이다.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import THRESHOLD, compute  # noqa: E402


def make(rows):
    """rows: (코드, 연도, 매출(원), R&D(백만원), 공시비율%, 공정) 목록 -> compute()에 넣을 두 표."""
    fin = pd.DataFrame([
        {"stock_code": c, "name": f"회사{c}", "year": y, "process": p, "segment": "장비", "fs_div": "CFS", "revenue": rev}
        for c, y, rev, _, _, p in rows
    ])
    man = pd.DataFrame([
        {"stock_code": c, "year": y, "rnd_total_mil_won": rnd, "reported_ratio_pct": rep, "source_note": "test"}
        for c, y, _, rnd, rep, _ in rows
    ])
    return fin, man


def get(df, code, year):
    return df[(df["stock_code"] == code) & (df["year"] == year)].iloc[0]


def test_calculated_ratio():
    # R&D 100억(10,000백만원) / 매출 1,000억 = 10%
    df = compute(*make([("A", 2025, 100_000_000_000, 10_000, 10.0, "P")]))
    r = get(df, "A", 2025)
    assert r["calc_ratio"] == pytest.approx(10.0)
    assert r["rnd_ratio"] == pytest.approx(10.0)
    assert r["ratio_source"] == "계산(R&D/매출)"


def test_small_gap_keeps_calculated():
    # 공시 10.4% vs 계산 10.0% -> 차이 0.4%p는 기준(0.5) 이하라 계산값을 쓴다
    df = compute(*make([("A", 2025, 100_000_000_000, 10_000, 10.4, "P")]))
    assert get(df, "A", 2025)["ratio_source"] == "계산(R&D/매출)"


def test_exactly_threshold_keeps_calculated():
    # 정확히 0.5%p 차이는 '넘지 않음'이라 계산값을 쓴다 (규칙은 '초과')
    assert THRESHOLD == 0.5
    df = compute(*make([("A", 2025, 100_000_000_000, 10_000, 10.5, "P")]))
    assert get(df, "A", 2025)["ratio_source"] == "계산(R&D/매출)"


def test_large_gap_switches_to_reported():
    df = compute(*make([("A", 2025, 100_000_000_000, 10_000, 12.0, "P")]))
    r = get(df, "A", 2025)
    assert r["rnd_ratio"] == pytest.approx(12.0)
    assert r["ratio_source"] == "공시비율(기준상이)"


def test_one_bad_year_switches_all_years_of_that_company():
    # 2023만 차이가 크고 2024·2025는 일치해도, 같은 회사 3개년 모두 공시 비율을 써야 한다(추이 왜곡 방지)
    rows = [
        ("A", 2023, 100_000_000_000, 10_000, 12.0, "P"),
        ("A", 2024, 100_000_000_000, 10_000, 10.0, "P"),
        ("A", 2025, 100_000_000_000, 10_000, 10.0, "P"),
        ("B", 2023, 100_000_000_000, 10_000, 10.0, "Q"),   # 다른 회사는 영향을 받지 않아야 한다
    ]
    df = compute(*make(rows))
    assert (df[df["stock_code"] == "A"]["ratio_source"] == "공시비율(기준상이)").all()
    assert get(df, "B", 2023)["ratio_source"] == "계산(R&D/매출)"


def test_yoy_and_first_year_is_nan():
    rows = [
        ("A", 2024, 100_000_000_000, 10_000, 10.0, "P"),
        ("A", 2025, 200_000_000_000, 15_000, 7.5, "P"),
    ]
    df = compute(*make(rows))
    assert pd.isna(get(df, "A", 2024)["revenue_yoy"])
    assert get(df, "A", 2025)["revenue_yoy"] == pytest.approx(100.0)
    assert get(df, "A", 2025)["rnd_yoy"] == pytest.approx(50.0)


def test_yoy_does_not_cross_companies():
    # 회사 A의 2025와 회사 B의 2023이 이어 붙어 YoY가 계산되면 안 된다
    rows = [
        ("A", 2025, 100_000_000_000, 10_000, 10.0, "P"),
        ("B", 2023, 300_000_000_000, 30_000, 10.0, "Q"),
    ]
    df = compute(*make(rows))
    assert df["revenue_yoy"].isna().all()


def test_single_company_in_process_has_no_median_comparison():
    # 공정에 기업이 하나뿐이면 자기 자신이 중앙값이라 비교가 의미 없으므로 빈 값이어야 한다
    df = compute(*make([("A", 2025, 100_000_000_000, 10_000, 10.0, "혼자")]))
    assert pd.isna(get(df, "A", 2025)["vs_process"])


def test_process_median_comparison():
    rows = [
        ("A", 2025, 100_000_000_000, 5_000, 5.0, "P"),
        ("B", 2025, 100_000_000_000, 10_000, 10.0, "P"),
        ("C", 2025, 100_000_000_000, 30_000, 30.0, "P"),
    ]
    df = compute(*make(rows))
    assert get(df, "B", 2025)["process_median"] == pytest.approx(10.0)
    assert get(df, "A", 2025)["vs_process"] == pytest.approx(-5.0)
    assert get(df, "C", 2025)["vs_process"] == pytest.approx(20.0)


def test_missing_rnd_gives_nan_not_crash():
    # 수동 입력이 비어 있으면 오류로 멈추지 말고 빈 값으로 남아야 한다(validate.py가 잡아낸다)
    fin, man = make([("A", 2025, 100_000_000_000, 10_000, 10.0, "P")])
    man["rnd_total_mil_won"] = float("nan")
    df = compute(fin, man)
    assert pd.isna(get(df, "A", 2025)["rnd_ratio"])
