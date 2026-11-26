"""validate.py가 '틀린 데이터를 정말 잡아내는지'와 '진짜 master.csv가 통과하는지'를 확인한다.

검사 코드는 틀린 데이터를 넣어 봐야 믿을 수 있다. (항상 PASS만 나오는 검사는 아무것도 보장하지 않는다.)
"""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import validate  # noqa: E402


@pytest.fixture(scope="module")
def real():
    df = pd.read_csv(validate.MASTER_PATH, dtype={"stock_code": str}, encoding="utf-8-sig")
    companies = pd.read_csv(validate.COMPANIES_PATH, dtype={"stock_code": str})
    return df, companies


def failed(results):
    return {name for status, name, _ in results if status == "FAIL"}


def test_real_master_passes_all_checks(real):
    df, companies = real
    assert failed(validate.run_checks(df, companies)) == set()


def test_catches_duplicate_row(real):
    df, companies = real
    bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert "(기업, 연도) 중복이 없다" in failed(validate.run_checks(bad, companies))


def test_catches_missing_year(real):
    df, companies = real
    bad = df.drop(df.index[0])
    assert any("연도를 갖췄다" in n for n in failed(validate.run_checks(bad, companies)))


def test_catches_missing_value(real):
    df, companies = real
    bad = df.copy()
    bad.loc[0, "rnd_total_mil_won"] = float("nan")
    assert "핵심 열에 빈 값이 없다" in failed(validate.run_checks(bad, companies))


def test_catches_wrong_ratio(real):
    df, companies = real
    bad = df.copy()
    bad.loc[0, "calc_ratio"] = bad.loc[0, "calc_ratio"] + 5   # 계산값을 일부러 틀리게
    assert "calc_ratio가 R&D÷매출로 다시 계산한 값과 같다" in failed(validate.run_checks(bad, companies))


def test_catches_unit_error(real):
    df, companies = real
    bad = df.copy()
    bad.loc[0, "rnd_ratio"] = 250.0   # 단위를 잘못 입력해 비중이 비정상적으로 큰 경우
    assert any("안에 있다" in n for n in failed(validate.run_checks(bad, companies)))


def test_catches_mixed_methods_within_company(real):
    df, companies = real
    bad = df.copy()
    bad.loc[bad["name"] == bad.loc[0, "name"], "ratio_source"] = ["공시비율(기준상이)", "계산(R&D/매출)", "계산(R&D/매출)"]
    assert "한 회사 안에서 산출 방식이 섞이지 않는다" in failed(validate.run_checks(bad, companies))
