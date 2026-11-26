"""고객사(삼성전자·SK하이닉스)의 설비투자와 장비사 5곳의 매출을 한 표로 모은다.

실행:  python analyze_capex.py   (먼저 python collect_data.py 를 실행해 data/raw/ 캐시가 있어야 한다)
결과:  data/processed/capex.csv   (연도별 한 줄: 고객사 설비투자, 장비사 매출, 전년 대비 증가율)

설비투자(CAPEX) = 현금흐름표의 '유형자산의 취득'. 공장·장비를 새로 사는 데 쓴 돈이다.
2021~2025 5개년을 쓴다. 사업보고서 한 권에 당기·전기·전전기 3개년이 들어 있어서
2023년 보고서(2021~2023)와 2025년 보고서(2023~2025)를 이어 붙인다.
"""
import json
import os
import xml.etree.ElementTree as ET

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "data", "raw")
OUT_PATH = os.path.join(BASE, "data", "processed", "capex.csv")

# 고객사: 종목코드 -> (corp_code, 이름)
CUSTOMERS = {"005930": ("00126380", "삼성전자"), "000660": ("00164779", "SK하이닉스")}
# 보고서 칸 이름: 당기 / 전기 / 전전기
FIELDS = ("thstrm_amount", "frmtrm_amount", "bfefrmtrm_amount")


def load(corp_code, year):
    with open(os.path.join(RAW_DIR, f"{corp_code}_{year}.json"), encoding="utf-8") as f:
        return json.load(f)["list"]


def to_int(text):
    return int(str(text).replace(",", "").strip())


def capex_by_year(corp_code):
    """보고서 2023, 2025의 '유형자산의 취득'을 이어 붙여 {연도: 금액(원)}을 만든다."""
    out = {}
    for report_year in (2023, 2025):  # 2025 보고서가 같은 연도를 다시 적으면 그 값을 쓴다 (나중에 덮어씀)
        rows = load(corp_code, report_year)
        row = next(r for r in rows
                   if r["sj_div"] == "CF"
                   and r.get("account_id") == "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities")
        for i, field in enumerate(FIELDS):
            out[report_year - i] = to_int(row[field])
    return out


def revenue_by_year(corp_code):
    """같은 방식으로 매출액을 {연도: 금액(원)}으로 만든다 (collect_data.py의 매출 찾는 규칙과 같다)."""
    out = {}
    for report_year in (2023, 2025):
        rows = [r for r in load(corp_code, report_year) if r["sj_div"] in ("IS", "CIS")]
        row = next((r for r in rows if r.get("account_id") == "ifrs-full_Revenue"), None)
        if row is None:
            row = next(r for r in rows if r["account_nm"].strip() in ("매출액", "수익(매출액)", "영업수익", "매출"))
        for i, field in enumerate(FIELDS):
            out[report_year - i] = to_int(row[field])
    return out


def main():
    master = pd.read_csv(os.path.join(BASE, "data", "processed", "master.csv"),
                         dtype={"stock_code": str}, encoding="utf-8-sig")
    corp_of = {}  # 종목코드 -> corp_code (파일 이름 규칙으로 찾는다)

    # 장비사: segment가 '장비'인 5곳. corp_code는 DART 매핑 파일에서 다시 찾는다.
    equip = master[master["segment"] == "장비"][["stock_code", "name"]].drop_duplicates()
    for it in ET.parse(os.path.join(RAW_DIR, "CORPCODE.xml")).getroot().iter("list"):
        sc = (it.findtext("stock_code") or "").strip()
        if sc in set(equip["stock_code"]):
            corp_of[sc] = it.findtext("corp_code")

    cust = {name: capex_by_year(cc) for _, (cc, name) in CUSTOMERS.items()}
    equip_rev = {row["name"]: revenue_by_year(corp_of[row["stock_code"]]) for _, row in equip.iterrows()}

    years = sorted(next(iter(cust.values())).keys())
    df = pd.DataFrame({"year": years})
    for name, d in cust.items():
        df[f"{name} 설비투자"] = [d[y] for y in years]
    df["고객사 설비투자 합계"] = df[[f"{n} 설비투자" for n in cust]].sum(axis=1)
    for name, d in equip_rev.items():
        df[f"{name} 매출"] = [d[y] for y in years]
    df["장비사 매출 합계"] = df[[f"{n} 매출" for n in equip_rev]].sum(axis=1)

    # 전년 대비 증가율(%). 합계뿐 아니라 고객사별로도 만든다
    # (삼성전자는 투자 규모가 크고 변동이 적어서 합계에서는 SK하이닉스의 출렁임이 묻힌다)
    yoy_cols = ["고객사 설비투자 합계", "장비사 매출 합계"] + [f"{n} 설비투자" for n in cust]
    for col in yoy_cols:
        df[col + " YoY"] = (df[col].pct_change() * 100).round(1)

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    show = pd.DataFrame({
        "연도": df["year"],
        "고객사 CAPEX(조)": (df["고객사 설비투자 합계"] / 1e12).round(1),
        "CAPEX YoY(%)": df["고객사 설비투자 합계 YoY"],
        "장비사 매출(억)": (df["장비사 매출 합계"] / 1e8).round(0),
        "매출 YoY(%)": df["장비사 매출 합계 YoY"],
    })
    print(show.to_string(index=False))
    both = df.dropna(subset=["고객사 설비투자 합계 YoY"])
    print(f"\n증가율끼리의 상관계수 (표본 {len(both)}개뿐이라 참고용)")
    for col in ["고객사 설비투자 합계"] + [f"{n} 설비투자" for n in cust]:
        corr = both[col + " YoY"].corr(both["장비사 매출 합계 YoY"])
        print(f"  {col} vs 장비사 매출: {corr:+.2f}")
    print(f"저장 위치: {OUT_PATH}")


if __name__ == "__main__":
    main()
