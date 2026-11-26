"""고객사 설비투자와 장비사 매출을 '분기' 단위로 모은다 (연 단위는 표본이 4개뿐이라 늘리기 위함).

실행:  python collect_quarterly.py
결과:  data/processed/quarterly.csv   (분기별 한 줄, 단위: 원)
       응답은 data/raw/q_*.json 에 저장되어 다시 실행해도 API를 덜 부른다.

DART 분기보고서는 '누적' 값을 준다.
  - 1분기 보고서: 1~3월 / 반기 보고서: 1~6월 / 3분기 보고서: 1~9월 / 사업보고서: 1~12월
그래서 분기 하나의 값은 '이번 누적 - 직전 누적'으로 구한다.
  예) 3분기 = (1~9월 누적) - (1~6월 누적),  4분기 = (1~12월 누적) - (1~9월 누적)
2021년 값은 2022년 보고서에 적힌 '전기(작년 같은 기간)' 칸에서 가져온다.
"""
import json
import os
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("DART_API_KEY")

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "data", "raw")
OUT_PATH = os.path.join(BASE, "data", "processed", "quarterly.csv")

# 보고서 종류 코드 -> 몇 번째 분기까지의 누적인지 (4 = 1년 전체)
REPORTS = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}
YEARS = [2022, 2023, 2024, 2025, 2026]  # 2026년은 3분기 보고서까지 나와 있다 (사업보고서는 아직 없음)

CAPEX_ID = "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"
REVENUE_NAMES = ("매출액", "수익(매출액)", "영업수익", "매출")


def corp_codes(stock_codes):
    """종목코드 -> (corp_code, 이름). DART가 준 전체 목록(CORPCODE.xml)에서 찾는다."""
    found = {}
    for item in ET.parse(os.path.join(RAW_DIR, "CORPCODE.xml")).getroot().iter("list"):
        code = (item.findtext("stock_code") or "").strip()
        if code in stock_codes:
            found[code] = (item.findtext("corp_code"), item.findtext("corp_name"))
    return found


def fetch(corp_code, year, rc):
    """재무제표 한 건을 가져온다 (연결 CFS 먼저, 없으면 별도 OFS). 받은 응답은 파일로 저장한다."""
    # 연 1회 사업보고서는 collect_data.py가 이미 받아 둔 파일이 있으면 그대로 쓴다
    old = os.path.join(RAW_DIR, f"{corp_code}_{year}.json")
    if rc == "11011" and os.path.exists(old):
        with open(old, encoding="utf-8") as f:
            return json.load(f)
    cache = os.path.join(RAW_DIR, f"q_{corp_code}_{year}_{rc}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)

    data = {"status": "999"}
    for fs_div in ("CFS", "OFS"):
        params = {"crtfc_key": KEY, "corp_code": corp_code, "bsns_year": str(year),
                  "reprt_code": rc, "fs_div": fs_div}
        data = requests.get("https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
                            params=params, timeout=30).json()
        time.sleep(0.3)  # 너무 빠르게 부르지 않기
        if data["status"] == "000":
            break
    if data["status"] in ("000", "013"):  # 정상 또는 '데이터 없음'만 저장 (한도 초과 등은 다음에 재시도)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    return data


def to_int(text):
    try:
        return int(str(text).replace(",", "").strip())
    except ValueError:
        return None


def pick_revenue(rows, current, rc):
    """매출 누적값. current=True면 올해 칸, False면 '전기(작년 같은 기간)' 칸."""
    income = [r for r in rows if r["sj_div"] in ("IS", "CIS")]
    row = next((r for r in income if r.get("account_id") == "ifrs-full_Revenue"), None)
    if row is None:
        row = next((r for r in income if r["account_nm"].strip() in REVENUE_NAMES), None)
    if row is None:
        return None
    if rc == "11011":  # 사업보고서는 1년치 값 하나뿐
        return to_int(row["thstrm_amount" if current else "frmtrm_amount"])
    # 분기 보고서: ~add_amount 가 누적, 없으면(1분기) 3개월 값이 곧 누적
    add, plain = ("thstrm_add_amount", "thstrm_amount") if current else ("frmtrm_add_amount", "frmtrm_amount")
    return to_int(row.get(add)) if row.get(add) else to_int(row.get(plain))


def pick_capex(rows, current):
    """설비투자 누적값(현금흐름표 '유형자산의 취득'). 현금흐름표는 원래 누적으로만 적혀 있다."""
    row = next((r for r in rows if r["sj_div"] == "CF" and r.get("account_id") == CAPEX_ID), None)
    if row is None:
        return None
    return to_int(row["thstrm_amount" if current else "frmtrm_amount"])


def main():
    if not KEY:
        raise SystemExit(".env 파일에 DART_API_KEY가 없습니다.")

    companies = pd.read_csv(os.path.join(BASE, "companies.csv"), dtype={"stock_code": str}, encoding="utf-8-sig")
    targets = companies[companies["segment"].isin(["장비", "고객사"])]
    corp = corp_codes(set(targets["stock_code"]))

    # cum[(회사, 연도, 분기번호)] = {"rev": 누적매출, "capex": 누적설비투자}
    cum = {}
    for _, c in targets.iterrows():
        cc = corp[c["stock_code"]][0]
        for year in YEARS:
            for rc, qn in REPORTS.items():
                data = fetch(cc, year, rc)
                if data["status"] != "000":
                    print(f"  (없음) {c['name']} {year} 보고서 {rc}: status {data['status']}")
                    continue
                rows = data["list"]
                cum[(c["name"], year, qn)] = {"rev": pick_revenue(rows, True, rc), "capex": pick_capex(rows, True)}
                if year == YEARS[0]:  # 2022년 보고서의 '전기' 칸이 곧 2021년 값
                    cum[(c["name"], year - 1, qn)] = {"rev": pick_revenue(rows, False, rc), "capex": pick_capex(rows, False)}
        print(f"{c['name']}: 수집 완료")

    # ---- 누적 -> 분기 값 (이번 누적 - 직전 누적) ----
    out = []
    for name in targets["name"]:
        for year in [YEARS[0] - 1] + YEARS:
            for q in (1, 2, 3, 4):
                now = cum.get((name, year, q))
                prev = cum.get((name, year, q - 1)) if q > 1 else {"rev": 0, "capex": 0}
                row = {"name": name, "year": year, "q": q, "quarter": f"{year}Q{q}", "revenue": None, "capex": None}
                if now and prev:
                    for key, col in (("rev", "revenue"), ("capex", "capex")):
                        if now[key] is not None and prev[key] is not None:
                            row[col] = now[key] - prev[key]
                out.append(row)
    long = pd.DataFrame(out)

    # ---- 회사별 열로 펼치고 합계 만들기 ----
    equip = list(targets.loc[targets["segment"] == "장비", "name"])
    cust = list(targets.loc[targets["segment"] == "고객사", "name"])
    wide = long[["year", "q", "quarter"]].drop_duplicates().reset_index(drop=True)
    for name in equip:
        s = long[long["name"] == name].set_index("quarter")["revenue"]
        wide[f"{name} 매출"] = wide["quarter"].map(s)
    for name in cust:
        s = long[long["name"] == name].set_index("quarter")["capex"]
        wide[f"{name} 설비투자"] = wide["quarter"].map(s)
    # 한 곳이라도 비어 있으면 합계도 비운다 (일부만 더하면 합계가 왜곡되므로)
    wide["장비사 매출 합계"] = wide[[f"{n} 매출" for n in equip]].sum(axis=1, min_count=len(equip))
    wide["고객사 설비투자 합계"] = wide[[f"{n} 설비투자" for n in cust]].sum(axis=1, min_count=len(cust))
    wide = wide.dropna(subset=["장비사 매출 합계", "고객사 설비투자 합계"], how="all")
    wide.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    show = wide[["quarter"]].copy()
    show["장비사 매출(억)"] = (wide["장비사 매출 합계"] / 1e8).round(0)
    for n in cust:
        show[f"{n}(조)"] = (wide[f"{n} 설비투자"] / 1e12).round(2)
    print("\n=== 분기별 요약 ===")
    print(show.to_string(index=False))
    print(f"\n저장 위치: {OUT_PATH}  ({len(wide)}개 분기)")


if __name__ == "__main__":
    main()
