"""기업 12개 x 3개년 재무 데이터를 DART에서 수집해 CSV로 저장한다.

실행:  python collect_data.py
결과:  data/processed/financials.csv
       (한 번 받은 응답은 data/raw/ 에 저장되어 다시 실행해도 API를 덜 부른다)
"""
import io
import json
import os
import time
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("DART_API_KEY")

YEARS = [2023, 2024, 2025]
BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "data", "raw")
OUT_DIR = os.path.join(BASE, "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 1. 종목코드 -> corp_code 지도 만들기 ----------
def load_corp_map():
    path = os.path.join(RAW_DIR, "CORPCODE.xml")
    if not os.path.exists(path):
        res = requests.get(
            "https://opendart.fss.or.kr/api/corpCode.xml",
            params={"crtfc_key": KEY},
            timeout=60,  # 서버가 응답하지 않아도 무한정 기다리지 않게 한다
        )
        res.raise_for_status()
        # 키가 잘못되면 zip 대신 오류 메시지(JSON/XML)가 온다. 그 경우 알아보기 쉬운 안내를 내고 멈춘다.
        if not zipfile.is_zipfile(io.BytesIO(res.content)):
            raise SystemExit("corpCode.xml 을 받지 못했습니다 (zip 파일이 아님). API 키 또는 사용량 한도를 확인하세요: "
                             + res.text[:200])
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            with open(path, "wb") as f:
                f.write(z.read("CORPCODE.xml"))
    root = ET.parse(path).getroot()
    corp_map = {}
    for item in root.iter("list"):
        code = (item.findtext("stock_code") or "").strip()
        if code:
            corp_map[code] = (item.findtext("corp_code"), item.findtext("corp_name"))
    return corp_map


# ---------- 2. 재무제표 가져오기 (연결 CFS -> 없으면 별도 OFS) ----------
def fetch_financials(corp_code, year):
    cache = os.path.join(RAW_DIR, f"{corp_code}_{year}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)

    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    data = {"status": "999", "message": "not called"}
    for fs_div in ("CFS", "OFS"):
        params = {
            "crtfc_key": KEY,
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": "11011",
            "fs_div": fs_div,
        }
        data = requests.get(url, params=params, timeout=30).json()
        time.sleep(0.3)  # 너무 빠르게 부르지 않기
        if data["status"] == "000":
            data["fs_div_used"] = fs_div
            break

    # 정상(000) 또는 '데이터 없음'(013)일 때만 저장. 그 외(한도 초과 등)는 다음에 재시도.
    if data["status"] in ("000", "013"):
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    return data


# ---------- 3. 필요한 값 꺼내기 ----------
def to_int(text):
    try:
        return int(str(text).replace(",", "").strip())
    except ValueError:
        return None


REVENUE_NAMES = ("매출액", "수익(매출액)", "영업수익", "매출")


def extract_revenue(rows, field="thstrm_amount"):
    # field: thstrm_amount=당기, frmtrm_amount=전기, bfefrmtrm_amount=전전기
    income = [r for r in rows if r["sj_div"] in ("IS", "CIS")]
    for r in income:  # 1순위: 표준 계정ID
        if r.get("account_id") == "ifrs-full_Revenue":
            return to_int(r.get(field))
    for r in income:  # 2순위: 계정명
        if r["account_nm"].strip() in REVENUE_NAMES:
            return to_int(r.get(field))
    return None


def extract_rnd(rows):
    """재무제표 본표에서 '연구개발'이 들어간 항목을 찾는다. 없으면 (None, None)."""
    for r in rows:
        if "연구개발" in r["account_nm"] and r["sj_div"] in ("IS", "CIS"):
            return to_int(r["thstrm_amount"]), r["account_nm"].strip()
    return None, None


# ---------- 4. 전체 실행 ----------
def main():
    if not KEY:
        raise SystemExit(".env 파일에 DART_API_KEY가 없습니다.")

    companies = pd.read_csv(
        os.path.join(BASE, "companies.csv"), dtype={"stock_code": str}, encoding="utf-8-sig"
    )
    corp_map = load_corp_map()
    results = []

    for _, c in companies.iterrows():
        code, name = c["stock_code"], c["name"]
        if code not in corp_map:
            print(f"[경고] {name}({code}) 종목코드를 DART에서 찾지 못했습니다.")
            continue
        corp_code, dart_name = corp_map[code]
        if dart_name != name:
            print(f"[확인] 입력 이름 '{name}' / DART 이름 '{dart_name}' -> 같은 회사인지 확인하세요")

        # 가장 최근 보고서(2025)에는 전기·전전기 매출이 '재작성된 값'으로 들어 있다.
        # (동진쎄미켐은 중단영업 재분류로 과거 매출이 약 3,000억 줄어 다시 적혔다.)
        # 그래서 3개년 매출을 모두 최근 보고서 기준으로 맞춘다. R&D를 2025년 보고서 한 권에서
        # 읽은 것과 같은 기준이다.
        latest = fetch_financials(corp_code, YEARS[-1])
        latest_ok = latest["status"] == "000"

        for year in YEARS:
            data = fetch_financials(corp_code, year)
            if data["status"] != "000":
                print(f"{name} {year}: 데이터 없음 (status {data['status']})")
                results.append({"stock_code": code, "name": name, "year": year})
                continue
            rows = data["list"]
            rnd, rnd_name = extract_rnd(rows)
            revenue = extract_revenue(rows)  # 기본: 그 해 보고서의 당기 매출
            if latest_ok and year != YEARS[-1]:
                # 몇 해 전인지에 따라 전기/전전기 칸을 고른다 (2024 -> 전기, 2023 -> 전전기)
                field = ("thstrm_amount", "frmtrm_amount", "bfefrmtrm_amount")[YEARS[-1] - year]
                restated = extract_revenue(latest["list"], field)
                if restated is not None:
                    if revenue != restated:
                        print(f"  [재작성] {name} {year} 매출: {revenue / 1e8:,.0f}억 -> {restated / 1e8:,.0f}억")
                    revenue = restated
            results.append({
                "stock_code": code, "name": name, "year": year,
                "process": c["process"], "segment": c["segment"],
                "fs_div": data.get("fs_div_used"),
                "revenue": revenue,
                "rnd": rnd, "rnd_account": rnd_name,
            })
            print(f"{name} {year}: 완료")

    df = pd.DataFrame(results)
    out = os.path.join(OUT_DIR, "financials.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # ---- 요약 출력 (단위: 억 원) ----
    show = df.copy()
    show["매출(억)"] = (show["revenue"] / 1e8).round(0)
    show["R&D(억)"] = (show["rnd"] / 1e8).round(1)
    print("\n=== 수집 결과 요약 ===")
    print(show[["name", "year", "fs_div", "매출(억)", "R&D(억)"]].to_string(index=False))

    total = len(df)
    print(f"\n매출액 확보: {df['revenue'].notna().sum()}/{total}")
    print(f"R&D 자동 확보: {df['rnd'].notna().sum()}/{total}  (나머지는 수동 입력 대상)")
    print(f"저장 위치: {out}")


if __name__ == "__main__":
    main()
