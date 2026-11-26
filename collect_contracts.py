"""장비사 5곳의 '단일판매·공급계약체결' 공시를 모아 '누구에게 얼마나 팔았는지'를 표로 만든다.

실행:  python collect_contracts.py
결과:  data/processed/contracts.csv   (공시 1건 = 1행, 정정공시는 하나로 합침)
       공시 원문은 data/raw/contracts/ 에 저장되어 다시 실행해도 API를 덜 부른다.

왜 이 공시를 쓰나?
  사업보고서의 '주요 고객' 표는 이름이 'A사, B사'로 가려져 있다. 반면 계약 공시에는
  계약상대(예: SK하이닉스)가 실명으로 적히는 경우가 많다. 다만 큰 계약만 공시하고,
  회사마다 공시를 멈춘 시기가 달라서(유진테크 2022년 이후 없음 등) '일부만 보이는 자료'이다.
"""
import io
import json
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("DART_API_KEY")

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "data", "raw")
DOC_DIR = os.path.join(RAW_DIR, "contracts")
OUT_PATH = os.path.join(BASE, "data", "processed", "contracts.csv")
os.makedirs(DOC_DIR, exist_ok=True)


def decode(raw):
    """오래된 공시는 cp949, 새 공시는 utf-8로 저장되어 있어서 둘 다 시도한다."""
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


def list_contract_filings(corp_code):
    """회사의 거래소 공시 중 '공급계약'이 제목에 들어간 것의 목록(접수번호, 날짜, 제목)."""
    found = []
    for page in (1, 2, 3):
        res = requests.get("https://opendart.fss.or.kr/api/list.json", params={
            "crtfc_key": KEY, "corp_code": corp_code, "bgn_de": "20210101", "end_de": "20261231",
            "pblntf_ty": "I", "page_no": page, "page_count": 100}, timeout=30).json()
        found += [x for x in res.get("list", []) if "공급계약" in x["report_nm"]]
        if page >= int(res.get("total_page", 1)):
            break
        time.sleep(0.3)
    return found


def filing_text(rcept_no):
    """공시 원문을 글자만 남겨 가져온다 (한 번 받으면 파일로 저장)."""
    path = os.path.join(DOC_DIR, rcept_no + ".txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    res = requests.get("https://opendart.fss.or.kr/api/document.xml",
                       params={"crtfc_key": KEY, "rcept_no": rcept_no}, timeout=60)
    z = zipfile.ZipFile(io.BytesIO(res.content))
    text = " ".join(decode(z.read(n)) for n in z.namelist())
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    time.sleep(0.2)
    return text


def first(pattern, text):
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def parse(text):
    """공시 양식이 두 가지(옛 양식 / 새 양식)라서 두 가지 표현을 모두 찾는다."""
    amount = first(r"계약금액 총액\(원\)\s*([\d,]+)", text) or first(r"계약금액\(원\)\s*([\d,]+)", text)
    counterparty = first(r"3\. 계약상대방?\s*(.+?)\s*(?:-\s*최근 매출액|-\s*주요사업|-?\s*회사와의 관계)", text)
    date = first(r"계약\(수주\)일자?\s*(\d{4}-\d{2}-\d{2})", text)
    return (int(amount.replace(",", "")) if amount else None), counterparty, date


def customer_group(counterparty):
    """계약상대를 '고객 묶음'으로 나눈다. 삼성전기·삼성디스플레이는 삼성전자가 아니므로 기타로 둔다."""
    c = (counterparty or "").lower()
    if "하이닉스" in c or "hynix" in c:
        return "SK하이닉스"
    if "삼성전자" in c or "samsung electronics" in c or "samsung(china) semiconductor" in c:
        return "삼성전자"
    return "기타"


def main():
    if not KEY:
        raise SystemExit(".env 파일에 DART_API_KEY가 없습니다.")
    companies = pd.read_csv(os.path.join(BASE, "companies.csv"), dtype={"stock_code": str}, encoding="utf-8-sig")
    equip = companies[companies["segment"] == "장비"]
    corp_of = {}
    for item in ET.parse(os.path.join(RAW_DIR, "CORPCODE.xml")).getroot().iter("list"):
        code = (item.findtext("stock_code") or "").strip()
        if code in set(equip["stock_code"]):
            corp_of[code] = item.findtext("corp_code")

    rows = []
    for _, c in equip.iterrows():
        filings = list_contract_filings(corp_of[c["stock_code"]])
        for f in filings:
            amount, cp, date = parse(filing_text(f["rcept_no"]))
            rows.append({"name": c["name"], "filed": f["rcept_dt"], "title": f["report_nm"].strip(),
                         "rcept_no": f["rcept_no"], "contract_date": date, "amount_won": amount,
                         "counterparty": cp, "customer": customer_group(cp)})
        print(f"{c['name']}: 공급계약 공시 {len(filings)}건")

    df = pd.DataFrame(rows)
    # 정정공시는 원래 공시와 같은 계약이므로, (회사, 계약상대, 계약일)이 같으면 가장 나중 것만 남긴다
    before = len(df)
    df = (df.sort_values("filed").drop_duplicates(["name", "counterparty", "contract_date"], keep="last")
            .sort_values(["name", "contract_date"]).reset_index(drop=True))
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n정정·중복 제거: {before}건 -> {len(df)}건")
    print("금액을 못 읽은 건:", int(df["amount_won"].isna().sum()))
    summary = df.pivot_table(index="name", columns="customer", values="amount_won", aggfunc="sum", fill_value=0)
    share = (summary.div(summary.sum(axis=1), axis=0) * 100).round(0)
    print("\n=== 공시된 계약금액의 고객별 비중(%) ===")
    print(share.to_string())
    print("\n=== 공시 기간 ===")
    print(df.groupby("name")["contract_date"].agg(["min", "max", "count"]).to_string())
    print(f"저장 위치: {OUT_PATH}")


if __name__ == "__main__":
    main()
