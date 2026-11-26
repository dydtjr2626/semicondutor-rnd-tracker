"""기업별 DART 사업보고서 원문 링크(접수번호)를 모아 CSV로 저장한다.

실행:  python collect_source_links.py
결과:  data/processed/source_links.csv
       (data/raw/ 에 이미 받아 둔 재무제표 응답만 읽는다. DART API를 다시 부르지 않으므로 API 키가 필요 없다)

왜 필요한가: 화면의 숫자가 어느 보고서에서 나왔는지 사용자가 직접 눌러서 확인할 수 있게 하려는 것이다.
접수번호(rcept_no)는 공시 한 건마다 붙는 고유 번호이고, 이 번호로 DART 원문 주소를 만들 수 있다.
"""
import json
import os
import xml.etree.ElementTree as ET

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "data", "raw")
OUT_PATH = os.path.join(BASE, "data", "processed", "source_links.csv")
REPORT_YEAR = 2025  # R&D와 3개년 매출을 모두 이 사업보고서 한 권에서 읽었다
VIEW_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo="


def load_corp_map():
    """종목코드 -> corp_code (collect_data.py가 받아 둔 CORPCODE.xml을 읽는다)."""
    root = ET.parse(os.path.join(RAW_DIR, "CORPCODE.xml")).getroot()
    out = {}
    for item in root.iter("list"):
        code = (item.findtext("stock_code") or "").strip()
        if code:
            out[code] = item.findtext("corp_code")
    return out


def main():
    companies = pd.read_csv(os.path.join(BASE, "companies.csv"), dtype={"stock_code": str})
    corp_map = load_corp_map()
    rows = []
    for _, c in companies.iterrows():
        path = os.path.join(RAW_DIR, f"{corp_map[c['stock_code']]}_{REPORT_YEAR}.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # 한 보고서의 모든 행은 같은 접수번호를 가지므로 첫 행에서 꺼낸다
        rcept = {r["rcept_no"] for r in data["list"]}
        if len(rcept) != 1:
            raise SystemExit(f"{c['name']}: 접수번호가 {len(rcept)}개입니다. 확인이 필요합니다: {sorted(rcept)}")
        no = rcept.pop()
        rows.append({"stock_code": c["stock_code"], "name": c["name"], "report_year": REPORT_YEAR,
                     "rcept_no": no, "url": VIEW_URL + no})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\n{len(out)}개사 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
