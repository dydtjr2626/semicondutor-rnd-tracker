import io
import os
import zipfile
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 API 키 읽기
KEY = os.getenv("DART_API_KEY")


# ---- A. 종목코드로 DART 고유번호(corp_code) 찾기 ----
def find_corp_code(stock_code):
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    res = requests.get(url, params={"crtfc_key": KEY})
    with zipfile.ZipFile(io.BytesIO(res.content)) as z:
        xml_data = z.read("CORPCODE.xml")
    root = ET.fromstring(xml_data)
    for item in root.iter("list"):
        if item.findtext("stock_code") == stock_code:
            return item.findtext("corp_code"), item.findtext("corp_name")
    return None, None


# ---- B. 재무제표 가져오기 (연결 CFS 먼저, 없으면 별도 OFS) ----
def get_financials(corp_code, year):
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    for fs_div in ("CFS", "OFS"):  # CFS = 연결, OFS = 별도
        params = {
            "crtfc_key": KEY,
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": "11011",  # 11011 = 사업보고서(연간)
            "fs_div": fs_div,
        }
        data = requests.get(url, params=params).json()
        if data["status"] == "000":  # 000 = 정상
            data["fs_div_used"] = fs_div
            return data
    return data  # 둘 다 실패하면 마지막 응답 반환


if __name__ == "__main__":
    if not KEY:
        raise SystemExit(".env 파일에 DART_API_KEY가 없습니다.")

    corp_code, name = find_corp_code("058470")  # 058470 = 리노공업
    print("회사:", name, "/ corp_code:", corp_code)

    data = get_financials(corp_code, 2024)
    print("응답 상태:", data["status"], data["message"], "/ 사용한 구분:", data.get("fs_div_used"))

    # 손익계산서(IS/CIS) 항목 출력 -> '연구개발' 단어가 있는지 눈으로 확인!
    for row in data.get("list", []):
        if row["sj_div"] in ("IS", "CIS"):
            print(row["account_nm"], "|", row["thstrm_amount"])