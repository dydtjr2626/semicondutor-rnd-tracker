"""장비사별로 '어느 고객에 얼마나 기대는가'와 '그 고객의 투자와 얼마나 같이 움직이는가'를 한 표로 모은다.

실행:  python analyze_exposure.py
        (먼저 collect_quarterly.py, collect_contracts.py 를 실행하고
         data/manual/customer_concentration.csv 가 있어야 한다)
결과:  data/processed/exposure.csv   (장비사 5곳 x 지표)

세 가지 자료를 붙인다.
  1) 사업보고서 '주요 고객' : 상위 고객이 매출에서 차지하는 비중 (고객 이름은 가려져 있다)
  2) 계약 공시              : 공시된 계약금액 중 SK하이닉스·삼성전자 몫 (이름이 보이지만 일부 계약만 공시)
  3) 분기 데이터            : 장비사 매출 증가율이 고객 투자 증가율과 얼마나 같이 움직였는지(상관계수)
'가설': 어느 고객에 기대는 회사일수록, 그 고객의 투자와 더 같이 움직일 것이다.
"""
import math
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
MIN_CONTRACTS = 5  # 공시 계약이 이 값보다 적으면 비중을 계산하지 않는다 (1~3건으로 비중을 말하면 오해를 부른다)

EQUIP = ["원익IPS", "주성엔지니어링", "유진테크", "한미반도체", "파크시스템스"]
CUSTOMERS = ["SK하이닉스", "삼성전자"]


def yoy(s):
    return (s / s.shift(4) - 1) * 100  # 작년 같은 분기 대비 증가율(%)


def corr_ci(x, y):
    """상관계수와 95% 신뢰구간(피셔 z), 표본 수."""
    p = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(p)
    r = p["x"].corr(p["y"])
    se = 1 / math.sqrt(n - 3)
    return r, math.tanh(math.atanh(r) - 1.96 * se), math.tanh(math.atanh(r) + 1.96 * se), n


def main():
    q = pd.read_csv(os.path.join(PROC, "quarterly.csv"), encoding="utf-8-sig")
    master = pd.read_csv(os.path.join(PROC, "master.csv"), dtype={"stock_code": str}, encoding="utf-8-sig")
    conc = pd.read_csv(os.path.join(BASE, "data", "manual", "customer_concentration.csv"), encoding="utf-8-sig")
    contracts = pd.read_csv(os.path.join(PROC, "contracts.csv"), encoding="utf-8-sig")

    rows = []
    for name in EQUIP:
        row = {"기업": name}

        # 1) 상위 고객 비중 = 고객 매출 / 그 해 매출 (2025년 사업보고서 기준)
        rev = master[(master["name"] == name) & (master["year"] == 2025)]["revenue"].iloc[0] / 1e6  # 백만원
        c = conc[(conc["name"] == name) & (conc["year"] == 2025)].sort_values("amount_mil_won", ascending=False)
        ratios = (c["amount_mil_won"] / rev * 100).tolist()
        row["상위 1개 고객 비중(%)"] = round(ratios[0], 1)
        row["상위 2개 고객 비중(%)"] = round(sum(ratios[:2]), 1)

        # 2) 공시된 계약금액 중 고객 몫
        k = contracts[contracts["name"] == name]
        row["공시 계약 수"] = len(k)
        for cust in CUSTOMERS:
            share = k.loc[k["customer"] == cust, "amount_won"].sum() / k["amount_won"].sum() * 100
            row[f"{cust} 계약 비중(%)"] = round(share, 0) if len(k) >= MIN_CONTRACTS else None

        # 3) 분기 매출 증가율과 고객 투자 증가율의 상관 (같은 분기, 시차 0)
        for cust in CUSTOMERS:
            r, lo, hi, n = corr_ci(yoy(q[f"{cust} 설비투자"]), yoy(q[f"{name} 매출"]))
            row[f"{cust} 상관"] = round(r, 2)
            row[f"{cust} 하한"] = round(lo, 2)
            row[f"{cust} 상한"] = round(hi, 2)
            row["표본수"] = n
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PROC, "exposure.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\n저장 위치: {os.path.join(PROC, 'exposure.csv')}")


if __name__ == "__main__":
    main()
