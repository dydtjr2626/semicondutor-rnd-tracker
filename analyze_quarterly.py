"""분기 데이터로 '고객사 설비투자 -> 장비사 매출' 관계를 시차(lag)별로 계산한다.

실행:  python analyze_quarterly.py   (먼저 python collect_quarterly.py)
결과:  data/processed/lag_corr.csv

핵심 아이디어
  - 증가율은 '작년 같은 분기 대비(YoY)'로 만든다. 반도체는 분기마다 계절 흐름이 있어서
    바로 직전 분기와 비교하면 계절 때문에 흔들리기 때문이다.
  - 시차 L분기: '고객사가 투자한 뒤 L분기 지나서 장비사 매출이 반응하는가'를 본다.
    L=0이면 같은 분기, L=2면 투자 2분기 뒤의 매출과 비교한다.
  - 상관계수(r)는 -1~+1. +1에 가까우면 같이 오르내리고, -1에 가까우면 반대로 움직인다.
  - 신뢰구간(95%)은 '표본이 이만큼이라 r이 이 범위 안에 있을 것'이라는 뜻이다.
    범위가 0을 포함하면 '관계가 없다'와 구분되지 않는다.
"""
import math
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(BASE, "data", "processed", "quarterly.csv")
OUT_PATH = os.path.join(BASE, "data", "processed", "lag_corr.csv")

MAX_LAG = 4
TARGET = "장비사 매출 합계"
CAPEX_COLS = ["SK하이닉스 설비투자", "삼성전자 설비투자", "고객사 설비투자 합계"]


def yoy(series):
    """작년 같은 분기 대비 증가율(%). 4칸 전(=1년 전 같은 분기)과 비교한다."""
    return (series / series.shift(4) - 1) * 100


def fisher_ci(r, n):
    """피셔 z 변환으로 구한 상관계수 95% 신뢰구간 (표본이 4개 이하면 계산 불가)."""
    if n <= 3 or abs(r) >= 1:
        return (np.nan, np.nan)
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    return (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))


def shift_perm_p(x, y):
    """순환 이동 순열검정 p값. x를 한 칸씩 돌려(맨 끝이 맨 앞으로) 상관계수를 다시 구하고,
    진짜 상관(절댓값)보다 크거나 같은 경우의 비율을 센다. 인접 분기가 비슷한 성질(자기상관)을 그대로 두고
    '짝'만 어긋나게 하므로, 신뢰구간보다 보수적이다. 무작위가 없어 실행할 때마다 값이 같다.
    표본이 n개면 가장 작은 p값은 1/n 이라서(예: 19개 -> 0.05) 값이 거칠다."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    obs = abs(np.corrcoef(x, y)[0, 1])
    hits = sum(abs(np.corrcoef(np.roll(x, k), y)[0, 1]) >= obs - 1e-12 for k in range(1, n))
    return (hits + 1) / n  # +1 은 원래 짝 자신


def main():
    q = pd.read_csv(IN_PATH, encoding="utf-8-sig")
    rev_yoy = yoy(q[TARGET])

    rows = []
    for col in CAPEX_COLS:
        capex_yoy = yoy(q[col])
        for lag in range(MAX_LAG + 1):
            # 고객사 투자를 lag칸 뒤로 밀어, 같은 줄에 'lag분기 전 투자'와 '이번 분기 매출'이 오게 한다
            pair = pd.DataFrame({"capex": capex_yoy.shift(lag), "rev": rev_yoy}).dropna()
            n = len(pair)
            r = pair["capex"].corr(pair["rev"])
            # 순위 기준 상관(이상치에 덜 민감). 값을 순위로 바꾼 뒤 상관을 구하면 스피어만 상관과 같다
            rho = pair["capex"].rank().corr(pair["rev"].rank())
            lo, hi = fisher_ci(r, n)
            rows.append({"고객사 투자": col.replace(" 설비투자", ""), "시차(분기)": lag, "표본수": n,
                         "상관계수": round(r, 2), "하한": round(lo, 2), "상한": round(hi, 2),
                         "순위상관": round(rho, 2),
                         "순열p": round(shift_perm_p(pair["capex"], pair["rev"]), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))

    # 신뢰구간이 0을 포함하지 않는 것만 '일단 눈에 띄는 관계'로 뽑는다
    sig = out[(out["하한"] > 0) | (out["상한"] < 0)]
    print(f"\n신뢰구간이 0을 포함하지 않는 조합: {len(sig)}개 / 전체 {len(out)}개")
    print(sig.to_string(index=False) if len(sig) else "  (없음)")
    print(f"저장 위치: {OUT_PATH}")


if __name__ == "__main__":
    main()
