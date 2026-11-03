"""자동 수집 데이터(financials.csv) + 수동 입력(rnd_manual.csv)을 합쳐 R&D 비중을 계산한다.

실행:  python metrics.py
결과:  data/processed/master.csv   (대시보드가 읽을 최종 데이터)
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
FIN_PATH = os.path.join(BASE, "data", "processed", "financials.csv")
MAN_PATH = os.path.join(BASE, "data", "manual", "rnd_manual.csv")
OUT_PATH = os.path.join(BASE, "data", "processed", "master.csv")

# 직접 계산한 비율과 보고서 공시 비율의 차이가 이 값(%p)보다 크면 '기준 상이'로 본다.
THRESHOLD = 0.5

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def main():
    fin = pd.read_csv(FIN_PATH, dtype={"stock_code": str}, encoding="utf-8-sig")
    man = pd.read_csv(MAN_PATH, dtype={"stock_code": str}, encoding="utf-8-sig")

    df = fin[["stock_code", "name", "year", "process", "segment", "fs_div", "revenue"]].merge(
        man[["stock_code", "year", "rnd_total_mil_won", "reported_ratio_pct", "source_note"]],
        on=["stock_code", "year"],
        how="left",
    )

    # ---- 1. R&D 비중 계산 ----
    df["rnd"] = df["rnd_total_mil_won"] * 1_000_000          # 백만원 -> 원
    df["calc_ratio"] = df["rnd"] / df["revenue"] * 100        # 직접 계산한 비중(%)
    df["diff"] = df["calc_ratio"] - df["reported_ratio_pct"]  # 공시 비율과의 차이(%p)

    # ---- 2. 최종 지표 선택 ----
    # 기본: 직접 계산값. 단, 공시 비율과 많이 다르면(R&D와 매출의 기준이 다르다는 뜻)
    #       보고서에 적힌 공시 비율을 사용하고 표시를 남긴다.
    df["rnd_ratio"] = df["calc_ratio"]
    df["ratio_source"] = "계산(R&D/매출)"
    # 회사 단위로 판정: 한 해라도 크게 다르면 그 회사는 3개년 모두 공시 비율을 쓴다
    # (같은 회사 안에서 계산 방식이 섞이면 연도별 추이가 왜곡되기 때문)
    max_diff = df.groupby("stock_code")["diff"].transform(lambda s: s.abs().max())
    mismatch = max_diff > THRESHOLD
    df.loc[mismatch, "rnd_ratio"] = df.loc[mismatch, "reported_ratio_pct"]
    df.loc[mismatch, "ratio_source"] = "공시비율(기준상이)"

    # ---- 3. 추가 지표 ----
    df = df.sort_values(["stock_code", "year"]).reset_index(drop=True)
    df["revenue_yoy"] = df.groupby("stock_code")["revenue"].pct_change() * 100
    df["rnd_yoy"] = df.groupby("stock_code")["rnd"].pct_change() * 100
    df["process_median"] = df.groupby(["process", "year"])["rnd_ratio"].transform("median")
    df["vs_process"] = df["rnd_ratio"] - df["process_median"]  # 공정 중앙값 대비 차이(%p)

    for col in ["calc_ratio", "diff", "rnd_ratio", "revenue_yoy", "rnd_yoy", "process_median", "vs_process"]:
        df[col] = df[col].round(2)

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    # ---- 4. 요약 출력 ----
    print("=== 기업별 R&D 비중(%) ===")
    pivot = df.pivot_table(index=["process", "name"], columns="year", values="rnd_ratio")
    print(pivot.round(1).to_string())

    print("\n=== 공정별 평균 R&D 비중(%) ===")
    print(df.pivot_table(index="process", columns="year", values="rnd_ratio", aggfunc="mean").round(1).to_string())

    print("\n=== 공정 구분(장비/소재/부품/후공정)별 평균 ===")
    print(df.pivot_table(index="segment", columns="year", values="rnd_ratio", aggfunc="mean").round(1).to_string())

    flagged = df[df["ratio_source"].str.contains("기준상이")][["name", "year", "calc_ratio", "reported_ratio_pct"]]
    print(f"\n=== 기준 상이로 공시비율을 쓴 행: {len(flagged)}개 ===")
    print(flagged.to_string(index=False))
    print(f"\nR&D 누락 행: {df['rnd_ratio'].isna().sum()}개")
    print(f"저장 위치: {OUT_PATH}")


if __name__ == "__main__":
    main()
