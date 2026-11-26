"""master.csv(대시보드가 읽는 최종 데이터)를 자동으로 검사한다.

실행:  python validate.py
결과:  항목마다 PASS / FAIL / 경고를 출력한다. FAIL이 하나라도 있으면 종료 코드 1로 끝난다.

왜 필요한가: 지금까지는 사람이 CSV를 눈으로 확인했다. 숫자가 한 군데 틀려도 놓칠 수 있어서,
'이 조건이면 틀린 것'이라는 규칙을 코드로 적어 두고 데이터를 바꿀 때마다 자동으로 돌린다.
- FAIL: 틀린 데이터. 고쳐야 한다.
- 경고: 틀린 건 아니지만 한 번 더 보라는 신호(급증, 기준값에 가까운 차이 등).
"""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
MASTER_PATH = os.path.join(PROC, "master.csv")
COMPANIES_PATH = os.path.join(BASE, "companies.csv")

EXPECTED_YEARS = [2023, 2024, 2025]
THRESHOLD = 0.5          # metrics.py의 THRESHOLD와 같아야 한다 (공시 비율로 바꾸는 기준, %p)
NEAR_THRESHOLD = 0.4     # 이 값 이상이면 '기준에 가깝다'고 경고만 한다 (ISC가 0.44)
RATIO_RANGE = (0, 40)    # R&D 비중(%)이 이 범위를 벗어나면 단위 오류를 의심한다
YOY_ALERT = 100          # 전년 대비 증감률이 이 값(%)을 넘으면 경고 (한미반도체 매출 +251%)
ALLOWED_SOURCES = {"계산(R&D/매출)", "공시비율(기준상이)"}
OPTIONAL_FILES = ["capex.csv", "quarterly.csv", "lag_corr.csv", "exposure.csv", "source_links.csv"]


def run_checks(df, companies):
    """검사 결과를 [(상태, 항목, 설명), ...] 로 돌려준다. 상태는 'PASS' / 'FAIL' / 'WARN'."""
    out = []

    def add(ok, name, detail=""):
        out.append(("PASS" if ok else "FAIL", name, detail))

    def warn(name, detail):
        out.append(("WARN", name, detail))

    # 1. 기업·연도 구성: companies.csv에 있는 기업이 모두, 연도마다 한 번씩만 있어야 한다
    want = set(companies["stock_code"])
    got = set(df["stock_code"])
    add(want == got, "기업 목록이 companies.csv와 같다",
        f"빠진 기업 {sorted(want - got)}, 더 있는 기업 {sorted(got - want)}" if want != got else f"{len(got)}곳")
    add(len(df) == len(want) * len(EXPECTED_YEARS), "행 수 = 기업 수 × 연도 수", f"{len(df)}행")
    dup = df.duplicated(["stock_code", "year"]).sum()
    add(dup == 0, "(기업, 연도) 중복이 없다", f"중복 {dup}행")
    bad_years = [n for n, g in df.groupby("name") if sorted(g["year"]) != EXPECTED_YEARS]
    add(not bad_years, f"모든 기업이 {EXPECTED_YEARS[0]}~{EXPECTED_YEARS[-1]} 연도를 갖췄다", f"문제 기업 {bad_years}")

    # 2. 결측: 핵심 열이 비어 있으면 화면 숫자가 틀려진다
    core = ["revenue", "rnd_total_mil_won", "reported_ratio_pct", "rnd_ratio", "calc_ratio"]
    nulls = {c: int(df[c].isna().sum()) for c in core if df[c].isna().any()}
    add(not nulls, "핵심 열에 빈 값이 없다", f"빈 값 {nulls}")

    # 3. 계산 재현: master.csv의 비중이 원자료(R&D, 매출)로 다시 계산한 값과 같은가
    recalc = df["rnd_total_mil_won"] * 1_000_000 / df["revenue"] * 100
    gap = (recalc - df["calc_ratio"]).abs().max()
    add(gap < 0.01, "calc_ratio가 R&D÷매출로 다시 계산한 값과 같다", f"최대 차이 {gap:.4f}%p")
    yoy = df.sort_values(["stock_code", "year"]).copy()
    yoy["re"] = yoy.groupby("stock_code")["revenue"].pct_change() * 100
    gap_y = (yoy["re"] - yoy["revenue_yoy"]).abs().max()
    add(pd.isna(gap_y) or gap_y < 0.01, "매출 YoY가 다시 계산한 값과 같다", f"최대 차이 {gap_y:.4f}%p")

    # 4. 값의 범위와 허용 값
    lo, hi = RATIO_RANGE
    out_of = df[(df["rnd_ratio"] < lo) | (df["rnd_ratio"] > hi)]["name"].unique().tolist()
    add(not out_of, f"R&D 비중이 {lo}~{hi}% 안에 있다", f"벗어난 기업 {out_of}")
    odd = sorted(set(df["ratio_source"]) - ALLOWED_SOURCES)
    add(not odd, "ratio_source가 허용된 값뿐이다", f"예상 밖 값 {odd}")

    # 5. 0.5%p 규칙: 회사 안에서 방식이 섞이지 않고, 기준을 넘는 회사만 공시 비율을 쓴다
    per = df.groupby("name").agg(max_diff=("diff", lambda s: s.abs().max()), n_src=("ratio_source", "nunique"))
    add((per["n_src"] == 1).all(), "한 회사 안에서 산출 방식이 섞이지 않는다", f"섞인 회사 {per[per['n_src'] > 1].index.tolist()}")
    uses_reported = df.groupby("name")["ratio_source"].first().str.contains("공시")
    wrong = [n for n in per.index if (per.loc[n, "max_diff"] > THRESHOLD) != uses_reported[n]]
    add(not wrong, f"공시 비율은 차이가 {THRESHOLD}%p를 넘는 회사만 쓴다", f"규칙과 다른 회사 {wrong}")
    rep = df[df["ratio_source"].str.contains("공시")]
    add((rep["rnd_ratio"] == rep["reported_ratio_pct"]).all(), "공시 비율 회사의 rnd_ratio는 공시 값과 같다")
    cal = df[~df["ratio_source"].str.contains("공시")]
    add(((cal["rnd_ratio"] - cal["calc_ratio"]).abs() < 1e-9).all(), "계산 회사의 rnd_ratio는 계산값과 같다")

    # 6. 함께 쓰는 파일 (앱이 읽는 다른 CSV)
    missing = [f for f in OPTIONAL_FILES if not os.path.exists(os.path.join(PROC, f))]
    add(not missing, "앱이 읽는 다른 CSV가 모두 있다", f"없는 파일 {missing}")
    sl = os.path.join(PROC, "source_links.csv")
    if os.path.exists(sl):
        s = pd.read_csv(sl, dtype=str)
        ok = set(s["name"]) >= set(df["name"]) and s["rcept_no"].str.fullmatch(r"\d{14}").all()
        add(ok, "source_links.csv에 모든 기업의 14자리 접수번호가 있다")

    # 7. R&D 분해표(rnd_breakdown.csv): 보고서 표에서 옮긴 총액이 rnd_manual.csv의 입력값과 같아야 한다
    bp = os.path.join(BASE, "data", "manual", "rnd_breakdown.csv")
    if os.path.exists(bp):
        b = pd.read_csv(bp, dtype={"stock_code": str}, encoding="utf-8-sig")
        m = b.merge(df[["stock_code", "year", "rnd_total_mil_won"]], on=["stock_code", "year"], how="left")
        bad = m[(m["report_total_mil_won"] - m["rnd_total_mil_won"]).abs() >= 1]
        add(bad.empty, "R&D 분해표의 총액이 수동 입력값과 같다", f"다른 행 {bad[['name', 'year']].values.tolist()}")
        gross = m["report_total_mil_won"] + m["govt_grant_mil_won"].where(m["total_basis"] == "보조금 차감 후", 0)
        # 자산화 값이 비어 있는 행(표에 구분이 없는 회사)은 비교에서 뺀다
        cap = m["capitalized_mil_won"]
        add((cap.dropna() <= gross[cap.notna()]).all() and (m["govt_grant_mil_won"] >= 0).all(),
            "분해표의 자산화·보조금 값이 말이 된다(자산화 ≤ 총액, 보조금 ≥ 0)")
        add(len(b) == len(set(b["stock_code"])) * len(EXPECTED_YEARS) and set(b["stock_code"]) <= want,
            "분해표가 기업마다 3개 연도를 갖췄고 모두 분석 대상 기업이다", f"{len(set(b['stock_code']))}곳, {len(b)}행")

    # --- 경고: 틀린 건 아니지만 사람이 한 번 더 볼 것 ---
    for _, r in df[df["revenue_yoy"].abs() > YOY_ALERT].iterrows():
        warn("급격한 변화", f"{r['name']} {int(r['year'])}년 매출 YoY {r['revenue_yoy']:+.1f}%")
    for n, row in per.iterrows():
        if NEAR_THRESHOLD <= row["max_diff"] <= THRESHOLD:
            warn("기준에 가까운 차이", f"{n}: 공시 비율과 최대 {row['max_diff']:.2f}%p 차이 (기준 {THRESHOLD}%p). 데이터가 바뀌면 방식이 바뀔 수 있음")
    return out


def main():
    df = pd.read_csv(MASTER_PATH, dtype={"stock_code": str}, encoding="utf-8-sig")
    companies = pd.read_csv(COMPANIES_PATH, dtype={"stock_code": str})
    results = run_checks(df, companies)
    for status, name, detail in results:
        print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    fails = sum(1 for s, _, _ in results if s == "FAIL")
    warns = sum(1 for s, _, _ in results if s == "WARN")
    print(f"\n검사 {sum(1 for s, _, _ in results if s != 'WARN')}개 중 실패 {fails}개, 경고 {warns}개")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
