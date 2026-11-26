"""반도체 소부장 R&D 투자 비중 트래커 - Streamlit 대시보드

실행:  streamlit run app.py
읽는 파일 (모두 data/processed/ 아래):
  master.csv     필수. python metrics.py 로 생성 (없으면 앱이 멈추고 안내만 보여 준다)
  capex.csv      선택. python analyze_capex.py          -> '고객사 투자' 탭의 연 단위 표
  quarterly.csv  선택. python collect_quarterly.py       -> '고객사 투자' 탭의 분기 차트
  lag_corr.csv   선택. python analyze_quarterly.py       -> '고객사 투자' 탭의 시차 차트
  exposure.csv   선택. python analyze_exposure.py        -> '고객사 투자' 탭의 회사별 차트
  contracts.csv  선택. python collect_contracts.py       -> 회사별 설명 문장의 계약 건수·기간
선택 파일이 없으면 그 부분만 "어떤 명령을 실행하세요"라는 안내로 바뀐다.
"""
import math
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MASTER_PATH = os.path.join(BASE, "data", "processed", "master.csv")
CAPEX_PATH = os.path.join(BASE, "data", "processed", "capex.csv")
QUARTERLY_PATH = os.path.join(BASE, "data", "processed", "quarterly.csv")
LAG_PATH = os.path.join(BASE, "data", "processed", "lag_corr.csv")
EXPOSURE_PATH = os.path.join(BASE, "data", "processed", "exposure.csv")
CONTRACTS_PATH = os.path.join(BASE, "data", "processed", "contracts.csv")
PEERS_PATH = os.path.join(BASE, "data", "manual", "global_peers.csv")  # 글로벌 장비사(SEC 10-K·20-F 수동 입력)
FABS_PATH = os.path.join(BASE, "data", "manual", "customer_fabs.csv")  # 고객이 될 수 있는 다른 팹(DB하이텍·Micron·Intel·TSMC, 수동 입력)

PROCESS_ORDER = ["증착/식각", "패키징", "테스트", "검사/계측", "소재"]

# 색은 '구분' 3가지에만 쓴다 (장비 / 부품·후공정 / 소재).
# 3색은 색각 이상 기준으로 검증된 팔레트의 앞 3칸이라, 어떤 차트에서도 서로 구분된다.
# "total"은 시차 차트의 '고객사 합계' 선 하나에만 쓰는 보조색이다 (구분 색과 겹치지 않게 보라 계열).
GROUP_OF_SEGMENT = {"장비": "장비", "부품": "부품·후공정", "후공정": "부품·후공정", "소재": "소재"}
GROUPS = ["장비", "부품·후공정", "소재"]
COLORS = {
    "light": {
        "장비": "#2a78d6", "부품·후공정": "#eb6834", "소재": "#1baf7a", "total": "#8a5cc2",
        "ink": "#52514e", "grid": "#e1e0d9", "muted": "#898781", "surface": "#ffffff",
    },
    "dark": {
        "장비": "#3987e5", "부품·후공정": "#d95926", "소재": "#199e70", "total": "#b08ae0",
        "ink": "#c3c2b7", "grid": "#2c2c2a", "muted": "#898781", "surface": "#0e1117",
    },
}
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "pentagon"]

DISCLAIMER = "공시 자료를 바탕으로 한 학습·분석용 자료이며 투자 권유가 아닙니다. 수치는 원문 공시(Open DART)로 확인하세요."


# 데이터를 마지막으로 내려받아 정리한 달. 새로 수집해서 CSV를 다시 만들면 이 값도 함께 고친다.
# (파일 수정 시각을 쓰지 않는 이유: 배포하면 파일 시각이 '배포한 날'로 바뀌어 기준일이 틀려 보인다)
COLLECTED_AT = "2026-11"
BREAKDOWN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "manual", "rnd_breakdown.csv")
SOURCE_LINKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "source_links.csv")



def theme_name():
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


# ---------- 데이터 ----------
@st.cache_data
def load_links():
    """기업 이름 -> DART 사업보고서 원문 주소. 파일이 없으면 빈 지도를 돌려줘서 링크 열만 비어 보이게 한다."""
    try:
        s = pd.read_csv(SOURCE_LINKS_PATH, dtype=str, encoding="utf-8-sig")
        return dict(zip(s["name"], s["url"]))
    except (OSError, KeyError, ValueError):
        return {}


@st.cache_data
def load_data(file_mtime):
    # file_mtime(파일 수정 시간)을 인자로 받는 이유: 값이 바뀌면 캐시 키가 달라져서
    # master.csv를 고칠 때마다 자동으로 새로 읽는다 (함수 안에서는 쓰지 않는다)
    df = pd.read_csv(MASTER_PATH, dtype={"stock_code": str}, encoding="utf-8-sig")
    df["group"] = df["segment"].map(GROUP_OF_SEGMENT)
    df["revenue_eok"] = df["revenue"] / 1e8
    df["rnd_eok"] = df["rnd"] / 1e8
    # 같은 회사는 연도가 달라도 한 줄/한 표기여야 하므로 회사 단위로 판정한다
    df["basis_flag"] = df.groupby("stock_code")["ratio_source"].transform(
        lambda s: s.str.contains("기준상이").any()
    )
    df["label"] = df["name"] + df["basis_flag"].map({True: "*", False: ""})
    return df


@st.cache_data
def read_table(path, file_mtime):
    """선택 CSV 한 개를 읽는다. master.csv처럼 파일 수정 시간을 키로 해서, 파일이 바뀌면 새로 읽는다."""
    return pd.read_csv(path, encoding="utf-8-sig")


def load_csv(path):
    """선택 CSV를 읽어 돌려준다. 파일이 없거나 읽을 수 없으면 None."""
    if not os.path.exists(path):
        return None
    try:
        return read_table(path, os.path.getmtime(path))
    except Exception:  # 빈 파일·깨진 파일 등
        return None


def need(df, path, command):
    """df가 None이면 '무슨 명령을 실행하라'는 안내를 보여 주고 False를 돌려준다."""
    if df is None or df.empty:
        rel = os.path.relpath(path, BASE).replace(os.sep, "/")
        st.info(f"{rel} 파일이 없거나 비어 있습니다. 터미널에서 `{command}` 를 실행하세요.")
        return False
    return True


@contextmanager
def guard(hint):
    """CSV 열 이름·행이 예상과 다르면(낡은 파일 등) 앱이 죽는 대신 안내를 보여 준다."""
    try:
        yield
    except (KeyError, IndexError, ValueError) as e:
        st.warning(f"{hint} 파일 내용이 예상과 달라 이 부분을 그리지 못했습니다 ({type(e).__name__}: {e}). 분석 스크립트를 다시 실행해 보세요.")


# ---------- 차트 ----------
def mk(symbol, color, size, c):
    """마커 설정. 삼성은 속이 빈 사각(square-open), 합계는 x처럼 '모양·채움'으로도 구분되게 한다
    (회색 두 계열이 색만으로는 비슷해 보일 수 있어서, 색각 이상이어도 모양으로 구분 가능)."""
    if symbol.endswith("-open") or symbol == "x":
        return dict(symbol=symbol, size=size, color=color, line=dict(width=2.5, color=color))
    return dict(symbol=symbol, size=size, color=color, line=dict(color=c["surface"], width=1.5))


def style(fig, height, c):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=24, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=c["ink"], size=13),
        legend=dict(orientation="h", x=0, y=1.08, xanchor="left", yanchor="bottom"),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(gridcolor=c["grid"], gridwidth=1, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=c["grid"], gridwidth=1, zeroline=False, showline=False)
    return fig


def fab_table(ref_all):
    """고객이 될 수 있는 반도체 제조사(팹)의 R&D 비중을 한 표로 모은다.
    삼성전자·SK하이닉스는 master.csv(DART), 나머지는 customer_fabs.csv(DART·SEC 수동 입력)에서 읽는다.
    표는 참고용이라 차트·12개사 평균에는 쓰지 않는다. 값이 없는 연도는 '-'로 둔다."""
    rows = []
    kind = {"삼성전자": ("종합 반도체(메모리·파운드리)", "한국"), "SK하이닉스": ("메모리 IDM", "한국")}
    if ref_all is not None and not ref_all.empty:
        for name, g in ref_all.groupby("name"):
            r = {int(y): v for y, v in zip(g["year"], g["rnd_ratio"])}
            rows.append({"회사": name, "유형": kind.get(name, ("", ""))[0], "국가": kind.get(name, ("", ""))[1], **r})
    f = load_csv(FABS_PATH)
    if f is not None and not f.empty:
        f["rnd_ratio"] = f["rnd_mil"] / f["revenue_mil"] * 100  # 같은 통화끼리 나누므로 환율이 필요 없다
        for name, g in f.groupby("name", sort=False):
            r = {int(y): v for y, v in zip(g["fiscal_year"], g["rnd_ratio"])}
            rows.append({"회사": name, "유형": g["type"].iloc[0], "국가": g["country"].iloc[0], **r})
    if not rows:
        return None
    t = pd.DataFrame(rows).set_index("회사")
    year_cols = sorted(col for col in t.columns if isinstance(col, int))
    t = t[["유형", "국가"] + year_cols]
    # 화면에서 빈 칸이 'None'으로 보이지 않게, 값을 미리 글자("12.8%")로 바꾸고 없으면 "-"로 둔다
    for col in year_cols:
        t[col] = t[col].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}%")
    t.columns = [f"{col}년" if isinstance(col, int) else col for col in t.columns]
    return t


def peer_ratios(year):
    """global_peers.csv에서 해당 연도(회계연도 종료 연도 기준)의 R&D 비중(%)을 계산해 돌려준다. 없으면 빈 표."""
    p = load_csv(PEERS_PATH)
    if p is None or p.empty:
        return pd.DataFrame(columns=["name", "rnd_ratio"])
    p = p[p["fiscal_year"] == year].copy()
    # 비중은 같은 통화끼리 나누므로 환율이 필요 없다
    p["rnd_ratio"] = p["rnd_mil"] / p["revenue_mil"] * 100
    return p[["name", "rnd_ratio"]]


def bar_chart(d, c, ref=None, peers=None):
    d = d.sort_values("rnd_ratio")  # 가로 막대는 오름차순으로 넣어야 큰 값이 위로 간다
    fig = go.Figure()
    for g in GROUPS:
        s = d[d["group"] == g]
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            x=s["rnd_ratio"], y=s["label"], orientation="h", name=g,
            marker=dict(color=c[g]),
            text=[f"{v:.1f}%" for v in s["rnd_ratio"]],
            textposition="outside", cliponaxis=False, textfont=dict(color=c["ink"]),
            customdata=s[["process", "revenue_eok", "rnd_eok", "ratio_source"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>공정: %{customdata[0]}<br>R&D 비중: %{x:.1f}%"
                "<br>매출: %{customdata[1]:,.0f}억원<br>R&D: %{customdata[2]:,.0f}억원"
                "<br>산출: %{customdata[3]}<extra></extra>"
            ),
        ))
    fig.update_layout(barmode="overlay", bargap=0.45)
    fig.update_yaxes(categoryorder="array", categoryarray=list(d["label"]), showgrid=False)

    # 기준선은 두 개뿐이다: 국내 고객사(삼성전자·SK하이닉스) 평균, 글로벌 장비사 4곳 평균.
    # 이름은 선 위에 직접 적고, 개별 회사 값은 이름에 마우스를 올리면(hovertext) 보인다.
    # 둘 다 12개사 평균·순위에는 넣지 않는다 (규모와 회계기준이 달라 비교용으로만 쓴다).
    guides = []
    for name, t in (("국내 고객사 평균", ref), ("글로벌 장비사 평균", peers)):
        if t is None or t.empty:
            continue
        tip = "<br>".join(f"{r['name']}{' (전사)' if r['name'] == '삼성전자' else ''} {r['rnd_ratio']:.1f}%" for _, r in t.iterrows())
        guides.append((name, float(t["rnd_ratio"].mean()), tip))

    right = max([d["rnd_ratio"].max()] + [v for _, v, _ in guides])  # 기준선이 축 밖으로 나가지 않게
    fig.update_xaxes(title_text="매출 대비 R&D 투자 비중 (%)", range=[0, right * 1.15])
    fig = style(fig, max(340, 40 * len(d) + 130), c)
    if guides:
        # y2: 눈에 보이지 않는 보조 세로축(0~1). 막대의 항목 이름 축과 섞이지 않게 호버용 점만 여기에 둔다
        fig.update_layout(margin=dict(l=8, r=24, t=72, b=8), legend=dict(y=1.12),
                          yaxis2=dict(overlaying="y", range=[0, 1], visible=False))
    for i, (name, v, tip) in enumerate(sorted(guides, key=lambda g: g[1])):
        # layer="below": 선을 막대·값 라벨 아래에 깔아 라벨(6.4% 등)을 가리지 않게 한다
        fig.add_vline(x=v, layer="below", line=dict(color=c["muted"], width=1.5, dash="dot"))
        # 점선(도형)은 마우스에 반응하지 않으므로, 선 위에 투명한 점을 촘촘히 깔아 선 어디에 올려도 툴팁이 뜨게 한다
        fig.add_trace(go.Scatter(
            x=[v] * 40, y=list(np.linspace(0, 1, 40)), yaxis="y2", mode="markers", showlegend=False,
            marker=dict(size=14, color="rgba(0,0,0,0)"),
            hovertemplate=f"<b>{name} {v:.1f}%</b><br>{tip}<extra></extra>",
        ))
        # 값이 작은 선의 이름은 왼쪽, 큰 선은 오른쪽에 적어 두 이름이 겹치지 않게 한다
        fig.add_annotation(
            x=v, y=1, yref="paper", yanchor="bottom", showarrow=False,
            xanchor="right" if i == 0 else "left", text=f"{name} {v:.1f}%",
            hovertext=tip, font=dict(size=12, color=c["ink"]),
        )
    return fig


def relax(values, gap, lo, hi):
    """라벨 위치를 위아래로 조금씩 밀어서 서로 gap 이상 떨어지게 한다."""
    pos = list(values)
    for _ in range(300):
        order = sorted(range(len(pos)), key=lambda i: pos[i])
        moved = False
        for a, b in zip(order, order[1:]):
            overlap = gap - (pos[b] - pos[a])
            if overlap > 1e-6:
                pos[a] -= overlap / 2
                pos[b] += overlap / 2
                moved = True
        pos = [min(max(p, lo), hi) for p in pos]
        if not moved:
            break
    return pos


def trend_chart(d, c):
    height = 640
    plot_px = height - 48 - 8 - 40  # 위·아래 여백과 x축 영역을 뺀 대략적인 그래프 높이(px)
    label_px = 24                    # 라벨 한 줄이 차지하는 높이(px)

    fig = go.Figure()
    years = [int(y) for y in sorted(d["year"].unique())]
    top = float(d["rnd_ratio"].max()) * 1.08
    unit = top / plot_px             # 1px이 나타내는 값(%)
    shown, count, ends = set(), {}, []
    # 범례 순서가 다른 탭(장비 -> 부품·후공정 -> 소재)과 같아지도록 구분 순서대로 정렬해서 그린다
    # (sorted는 같은 구분 안에서는 원래 순서를 그대로 유지한다)
    by_company = sorted(d.sort_values("year").groupby("label", sort=False),
                        key=lambda kv: GROUPS.index(kv[1]["group"].iloc[0]))
    for label, s in by_company:
        g = s["group"].iloc[0]
        k = count.get(g, 0)
        count[g] = k + 1
        fig.add_trace(go.Scatter(
            x=s["year"], y=s["rnd_ratio"], mode="lines+markers",
            name=g, legendgroup=g, showlegend=g not in shown,
            line=dict(color=c[g], width=2),
            marker=dict(size=9, symbol=SYMBOLS[k % len(SYMBOLS)], color=c[g],
                        line=dict(color=c["surface"], width=2)),
            hovertemplate=f"<b>{label}</b> ({g})<br>%{{x}}년  R&D 비중 %{{y:.1f}}%<extra></extra>",
        ))
        shown.add(g)
        ends.append((label, float(s["rnd_ratio"].iloc[-1])))

    gap = label_px * unit
    placed = relax([v for _, v in ends], gap, lo=gap / 2, hi=top - gap / 2)
    for (label, y), p in zip(ends, placed):
        # 선 끝(x, y)에서 안내선을 그어 겹치지 않게 옮긴 자리(p)에 이름을 적는다
        fig.add_annotation(
            x=years[-1], y=y, ax=40, ay=-(p - y) / unit, axref="pixel", ayref="pixel",
            text=label, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=c["muted"],
            standoff=6, xanchor="left", font=dict(size=12, color=c["ink"]),
        )
    fig.update_xaxes(tickmode="array", tickvals=years, ticktext=[str(y) for y in years],
                     range=[years[0] - 0.25, years[-1] + 0.5], showgrid=False)
    fig.update_yaxes(title_text="매출 대비 R&D 투자 비중 (%)", range=[0, top])
    fig = style(fig, height, c)
    # 기업 이름은 그래프 오른쪽 '여백'에 놓이므로, 여백을 넉넉히(100px) 잡아 좁은 화면(모바일)에서도 이름이 잘리지 않게 한다
    fig.update_layout(margin=dict(l=8, r=100, t=48, b=8))
    return fig


def pick_text_positions(xs, ys, labels, w=900, h=420):
    """점마다 라벨을 놓을 방향(위/아래/오른쪽/왼쪽)을 골라, 다른 라벨·점과 겹치지 않게 한다.

    화면 크기를 w x h 픽셀로 가정하고 점 좌표를 픽셀로 바꿔 사각형끼리 겹치는지 본다.
    """
    xmax, ymax = max(xs) * 1.12, max(ys) * 1.18
    pts = [(x / xmax * w, y / ymax * h) for x, y in zip(xs, ys)]
    char_w, line_h, r = 13, 16, 8  # 글자 폭, 글줄 높이, 점 반지름(px)

    def box(i, pos):
        px, py = pts[i]
        bw = char_w * len(labels[i])
        if pos == "top center":
            return (px - bw / 2, py + r, px + bw / 2, py + r + line_h)
        if pos == "bottom center":
            return (px - bw / 2, py - r - line_h, px + bw / 2, py - r)
        if pos == "middle right":
            return (px + r, py - line_h / 2, px + r + bw, py + line_h / 2)
        return (px - r - bw, py - line_h / 2, px - r, py + line_h / 2)  # middle left

    def hit(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    dots = [(px - r, py - r, px + r, py + r) for px, py in pts]
    placed, result = {}, [None] * len(pts)
    for i in sorted(range(len(pts)), key=lambda k: pts[k][0]):  # 왼쪽 점부터 차례로
        best, best_score = "top center", None
        for pos in ("top center", "bottom center", "middle right", "middle left"):
            b = box(i, pos)
            score = sum(hit(b, o) for j, o in placed.items()) + sum(hit(b, dots[j]) for j in range(len(pts)) if j != i)
            if b[0] < 0 or b[2] > w or b[1] < 0:  # 그래프 밖(왼쪽·오른쪽·아래)으로 나가는 위치는 감점
                score += 1
            if best_score is None or score < best_score:
                best, best_score = pos, score
            if score == 0:
                break
        result[i] = best
        placed[i] = box(i, best)
    return result


def scatter_chart(d, c):
    fig = go.Figure()
    d = d.reset_index(drop=True)
    positions = pick_text_positions(list(d["revenue_eok"]), list(d["rnd_ratio"]), list(d["label"]))
    d["text_pos"] = positions
    for g in GROUPS:
        s = d[d["group"] == g]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["revenue_eok"], y=s["rnd_ratio"], mode="markers+text", name=g,
            text=s["label"], textposition=list(s["text_pos"]), textfont=dict(color=c["ink"], size=12),
            marker=dict(size=13, color=c[g], line=dict(color=c["surface"], width=2)),
            customdata=s[["process"]].to_numpy(),
            hovertemplate=(
                "<b>%{text}</b><br>공정: %{customdata[0]}"
                "<br>매출: %{x:,.0f}억원<br>R&D 비중: %{y:.1f}%<extra></extra>"
            ),
        ))
    # 라벨이 그래프 밖으로 잘리지 않도록 위·오른쪽에 여유를 둔다 (위 pick_text_positions의 계산과 같은 값)
    # x축 왼쪽 끝을 살짝 음수로 둬서, 매출이 작은 기업(파크시스템스 등)의 이름이 왼쪽에서 잘리지 않게 한다
    xmax = d["revenue_eok"].max() * 1.12
    # 오른쪽 끝 점(하나마이크론 등)의 이름이 좁은 화면에서 잘리지 않도록 오른쪽도 조금 더 여유를 둔다
    fig.update_xaxes(title_text="매출액 (억 원)", range=[-0.08 * xmax, xmax * 1.1], tickformat=",")
    fig.update_yaxes(title_text="매출 대비 R&D 투자 비중 (%)", range=[0, d["rnd_ratio"].max() * 1.18])
    return style(fig, 520, c)


def capex_chart(cx, c):
    """고객사 설비투자 증가율과 장비사 매출 증가율을 한 그래프에 겹쳐 그린다."""
    d = cx.dropna(subset=["장비사 매출 합계 YoY"])  # 첫 해(2021)는 전년 값이 없어 뺀다
    years = [int(y) for y in d["year"]]
    # (이름, 열, 색, 선 모양, 마커) - 장비사는 파랑(대시보드의 '장비' 색), 고객사 둘은 회색 계열을 선 모양으로 구분
    series = [
        ("장비사 5곳 매출 합계", "장비사 매출 합계 YoY", c["장비"], "solid", "circle"),
        ("삼성전자 설비투자", "삼성전자 설비투자 YoY", c["muted"], "dash", "square-open"),
        ("SK하이닉스 설비투자", "SK하이닉스 설비투자 YoY", c["ink"], "solid", "diamond"),
    ]
    fig = go.Figure()
    for name, col, color, dash, symbol in series:
        fig.add_trace(go.Scatter(
            x=years, y=d[col], mode="lines+markers", name=name,
            line=dict(color=color, width=3 if "장비사" in name else 2, dash=dash),
            marker=mk(symbol, color, 11, c),
            hovertemplate=f"<b>{name}</b><br>%{{x}}년  전년 대비 %{{y:+.1f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=c["muted"], width=1))
    fig.update_xaxes(tickmode="array", tickvals=years, ticktext=[str(y) for y in years], showgrid=False)
    fig.update_yaxes(title_text="전년 대비 증가율 (%)", zeroline=False)
    return style(fig, 460, c)


def samsung_spike_note(q):
    """삼성전자 마지막 분기 증가율이 100%를 넘으면, 기저효과가 아니라는 점과 이유를 모른다는 점을 CSV 값으로 알려 준다.

    마지막 분기 값이 작년·재작성의 영향을 받는지는 비교 대상(작년 같은 분기, 2년 전 같은 분기)을 같이 보여 주는 것으로 대신한다.
    """
    col = "삼성전자 설비투자"
    if col not in q.columns or len(q) < 9:
        return ""
    now, prev1, prev2 = q[col].iloc[-1], q[col].iloc[-5], q[col].iloc[-9]
    if not (prev1 > 0 and prev2 > 0 and now / prev1 - 1 > 1):
        return ""
    return (f"삼성전자 {q['quarter'].iloc[-1]} 증가율({now / prev1 - 1:+.0%})은 이 분기 투자({now / 1e12:.1f}조원)가 "
            f"2년 전 같은 분기의 {now / prev2:.1f}배라서 나온 값입니다. 이유는 확인하지 못했습니다.")


def quarterly_chart(q, c):
    """분기별로 '작년 같은 분기 대비' 증가율을 겹쳐 그린다 (계절 흐름을 없애려고 전분기가 아니라 작년 같은 분기와 비교).

    SK하이닉스 2024Q4처럼 혼자 튀는 값이 y축을 다 차지하지 않도록, 축 범위는 대부분의 값이 들어가게 데이터에서 정하고
    범위를 넘는 점은 '▲ +320% (축 밖)'으로 표시한다. 마우스를 올리면 실제 값이 그대로 나온다.
    """
    cols = [  # 장비사(파랑)를 마지막에 그려서 다른 선에 가려지지 않게 한다. legendrank로 범례 순서는 그대로 유지
        ("삼성전자 설비투자", "삼성전자 설비투자", c["muted"], "dash", "square-open", 2),
        ("SK하이닉스 설비투자", "SK하이닉스 설비투자", c["ink"], "solid", "diamond", 3),
        ("장비사 5곳 매출 합계", "장비사 매출 합계", c["장비"], "solid", "circle", 1),
    ]
    labels = list(q["quarter"])
    ys = {col: (q[col] / q[col].shift(4) - 1) * 100 for _, col, *_ in cols}
    allv = pd.concat(ys.values()).dropna()
    lo = math.floor((allv.min() - 5) / 50) * 50                      # 가장 낮은 값이 들어가는 50 단위
    hi = math.ceil(np.quantile(allv, 0.95) * 1.15 / 50) * 50          # 상위 5%를 뺀 값 기준 50 단위
    clip = allv.max() > hi
    fig = go.Figure()
    for name, col, color, dash, symbol, rank in cols:
        fig.add_trace(go.Scatter(
            x=labels, y=ys[col], mode="lines+markers", name=name, legendrank=rank,
            line=dict(color=color, width=4 if "장비사" in name else 2, dash=dash),
            marker=mk(symbol, color, 8, c),
            hovertemplate=f"<b>{name}</b><br>%{{x}}  작년 같은 분기 대비 %{{y:+.0f}}%<extra></extra>",
        ))
        if clip:
            for lab, v in zip(labels, ys[col]):
                if pd.notna(v) and v > hi:
                    # 어느 분기의 값인지 글자에 적고, 점에서 오른쪽으로 조금 띄워 내려오는 선과 글자가 겹치지 않게 한다
                    fig.add_annotation(x=lab, y=hi, text=f"{lab} ▲ {v:+.0f}% (축 밖)", showarrow=False,
                                       xanchor="left", yanchor="top", xshift=34, font=dict(size=12, color=c["ink"]))
    fig.add_hline(y=0, line=dict(color=c["muted"], width=1))
    fig.update_xaxes(type="category", showgrid=False, tickangle=-45, range=[3.5, len(labels) - 0.5])  # 첫 4분기는 비교할 작년 값이 없다
    fig.update_yaxes(title_text="작년 같은 분기 대비 증가율 (%)", zeroline=False,
                     range=[lo, hi] if clip else None)
    return style(fig, 440, c)


def ci_legend(c):
    """범례에 '세로선 = 95% 범위' 한 줄을 넣기 위한 빈 점(화면에는 점이 안 그려지고 범례만 생긴다)."""
    return go.Scatter(x=[None], y=[None], mode="markers", name="세로선 = 95% 범위", hoverinfo="skip",
                      marker=dict(symbol="line-ns-open", size=12, color=c["muted"], line=dict(width=2, color=c["muted"])))


def lag_chart(lag, c):
    """시차별 상관계수와 95% 신뢰구간. 막대(세로선)가 0을 가로지르면 '관계 없음'과 구분되지 않는다."""
    fig = go.Figure()
    # (CSV 속 이름, 범례 이름, 색, 마커, 선, 좌우 어긋남) - 같은 시차의 세 점이 겹치지 않게 조금씩 옆으로 민다
    series = [
        ("SK하이닉스", "SK하이닉스 설비투자", c["ink"], "diamond", None, -0.12),
        ("고객사 합계", "고객사 합계(삼성+하이닉스)", c["total"], "x", "dashdot", 0.0),
        ("삼성전자", "삼성전자 설비투자", c["muted"], "square-open", None, 0.12),
    ]
    for key, name, color, symbol, dash, dx in series:
        s = lag[lag["고객사 투자"] == key]
        if s.empty:  # 낡은 lag_corr.csv에 합계 행이 없어도 나머지는 그린다
            continue
        has_perm = "순열p" in s.columns  # 순열p 열이 없는 낡은 CSV에서도 죽지 않게 한다 (없으면 hover에서 뺀다)
        perm_col = s["순열p"] if has_perm else 0.0
        fig.add_trace(go.Scatter(
            x=s["시차(분기)"] + dx, y=s["상관계수"], mode="lines+markers" if dash else "markers", name=name,
            line=dict(color=color, width=1.5, dash=dash) if dash else None,
            marker=mk(symbol, color, 12, c),
            error_y=dict(type="data", symmetric=False, array=s["상한"] - s["상관계수"],
                         arrayminus=s["상관계수"] - s["하한"], color=color, thickness=1.6, width=5),
            customdata=s[["표본수", "하한", "상한"]].assign(순열p=perm_col).to_numpy(),
            hovertemplate=(f"<b>{name}</b><br>표본 %{{customdata[0]}}개<br>"
                           "상관계수 %{y:+.2f} (95% 범위 %{customdata[1]:+.2f} ~ %{customdata[2]:+.2f})"
                           + ("<br>순열p %{customdata[3]:.2f}" if has_perm else "") + "<extra></extra>"),
        ))
    fig.add_trace(ci_legend(c))
    fig.add_hline(y=0, line=dict(color=c["muted"], width=1))
    lags = sorted(lag["시차(분기)"].unique())
    fig.update_xaxes(tickmode="array", tickvals=lags,
                     ticktext=["같은 분기" if v == 0 else f"{v}분기 뒤" for v in lags],
                     title_text="고객사 투자 후 장비사 매출이 반응하기까지의 시차", showgrid=False)
    fig.update_yaxes(title_text="같이 움직이는 정도 (+1 같이, 0 무관, -1 반대)", range=[-1, 1], zeroline=False)
    return style(fig, 400, c)


def exposure_chart(ex, c):
    """장비사별로 '고객 투자 증가율과 내 매출 증가율의 상관'을 고객별로 점과 신뢰구간으로 그린다."""
    fig = go.Figure()
    n = len(ex)
    for name, color, symbol, dx in (("SK하이닉스", c["ink"], "diamond", -0.08), ("삼성전자", c["muted"], "square-open", 0.08)):
        r, lo, hi = ex[f"{name} 상관"], ex[f"{name} 하한"], ex[f"{name} 상한"]
        fig.add_trace(go.Scatter(
            x=[i + dx for i in range(n)], y=r, mode="markers", name=f"{name} 설비투자",
            marker=mk(symbol, color, 12, c),
            error_y=dict(type="data", symmetric=False, array=hi - r, arrayminus=r - lo, color=color, thickness=1.6, width=5),
            customdata=list(zip(ex["기업"], lo, hi)),
            hovertemplate=(f"<b>%{{customdata[0]}}</b> × {name}<br>상관계수 %{{y:+.2f}} "
                           "(95% 범위 %{customdata[1]:+.2f} ~ %{customdata[2]:+.2f})<extra></extra>"),
        ))
    fig.add_trace(ci_legend(c))
    fig.add_hline(y=0, line=dict(color=c["muted"], width=1))
    # 좁은 화면에서 이름 5개가 가로로 겹치면 Plotly가 자동으로 일부를 생략한다 -> 각도를 고정(-45도)하고 여백을 자동으로 늘려 모두 보이게 한다
    fig.update_xaxes(tickmode="array", tickvals=list(range(n)), ticktext=list(ex["기업"]), showgrid=False,
                     tickangle=-45, automargin=True, tickfont=dict(size=12))
    fig.update_yaxes(title_text="같이 움직이는 정도 (+1 같이, 0 무관, -1 반대)", range=[-1, 1], zeroline=False)
    return style(fig, 400, c)


def show(fig):
    st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")


# ---------- 설명 문장 만들기 (숫자는 CSV에서 읽어 와서 문장에 넣는다) ----------
def pick(lag, who, lg):
    """lag_corr.csv에서 (고객사, 시차) 한 줄을 꺼낸다. 없으면 IndexError (guard가 안내로 바꾼다)."""
    s = lag[(lag["고객사 투자"] == who) & (lag["시차(분기)"] == lg)]
    if s.empty:
        raise IndexError(f"lag_corr.csv에 {who} 시차 {lg} 행이 없습니다")
    return s.iloc[0]


def span(r):
    """'95% 범위 a ~ b' 글자. ~ 가 두 번 나오면 마크다운이 취소선으로 바꾸므로 \\~ 로 막아 둔다."""
    return f"95% 범위 {r['하한']:+.2f} \\~ {r['상한']:+.2f}"


def perm(r):
    """순열p 열이 있을 때만 붙인다."""
    return f", 순열p {r['순열p']:.2f}" if "순열p" in r.index else ""


def yoy4(s):
    """작년 같은 분기 대비 증가율(%)."""
    return (s / s.shift(4) - 1) * 100


def lag_bullets(lag, q):
    skh = [pick(lag, "SK하이닉스", n) for n in range(5)]
    sam = [pick(lag, "삼성전자", n) for n in range(5)]
    strongest = lag.loc[lag["상관계수"].abs().idxmax()]
    # 한 곳씩 빼고 다시 계산했을 때 SK하이닉스 같은 분기 상관이 얼마나 흔들리는지
    names5 = [col[:-3] for col in q.columns if col.endswith(" 매출") and not col.startswith("장비사")]
    sens = []
    for drop in names5:
        tot = q[[f"{n} 매출" for n in names5 if n != drop]].sum(axis=1)
        sens.append(yoy4(q["SK하이닉스 설비투자"]).corr(yoy4(tot)))
    ac_c = yoy4(q["SK하이닉스 설비투자"]).dropna().autocorr(4)
    ac_r = yoy4(q["장비사 매출 합계"]).dropna().autocorr(4)
    # 접힌 '자세히' 칸에 들어가는 내용만 둔다 (결론 3줄은 lag_cards가 위에 따로 보여 준다)
    return (
        f"- 차트에서 가장 센 값은 **{strongest['고객사 투자']} 투자 {int(strongest['시차(분기)'])}분기 뒤 {strongest['상관계수']:+.2f}**"
        f"({span(strongest)}{perm(strongest)})이고, 음수입니다. "
        "**추정**: 투자가 늘면 매출이 줄어드는 진짜 관계라기보다, 작년 같은 분기와 비교하는 방식 때문에 생긴 착시(작년 수치가 유난히 높거나 낮았던 영향)와 약 1년 주기의 업황 때문일 가능성이 있습니다. "
        f"근거: 4분기 전 값과의 상관이 SK하이닉스 투자 {ac_c:+.2f}, 장비사 매출 {ac_r:+.2f}로 둘 다 음수입니다. "
        "다만 표본이 15개에서 19개뿐이라 근거가 약하고 직접 검증하지는 못했습니다. 같은 논리가 위의 양(+)의 상관에도 적용될 수 있어서 두 값 모두 조심해서 읽어야 합니다.\n"
        f"- 장비사를 한 곳씩 빼고 다시 계산해도 SK하이닉스 같은 분기 상관은 {min(sens):+.2f}에서 {max(sens):+.2f} 사이로, 한 회사 때문에 나온 결과는 아닙니다."
    )


def lag_cards(lag):
    """시차 상관 탭 맨 위에 보여 줄 결론 3줄. 숫자는 lag_corr.csv에서 읽고, 방향이 달라지면 문장도 바뀌게 조건을 둔다."""
    skh = [pick(lag, "SK하이닉스", n) for n in range(5)]
    sam = [pick(lag, "삼성전자", n) for n in range(5)]
    s0 = skh[0]
    together = ("**같은 분기에는 같이 움직이는 경향이 보입니다(확정은 어렵습니다).**" if s0["하한"] > 0 else "**같은 분기에도 뚜렷한 관계는 보이지 않습니다.**")
    fading = skh[1]["상관계수"] < s0["상관계수"] and skh[2]["상관계수"] < skh[1]["상관계수"]
    lead = ("**투자가 매출보다 앞선다는 증거는 약합니다.** "
            f"상관이 같은 분기 {s0['상관계수']:+.2f}, 1분기 뒤 {skh[1]['상관계수']:+.2f}, 2분기 뒤 {skh[2]['상관계수']:+.2f}로 시차가 길수록 약해집니다."
            if fading else "**시차에 따른 변화는 일정하지 않습니다.** 세부 값은 차트에서 확인하세요.")
    sam_pos = any(r["하한"] > 0 for r in sam)
    sam_txt = ("**삼성전자는 시차를 늘려도 같이 움직이지 않았습니다.** " if not sam_pos else "**삼성전자는 일부 시차에서 양의 관계가 보입니다.** ")
    return [
        f"{together} SK하이닉스 투자와 장비사 매출의 상관 {s0['상관계수']:+.2f} ({span(s0)}).",
        lead,
        f"{sam_txt}같은 분기 상관 {sam[0]['상관계수']:+.2f} ({span(sam[0])}).",
    ]


def lag_warning(lag):
    return (
        "분기 숫자가 서로 닮아 95% 범위가 실제보다 좁게 나옵니다. "
        f"{len(lag)}개 조합을 한꺼번에 봐서 우연히 눈에 띄는 값이 섞일 수 있습니다. "
        "**상관은 원인을 증명하지 않습니다.**"
    )


def exposure_bullets(ex):
    hi_row = ex.sort_values("SK하이닉스 상관").iloc[-1]
    lo_row = ex.sort_values("SK하이닉스 상관").iloc[0]
    # 고객 집중도(상위 2개 고객 비중) 순위와 SK하이닉스 상관 순위가 서로 다른 기업을 찾는다
    rk_share = ex["상위 2개 고객 비중(%)"].rank(method="min")
    rk_corr = ex["SK하이닉스 상관"].rank(method="min")
    off = list(ex.loc[rk_share != rk_corr, "기업"])
    jus = ex[ex["기업"] == "주성엔지니어링"]
    return (
        f"- 상관이 가장 높은 곳은 {hi_row['기업']}({hi_row['SK하이닉스 상관']:+.2f})입니다. "
        f"공시 계약의 {hi_row['SK하이닉스 계약 비중(%)']:.0f}%가 SK하이닉스이고 상위 2개 고객이 매출의 {hi_row['상위 2개 고객 비중(%)']:.0f}%입니다. "
        f"가장 낮은 곳은 {lo_row['기업']}({lo_row['SK하이닉스 상관']:+.2f})이며 상위 1개 고객이 {lo_row['상위 1개 고객 비중(%)']:.0f}%뿐입니다. "
        "두 곳 모두 가설과 방향이 같습니다.\n"
        "- 다만 맞지 않는 회사도 있습니다"
        + (f"({', '.join(off)})" if off else "") + ". "
        + (f"주성엔지니어링은 공시 계약의 {jus.iloc[0]['SK하이닉스 계약 비중(%)']:.0f}%가 SK하이닉스인데도 상관은 {jus.iloc[0]['SK하이닉스 상관']:+.2f}로 낮습니다. "
           if not jus.empty and pd.notna(jus.iloc[0]["SK하이닉스 계약 비중(%)"]) else "")
        + "회사가 5곳뿐이라 확정하기 어렵습니다."
    )


def exposure_warning():
    """계약 공시 요약(기간·건수)을 contracts.csv에서 읽어 문장을 만든다. 파일이 없거나 모양이 다르면 그 부분만 뺀다."""
    notes = ""
    ct = load_csv(CONTRACTS_PATH)
    if ct is not None and not ct.empty:
        try:
            last = ct.groupby("name")["contract_date"].max()
            when = lambda n: f"{int(last[n][:4])}년 {int(last[n][5:7])}월"
            cnt = ct.groupby("name").size()
            notes = f"유진테크 {when('유진테크')}까지, 주성엔지니어링 {when('주성엔지니어링')}까지, 원익IPS {cnt['원익IPS']}건·파크시스템스 {cnt['파크시스템스']}건뿐"
        except (KeyError, ValueError, TypeError):
            notes = ""
    return (
        "사업보고서의 주요 고객은 'A사, B사'로 가려져 있어 삼성전자·SK하이닉스인지 알 수 없습니다. "
        "계약 공시는 큰 계약만 나오고 기간도 회사마다 다릅니다"
        + (f"({notes}). " if notes else ". ")
        + "그래서 원익IPS·파크시스템스는 계약 비중을 계산하지 않았습니다."
    )


def annual_text(cx):
    both = cx.dropna(subset=["장비사 매출 합계 YoY"])
    corr = {n: both[f"{n} 설비투자 YoY"].corr(both["장비사 매출 합계 YoY"]) for n in ("삼성전자", "SK하이닉스")}
    yr = both.set_index("year")  # 연도로 한 줄씩 꺼내 문장에 숫자를 넣는다

    def yv(y, col):
        return yr.loc[y, col] if y in yr.index else float("nan")

    return (
        f"증가율 상관계수 (표본 {len(both)}개, 참고용): "
        f"SK하이닉스 {corr['SK하이닉스']:+.2f} / 삼성전자 {corr['삼성전자']:+.2f}  \n"
        "- SK하이닉스가 투자를 크게 줄인 2023년에 장비사 매출도 줄었고, 크게 늘린 2024년에는 함께 늘었습니다. "
        f"다만 2022년(투자 {yv(2022, 'SK하이닉스 설비투자 YoY'):+.1f}%, 장비사 매출 {yv(2022, '장비사 매출 합계 YoY'):+.1f}%)과 "
        f"2025년(투자 {yv(2025, 'SK하이닉스 설비투자 YoY'):+.1f}%, 장비사 매출 {yv(2025, '장비사 매출 합계 YoY'):+.1f}%)에는 "
        "투자가 늘었는데 매출은 줄거나 거의 그대로였습니다. 수주에서 매출 인식까지 수개월 걸려 시점이 어긋날 수 있습니다.\n"
        f"- 삼성전자는 {corr['삼성전자']:+.2f}로 음의 값이지만 표본이 {len(both)}개라 해석하지 않습니다."
    )


TERMS = (
    "- **설비투자(CAPEX)**: 공장·장비를 사는 데 쓴 돈(현금흐름표 '유형자산의 취득').\n"
    "- **장비사 5곳**: 원익IPS, 주성엔지니어링, 유진테크, 한미반도체, 파크시스템스.\n"
    "- **증가율(YoY)**: 작년 같은 분기와 비교한 증가율.\n"
    "- **상관계수**: 두 증가율이 같이 움직이는 정도. +1 같이, 0 무관, -1 반대.\n"
    "- **95% 범위**: 상관계수가 이 안에 있을 것이라고 보는 구간. 0을 포함하면 관계가 있다고 말하기 어렵습니다.\n"
    "- **시차**: 고객사 투자 뒤 몇 분기 지나서 매출과 비교했는지.\n"
    "- **순열p**: 짝을 섞어도 이만큼 센 상관이 나올 확률. 작을수록 우연일 가능성이 낮습니다."
)
CI_CAPTION = "점이 위쪽일수록 같이 움직입니다. 세로선(95% 범위)이 0선을 지나면 관계가 있다고 말하기 어렵습니다."


# ---------- 화면: 고객사 투자 탭 ----------
def render_summary(q, lag):
    """탭 맨 위 요약: 숫자 카드 3개와 한 줄 결론(한계 안내 포함)."""
    s0, h0 = pick(lag, "삼성전자", 0), pick(lag, "SK하이닉스", 0)
    with st.container(key="kpis_customer"):  # key는 모바일에서 카드를 가로로 유지하는 CSS가 이 상자만 고르게 하려는 이름표
        k1, k2, k3 = st.columns(3)
        k1.metric("SK하이닉스 상관", f"{h0['상관계수']:+.2f}",
                  f"95% 범위 {h0['하한']:+.2f}에서 {h0['상한']:+.2f}", delta_color="off", delta_arrow="off")
        k2.metric("삼성전자 상관", f"{s0['상관계수']:+.2f}",
                  f"95% 범위 {s0['하한']:+.2f}에서 {s0['상한']:+.2f}", delta_color="off", delta_arrow="off")
        k3.metric("분석 기간", f"{len(q)}개 분기", f"{q['quarter'].iloc[0]}부터 {q['quarter'].iloc[-1]}까지",
                  delta_color="off", delta_arrow="off")
    # 95% 범위는 실제보다 좁게 나오므로, 순열p가 있으면 함께 보고 둘 다 통과해야 '경향'이라고 쓴다
    has_p = "순열p" in h0.index and pd.notna(h0["순열p"])
    ok = h0["하한"] > 0 and (not has_p or h0["순열p"] < 0.05)
    trend = ("함께 움직이는 경향이 있었습니다" if ok else
             "함께 움직이는 경향이 보이지만 확정하기는 어렵습니다" if h0["하한"] > 0 else "뚜렷한 관계가 보이지 않았습니다")
    # 결론은 한 문장, 단서는 한 줄 캡션으로 나눈다 (순열p와 '가장 센 값'은 '시차 상관' 탭에 있어 여기서는 뺐다)
    st.markdown(f"**한 줄 요약**: SK하이닉스 설비투자와 장비사 매출은 같은 분기에 {trend}. (위 숫자는 모두 같은 분기 기준)")
    st.caption("표본이 적어 '같이 움직인다' 정도로만 읽어 주세요.")


def render_customer_tab(c):
    # collect_quarterly.py -> quarterly.csv, analyze_quarterly.py -> lag_corr.csv,
    # analyze_exposure.py -> exposure.csv, analyze_capex.py -> capex.csv
    st.subheader("고객사가 투자를 늘리면 장비사 매출도 늘까?")
    q, lag = load_csv(QUARTERLY_PATH), load_csv(LAG_PATH)
    ex, cx = load_csv(EXPOSURE_PATH), load_csv(CAPEX_PATH)

    if q is not None and lag is not None and not q.empty and not lag.empty:
        with guard("요약:"):
            render_summary(q, lag)
    with st.expander("용어 설명"):
        st.markdown(TERMS)

    # 서브탭은 3개: 핵심(분기 흐름, 회사별)과 통계 세부(시차 상관, 연 단위)로 나눈다
    t_flow, t_firm, t_more = st.tabs(["분기 흐름", "회사별", "통계 자세히"])

    with t_flow:
        if need(q, QUARTERLY_PATH, "python collect_quarterly.py"):
            with guard("quarterly.csv:"):
                st.caption("파란 굵은 선(장비사 매출)이 고객사 투자 선과 같은 방향으로 움직이는지 보세요. 모두 작년 같은 분기 대비 증가율입니다.")
                show(quarterly_chart(q, c))
                footnote([
                    "작년 수준이 낮았던 분기는 증가율이 크게 나옵니다(기저효과).",
                    samsung_spike_note(q),
                    "삼성전자는 전사 연결 기준이라 디스플레이·하만 투자가 섞여 있습니다.",
                    f"{q['quarter'].iloc[0]}부터 {q['quarter'].iloc[-1]}까지 {len(q)}개 분기이고, 왼쪽 필터와 무관합니다.",
                ])

    with t_firm:
        if need(ex, EXPOSURE_PATH, "python analyze_exposure.py"):
            with guard("exposure.csv:"):
                st.caption("가설: 한 고객에 많이 기대는 장비사일수록 그 고객의 투자와 매출이 같이 움직인다.")
                st.markdown(exposure_bullets(ex))  # 결론을 차트 위에 둔다
                st.caption(CI_CAPTION)
                show(exposure_chart(ex, c))
                cc = st.column_config
                main_tbl = pd.DataFrame({
                    "기업": ex["기업"],
                    "상위1": ex["상위 1개 고객 비중(%)"],
                    "상위2": ex["상위 2개 고객 비중(%)"],
                    "SK 상관": ex["SK하이닉스 상관"],
                    "삼성 상관": ex["삼성전자 상관"],
                })
                # 열 폭을 좁게 잡아 모바일(375px)에서도 가로 스크롤이 거의 없게 한다
                st.dataframe(main_tbl, hide_index=True, width="stretch", column_config={
                    "기업": cc.Column(width=106),
                    "상위1": cc.NumberColumn(format="%.0f", width=48), "상위2": cc.NumberColumn(format="%.0f", width=48),
                    "SK 상관": cc.NumberColumn(format="%+.2f", width=62), "삼성 상관": cc.NumberColumn(format="%+.2f", width=68),
                })
                st.caption("상위1·2 = 매출 중 상위 1·2개 고객 비중(%, 2025년 사업보고서). 상관 = 그 고객 설비투자와의 같은 분기 상관계수.")
                with st.expander("공시 계약 수, 계약 중 SK하이닉스 몫"):
                    detail = pd.DataFrame({
                        "기업": ex["기업"], "계약 수": ex["공시 계약 수"], "계약 중 SK(%)": ex["SK하이닉스 계약 비중(%)"],
                    })
                    st.dataframe(detail, hide_index=True, width="stretch", column_config={
                        "계약 수": cc.NumberColumn(format="%d"), "계약 중 SK(%)": cc.NumberColumn(format="%.0f"),
                    })
                footnote([exposure_warning()])

    with t_more:
        if need(lag, LAG_PATH, "python analyze_quarterly.py"):
            with guard("lag_corr.csv:"):
                st.markdown("**시차 상관**: 고객사가 투자한 뒤 몇 분기 만에 장비사 매출이 반응하는가")
                for line in lag_cards(lag):
                    st.info(line)
                st.caption(CI_CAPTION)
                show(lag_chart(lag, c))
                if q is not None and not q.empty:
                    with st.expander("음수 값은 왜 나왔나, 한 곳씩 빼 봐도 같은가"):
                        st.markdown(lag_bullets(lag, q))
                footnote([lag_warning(lag)])
        if need(cx, CAPEX_PATH, "python analyze_capex.py"):
            with guard("capex.csv:"):
                with st.expander("연 단위로 보면 (표본 4개, 참고용)"):
                    show(capex_chart(cx, c))
                    st.markdown(annual_text(cx))
                    cc = st.column_config
                    tbl = pd.DataFrame({
                        "연도": cx["year"].astype(str),
                        "삼성전자 투자(조원)": cx["삼성전자 설비투자"] / 1e12,
                        "SK하이닉스 투자(조원)": cx["SK하이닉스 설비투자"] / 1e12,
                        "장비사 매출 합계(억원)": (cx["장비사 매출 합계"] / 1e8).round(0),  # 반올림해야 'localized'가 소수점 없이 쉼표만 붙인다
                        "장비사 매출 YoY(%)": cx["장비사 매출 합계 YoY"].map(lambda v: "-" if pd.isna(v) else f"{v:+.1f}"),  # 첫 해는 '-'
                    })
                    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
                        "삼성전자 투자(조원)": cc.NumberColumn(format="%.1f"),
                        "SK하이닉스 투자(조원)": cc.NumberColumn(format="%.1f"),
                        "장비사 매출 합계(억원)": cc.NumberColumn(format="localized"),
                    })

    # 이 탭 전체에 해당하는 참고사항 (맨 아래에 한 번만)
    footnote([
        "고객사는 삼성전자·SK하이닉스 둘뿐이라 장비사 매출 중 다른 고객에서 나온 부분은 설명하지 못합니다.",
        "표본이 적어 가설을 세우는 수준이며, 상관은 원인을 증명하지 않습니다.",
    ])


# ---------- 화면: 나머지 탭 ----------
def render_kpis(cur, prev, year):
    avg = cur["rnd_ratio"].mean()
    delta = None if prev.empty else f"{avg - prev['rnd_ratio'].mean():+.1f}%p (전년 대비)"
    top = cur.loc[cur["rnd_ratio"].idxmax()]
    low = cur.loc[cur["rnd_ratio"].idxmin()]
    with st.container(key="kpis_main"):
        k1, k2, k3 = st.columns(3)
        k1.metric(f"{year}년 선택 기업 평균", f"{avg:.1f}%", delta, delta_color="off")  # 필터에 따라 달라지는 평균임을 라벨로 알린다
        k2.metric("가장 높은 기업", f"{top['rnd_ratio']:.1f}%", top["name"], delta_color="off", delta_arrow="off")
        k3.metric("가장 낮은 기업", f"{low['rnd_ratio']:.1f}%", low["name"], delta_color="off", delta_arrow="off")


# 모바일(폭 640px 이하)에서는 st.columns가 세로로 쌓여 카드 3개가 첫 화면을 다 차지한다.
# 그래서 카드 상자(kpis_*)만 가로 한 줄을 유지하고 글자를 줄인다.
MOBILE_CSS = """
<style>
@media (max-width: 640px) {
  h1 { font-size: 1.9rem !important; line-height: 1.25 !important; }
  .st-key-kpis_main [data-testid="stHorizontalBlock"],
  .st-key-kpis_customer [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 0.5rem !important; }
  .st-key-kpis_main [data-testid="stColumn"],
  .st-key-kpis_customer [data-testid="stColumn"] { min-width: 0 !important; flex: 1 1 0 !important; width: auto !important; }
  .st-key-kpis_main [data-testid="stMetricValue"],
  .st-key-kpis_customer [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
  .st-key-kpis_main [data-testid="stMetricLabel"] p,
  .st-key-kpis_customer [data-testid="stMetricLabel"] p { font-size: 0.72rem !important; line-height: 1.2 !important; }
  .st-key-kpis_main [data-testid="stMetricDelta"],
  .st-key-kpis_customer [data-testid="stMetricDelta"] { font-size: 0.68rem !important; }
}
</style>
"""


def headline(df):
    """제목 아래 한 줄 결론. 필터와 무관하게 12개사 전체의 가장 최근 연도 기준이며, 숫자는 모두 master.csv에서 계산한다."""
    ly = int(df["year"].max())
    d = df[df["year"] == ly]
    # 구분 이름으로 직접 꺼낸다 (GROUPS의 순서는 장비, 부품·후공정, 소재라서 순서로 풀면 뒤바뀐다)
    eq, part, mat = d[d["group"] == "장비"], d[d["group"] == "부품·후공정"], d[d["group"] == "소재"]
    if eq.empty or mat.empty or part.empty:
        return
    st.markdown(
        f"**한눈에 보기 ({ly}년, {len(d)}개사 단순평균)**: 장비 {len(eq)}곳 **{eq['rnd_ratio'].mean():.1f}%**, "
        f"소재 **{mat['rnd_ratio'].mean():.1f}%**, 부품·후공정 **{part['rnd_ratio'].mean():.1f}%**."  # 숫자 바로 뒤에 한글을 붙이면 굵게 표시가 깨져서 쉼표·공백으로 끊는다
    )


NOTE_SAMPLE = "12개사 표본이라 평균은 참고용입니다."
NOTE_HIGH = "R&D 비중이 높다고 좋은 것은 아닙니다. 매출이 급증하면 비중은 오히려 낮아질 수 있습니다."


def footnote(lines):
    """탭 맨 아래에 '참고사항'을 회색 작은 글씨로 붙인다. 빈 문장은 건너뛴다."""
    lines = [x for x in lines if x]
    if lines:
        st.caption("**참고사항**\n\n" + "\n".join(f"- {x}" for x in lines))


def definition_table(df, year):
    """R&D를 어떻게 세느냐(정부보조금, 개발비 자산화)에 따라 비중이 얼마나 달라지는지 보여 주는 표.
    사업보고서의 '연구개발비용' 표에서 손으로 옮긴 rnd_breakdown.csv와 master.csv의 매출을 합쳐 계산한다.
    - 같은 기준 비중 = 정부보조금을 빼기 전 총액 ÷ 매출 (자산화한 개발비 포함)
    - 자산화 제외 비중 = 같은 기준에서 개발비(무형자산)로 자산화한 부분을 뺀 비용 처리분 ÷ 매출
    파일이 없거나 해당 연도가 없으면 None."""
    try:
        b = pd.read_csv(BREAKDOWN_PATH, dtype={"stock_code": str}, encoding="utf-8-sig")
    except (OSError, ValueError):
        return None, []
    b = b[b["year"] == year].merge(df[["stock_code", "year", "revenue", "rnd_ratio", "ratio_source"]],
                                   on=["stock_code", "year"], how="inner")
    if b.empty:
        return None, {}
    # '계'가 보조금을 뺀 값이면 보조금을 더해 차감 전 총액으로 맞춘다
    gross = b["report_total_mil_won"] + b["govt_grant_mil_won"].where(b["total_basis"] == "보조금 차감 후", 0)
    rev = b["revenue"] / 1e6  # 원 -> 백만원
    b["same"] = gross / rev * 100
    b["expensed"] = (gross - b["capitalized_mil_won"]) / rev * 100  # 자산화 구분을 모르면 빈 값(NaN)
    b["cap_share"] = b["capitalized_mil_won"] / gross * 100         # 총액 중 자산화한 몫(%). 매출과 무관하다
    # R&D와 매출의 기준이 달라 공시 비율을 쓰는 회사(*)는 R&D÷연결 매출이 틀린 값이라 새 비중을 계산하지 않는다
    b["star"] = b["ratio_source"].str.contains("공시")  # 정렬 전에 열로 저장해 둬야 정렬 뒤에도 행과 짝이 맞는다
    b.loc[b["star"], ["same", "expensed"]] = float("nan")
    b = b.sort_values("rnd_ratio", ascending=False).reset_index(drop=True)
    out = pd.DataFrame({
        "기업": b["name"] + b["star"].map({True: " *", False: ""}),
        "화면 비중(%)": b["rnd_ratio"].round(1),
        "총액 기준": b["total_basis"],
        "정부보조금(억원)": (b["govt_grant_mil_won"] / 100).round(1),
        "자산화 개발비(억원)": (b["capitalized_mil_won"] / 100).round(1),
        "자산화 몫(%)": b["cap_share"].round(0),
        "같은 기준 비중(%)": b["same"].round(1),
        "자산화 제외 비중(%)": b["expensed"].round(1),
    })
    ok = b.dropna(subset=["same"])
    gap = max((ok["same"] - ok["rnd_ratio"]).abs().max(), (ok["same"] - ok["expensed"]).abs().max())
    # 정의를 맞추면 순서가 바뀌는지도 함께 계산해서 문장에 쓴다
    same_order = list(ok.sort_values("rnd_ratio")["name"]) == list(ok.sort_values("same")["name"])
    after = list(b.loc[b["total_basis"] == "보조금 차감 후", "name"])
    cap = b[b["capitalized_mil_won"] > 0].sort_values("cap_share", ascending=False)
    unknown = list(b.loc[b["capitalized_mil_won"].isna(), "name"])
    return out, dict(gap=gap, same_order=same_order, n_ok=len(ok), n=len(b), after=after,
                     cap=[(r["name"], r["cap_share"]) for _, r in cap.iterrows()], unknown=unknown)


def render_company_tab(view, cur, ref, sel, year, flagged, note, c, ref_all=None):
    st.subheader(f"{year}년 매출 대비 R&D 투자 비중")
    peers = peer_ratios(year)
    show(bar_chart(cur, c, ref, peers))
    st.markdown("**공정별 평균 R&D 비중 (%)**")
    tbl = (view.pivot_table(index="process", columns="year", values="rnd_ratio", aggfunc="mean")
           .reindex([p for p in PROCESS_ORDER if p in sel]).round(1))
    tbl.columns = [f"{y}년" for y in tbl.columns]
    tbl.index.name = "공정"  # 맨 왼쪽 머리글이 영어(process)로 나오지 않게 한다
    st.dataframe(tbl.style.format("{:.1f}", na_rep="-"), width="stretch")

    # 공정 필터와 무관하게 12곳이 항상 보이도록 필터 전 전체 데이터(캐시됨)를 쓴다
    deftbl, info = definition_table(load_data(os.path.getmtime(MASTER_PATH)), year)
    if deftbl is not None:
        with st.expander("R&D를 세는 방식이 다르면 비중이 얼마나 달라질까?"):
            st.dataframe(deftbl, hide_index=True, width="stretch", height=35 * (len(deftbl) + 1) + 3)
            cap_txt = ", ".join(f"{n} {s:.0f}%" for n, s in info["cap"])
            st.caption(
                (f"정부보조금을 뺀 값으로 적은 곳: {', '.join(info['after'])}. " if info["after"] else "")
                + (f"개발비로 자산화한 금액이 총액에 포함된 곳(자산화 몫): {cap_txt}. " if cap_txt else "")
                + (f"자산화 구분을 알 수 없는 곳: {', '.join(info['unknown'])}. " if info["unknown"] else "")
                + f"같은 기준으로 맞춘 비중은 `*` 기업을 뺀 {info['n_ok']}곳만 계산했고, 화면 비중과 최대 {info['gap']:.1f}%p 다르며 "
                + ("순위는 그대로입니다." if info["same_order"] else "순위가 일부 바뀝니다.")
            )
    # 점선에 쓴 개별 값과 다른 팹을 한 곳에 모은다 (차트에는 평균만 나온다)
    fabs = fab_table(ref_all)
    base = pd.concat([ref[["name", "rnd_ratio"]].assign(구분="국내 고객사"), peers.assign(구분="글로벌 장비사")])
    if fabs is not None or not base.empty:
        with st.expander("점선에 쓴 개별 값과 다른 팹"):
            if not base.empty:
                b = base.rename(columns={"name": "회사", "rnd_ratio": f"{year}년 R&D 비중(%)"})[["구분", "회사", f"{year}년 R&D 비중(%)"]]
                st.dataframe(b.style.format({f"{year}년 R&D 비중(%)": "{:.1f}"}), hide_index=True, width="stretch",
                             height=35 * (len(b) + 1) + 3)
            if fabs is not None:
                st.markdown("**다른 반도체 제조사(참고)**")
                st.dataframe(fabs, width="stretch", height=35 * (len(fabs) + 1) + 3)  # 줄 수만큼 높이를 줘서 안쪽 스크롤이 생기지 않게 한다
                st.caption("DB하이텍(DART), Micron·Intel·TSMC(SEC) 값이며 회계연도·통화·사업 범위가 달라 직접 비교가 아닙니다. TSMC 2025년은 확인하지 못해 비웠습니다.")

    # 한계는 탭 맨 아래에 회색 작은 글씨로 모은다 (참고사항 느낌)
    lines = []
    if not ref.empty or not peers.empty:
        lines.append("점선은 12개사 평균·순위에 넣지 않은 비교용입니다. 삼성전자는 가전·모바일을 포함한 전사 값이고, 글로벌 장비사는 SEC 공시(미국 회계기준, 회계연도·규모 차이)라 직접 비교가 아닙니다.")
    if flagged:
        lines.append(note)
    lines += [NOTE_SAMPLE, NOTE_HIGH]
    footnote(lines)


def render_trend_tab(view, flagged, note, c):
    st.subheader("연도별 R&D 투자 비중 추이")
    show(trend_chart(view, c))
    st.caption("색은 구분(장비·부품/후공정·소재), 마커 모양과 오른쪽 이름은 기업입니다.")
    footnote([note if flagged else "", NOTE_SAMPLE])


def render_scale_tab(cur, year, flagged, note, c):
    st.subheader(f"{year}년 매출 규모와 R&D 투자 강도")
    show(scatter_chart(cur, c))
    st.caption("매출이 커도 R&D 비중은 낮을 수 있고, 작아도 높을 수 있습니다. 좁은 화면에서는 점을 눌러 이름을 확인하세요.")
    footnote([note if flagged else "", "R&D 정의(개발비 자산화 포함 여부 등)가 회사마다 다를 수 있습니다.", NOTE_SAMPLE])


def yoy_alert(view, limit=100):
    """매출·R&D가 전년보다 limit(%)를 넘게 변한 행을 찾아 안내 문장으로 만든다. 없으면 빈 문자열(footnote가 건너뛴다).
    급변은 오류가 아닐 수도 있지만(한미반도체 매출 급증) 비중 해석을 흔들기 때문에 눈에 띄게 알린다."""
    items = []
    for col, label in (("revenue_yoy", "매출"), ("rnd_yoy", "R&D")):
        for _, r in view[view[col].abs() > limit].sort_values(["name", "year"]).iterrows():
            items.append(f"{r['name']} {int(r['year'])}년 {label} {r[col]:+.0f}%")
    if not items:
        return ""
    return (f"전년 대비 ±{limit}%를 넘는 급변이 있습니다: " + ", ".join(items)
            + ". 오류가 아니라 실제 변화일 수 있지만, 이런 해는 비중이 크게 흔들리니 원문(출처 열)에서 확인하세요.")


def render_data_tab(view, flagged):
    st.subheader("데이터 표")
    if view.empty:
        st.caption("공정을 선택하면 표가 나타납니다.")
    else:
        table = view[["name", "process", "year", "revenue_eok", "rnd_eok", "rnd_ratio",
                      "ratio_source", "revenue_yoy", "rnd_yoy", "vs_process"]].copy()
        table["source"] = table["name"].map(load_links())  # 기업 이름으로 DART 원문 주소를 붙인다
        table["revenue_eok"] = table["revenue_eok"].round(0)
        table["rnd_eok"] = table["rnd_eok"].round(1)
        table["rnd_ratio"] = table["rnd_ratio"].round(1)
        table["year"] = table["year"].astype(str)
        table["ratio_source"] = table["ratio_source"].map(
            {"계산(R&D/매출)": "계산", "공시비율(기준상이)": "공시비율"})
        # 첫 해(2023)는 전년 값이 없고, 공정에 기업이 1개뿐이면 중앙값 대비가 비어 있다 -> 'None' 대신 '-'로 표시
        for col in ["revenue_yoy", "rnd_yoy", "vs_process"]:
            table[col] = table[col].map(lambda v: "-" if pd.isna(v) else f"{v:+.1f}")
        table = table.rename(columns={
            "name": "기업", "process": "공정", "year": "연도",
            "revenue_eok": "매출(억원)", "rnd_eok": "R&D(억원)", "rnd_ratio": "R&D 비중(%)",
            "ratio_source": "산출", "revenue_yoy": "매출 YoY(%)",
            "rnd_yoy": "R&D YoY(%)", "vs_process": "중앙값 대비(%p)", "source": "출처",
        }).sort_values(["기업", "연도"])
        cc = st.column_config
        widths = {
            "기업": cc.Column(width=98, pinned=True),  # 가로로 스크롤해도 기업 이름은 왼쪽에 고정
            "공정": cc.Column(width=72), "연도": cc.Column(width=46),
            "매출(억원)": cc.NumberColumn(width=78, format="localized"),  # 10,097처럼 천 단위 쉼표
            "R&D(억원)": cc.NumberColumn(width=76, format="%.1f"),
            "R&D 비중(%)": cc.NumberColumn(width=88, format="%.1f"),
            "산출": cc.Column(width=66), "매출 YoY(%)": cc.Column(width=82),
            "R&D YoY(%)": cc.Column(width=80),
            "중앙값 대비(%p)": cc.Column(width=96),
            "출처": cc.LinkColumn(display_text="DART 원문"),  # 눌러서 사업보고서 원문(접수번호 주소)으로 이동. 남는 칸을 채운다
        }
        # 표 높이를 행 수에 맞춰 키워서 행이 중간에 잘리거나 안쪽 스크롤이 생기지 않게 한다 (행 35px + 머리글)
        st.dataframe(table, hide_index=True, column_config=widths, height=(len(table) + 1) * 35 + 3, width="stretch")
        st.caption("'공시비율'인 기업(*)은 R&D 비중이 표의 R&D ÷ 매출과 다르게 보일 수 있습니다(R&D와 매출의 기준이 달라 보고서 공시 비율을 썼습니다).")
        st.download_button("CSV 다운로드", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name="semi_rnd_tracker.csv", mime="text/csv")

    # 출처와 계산 방법은 표 바로 아래에 참고사항으로 둔다
    footnote([
        "출처: 금융감독원 Open DART. 매출은 재무제표 API, R&D는 2025 사업보고서 '연구개발활동' 표(사람이 입력)이며, 3개년 값을 모두 이 한 권에서 읽었습니다.",
        "R&D 비중 = 연구개발비용 ÷ 매출액 × 100. 직접 계산한 값이 보고서 공시 비율과 0.5%p 넘게 다르면 공시 비율을 썼습니다.",
        (f"기준이 다른 기업(*): {', '.join(flagged)}. " if flagged else ""),
        yoy_alert(view),
    ])


# ---------- 화면: 전체 ----------
def main():
    st.set_page_config(page_title="반도체 소부장 R&D 트래커", layout="wide")
    c = COLORS[theme_name()]

    if not os.path.exists(MASTER_PATH):
        st.error("data/processed/master.csv 파일이 없습니다. 터미널에서 `python metrics.py`를 먼저 실행하세요.")
        st.stop()
    df = load_data(os.path.getmtime(MASTER_PATH))
    # 고객사(삼성전자·SK하이닉스)는 소부장 12개사와 따로 둔다 (기업 비교 차트의 기준선에만 사용)
    ref_all = df[df["segment"] == "고객사"]
    df = df[df["segment"] != "고객사"]

    st.title("반도체 소부장 R&D 투자 비중 트래커")
    st.caption(f"국내 소부장 12개사 · {int(df['year'].min())}~{int(df['year'].max())}년 사업보고서 · 출처 Open DART · 수집 {COLLECTED_AT}")
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    headline(df)

    # --- 필터 (공정·연도는 '기업 비교'·'추이'·'규모와 비중'·'데이터' 탭에 적용) ---
    with st.sidebar:
        st.header("필터")
        processes = [p for p in PROCESS_ORDER if p in set(df["process"])]
        sel = st.multiselect("공정", processes, default=processes)
        years = sorted(int(y) for y in df["year"].unique())
        year = st.selectbox("연도", years, index=len(years) - 1)
        st.caption("연도는 '기업 비교'와 '규모와 비중'에 적용됩니다.")

    view = df[df["process"].isin(sel)]  # 공정을 전부 해제하면 빈 표가 된다 (각 탭이 안내로 대신한다)
    cur = view[view["year"] == year]
    flagged = sorted(df.loc[df["basis_flag"], "name"].unique())
    note = "\\* 표시: R&D와 매출의 기준이 달라 보고서의 공시 비율을 쓴 기업(비율만 참고)"
    # 안내는 맨 위 경고 한 곳에만 크게 띄우고, 비어 있는 탭에는 눈에 덜 띄는 한 줄만 둔다 (같은 말이 5번 반복되지 않게)
    pick_prompt = "왼쪽 필터에서 공정을 하나 이상 선택하세요."
    empty_hint = "공정을 선택하면 이 탭에 내용이 나타납니다."

    if cur.empty:
        st.warning(pick_prompt)
    else:
        render_kpis(cur, view[view["year"] == year - 1], year)

    # 핵심 3개(기업 비교·추이·데이터)를 앞에, 추가 분석 2개를 뒤에 둔다
    tab_company, tab_trend, tab_data, tab_scale, tab_customer = st.tabs(
        ["기업 비교", "추이", "데이터", "규모와 비중", "고객사 투자"])
    with tab_company:
        if cur.empty:
            st.caption(empty_hint)
        else:
            render_company_tab(view, cur, ref_all[ref_all["year"] == year], sel, year, flagged, note, c, ref_all)
    with tab_trend:
        if view.empty:
            st.caption(empty_hint)
        else:
            render_trend_tab(view, flagged, note, c)
    with tab_data:
        render_data_tab(view, flagged)
    with tab_scale:
        if cur.empty:
            st.caption(empty_hint)
        else:
            render_scale_tab(cur, year, flagged, note, c)
    with tab_customer:
        render_customer_tab(c)

    # 모든 탭 아래에 항상 보이는 안내문 (투자 권유가 아님을 밝힌다)
    st.divider()
    st.caption(DISCLAIMER)


main()
