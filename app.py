"""반도체 소부장 R&D 투자 비중 트래커 - Streamlit 대시보드

실행:  streamlit run app.py
필요:  data/processed/master.csv  (python metrics.py 로 생성)
"""
import inspect
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MASTER_PATH = os.path.join(BASE, "data", "processed", "master.csv")

PROCESS_ORDER = ["증착/식각", "패키징", "테스트", "검사/계측", "소재"]

# 색은 '구분' 3가지에만 쓴다 (장비 / 부품·후공정 / 소재).
# 3색은 색각 이상 기준으로 검증된 팔레트의 앞 3칸이라, 어떤 차트에서도 서로 구분된다.
GROUP_OF_SEGMENT = {"장비": "장비", "부품": "부품·후공정", "후공정": "부품·후공정", "소재": "소재"}
GROUPS = ["장비", "부품·후공정", "소재"]
COLORS = {
    "light": {
        "장비": "#2a78d6", "부품·후공정": "#eb6834", "소재": "#1baf7a",
        "ink": "#52514e", "grid": "#e1e0d9", "muted": "#898781", "surface": "#ffffff",
    },
    "dark": {
        "장비": "#3987e5", "부품·후공정": "#d95926", "소재": "#199e70",
        "ink": "#c3c2b7", "grid": "#2c2c2a", "muted": "#898781", "surface": "#0e1117",
    },
}
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "pentagon"]


# ---------- Streamlit 버전 차이를 흡수하는 작은 도우미 ----------
def _stretch(func):
    """차트/표를 가로로 꽉 채우는 옵션 이름이 버전마다 달라서 자동으로 고른다."""
    params = inspect.signature(func).parameters
    return {"width": "stretch"} if "width" in params else {"use_container_width": True}


PLOT_KW = _stretch(st.plotly_chart)
DF_KW = _stretch(st.dataframe)


def theme_name():
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


# ---------- 데이터 ----------
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


# ---------- 차트 ----------
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


def bar_chart(d, c):
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
    fig.update_xaxes(title_text="매출 대비 R&D 투자 비중 (%)", range=[0, d["rnd_ratio"].max() * 1.15])
    return style(fig, max(340, 40 * len(d) + 110), c)


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
            x=years[-1], y=y, ax=48, ay=-(p - y) / unit, axref="pixel", ayref="pixel",
            text=label, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=c["muted"],
            standoff=6, xanchor="left", font=dict(size=12, color=c["ink"]),
        )
    fig.update_xaxes(tickmode="array", tickvals=years, ticktext=[str(y) for y in years],
                     range=[years[0] - 0.25, years[-1] + 0.9], showgrid=False)
    fig.update_yaxes(title_text="매출 대비 R&D 투자 비중 (%)", range=[0, top])
    return style(fig, height, c)


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
    fig.update_xaxes(title_text="매출액 (억 원)", range=[0, d["revenue_eok"].max() * 1.12], tickformat=",")
    fig.update_yaxes(title_text="매출 대비 R&D 투자 비중 (%)", range=[0, d["rnd_ratio"].max() * 1.18])
    return style(fig, 520, c)


def show(fig):
    st.plotly_chart(fig, config={"displayModeBar": False}, **PLOT_KW)


# ---------- 화면 ----------
def main():
    st.set_page_config(page_title="반도체 소부장 R&D 트래커", layout="wide")
    c = COLORS[theme_name()]

    if not os.path.exists(MASTER_PATH):
        st.error("data/processed/master.csv 파일이 없습니다. 터미널에서 `python metrics.py`를 먼저 실행하세요.")
        st.stop()
    df = load_data(os.path.getmtime(MASTER_PATH))

    st.title("반도체 소부장 R&D 투자 비중 트래커")
    st.caption("국내 반도체 소부장 12개사의 매출 대비 R&D 투자 비중 · 출처: 금융감독원 Open DART 사업보고서")

    # --- 필터 (모든 차트에 같은 조건으로 적용) ---
    with st.sidebar:
        st.header("필터")
        processes = [p for p in PROCESS_ORDER if p in set(df["process"])]
        sel = st.multiselect("공정", processes, default=processes)
        years = sorted(int(y) for y in df["year"].unique())
        year = st.selectbox("연도", years, index=len(years) - 1)
        st.caption("연도는 '기업 비교'와 '규모 vs 투자 강도'에 적용됩니다. 추이 차트는 전체 기간을 보여 줍니다.")

    if not sel:
        st.warning("공정을 하나 이상 선택해 주세요.")
        st.stop()

    view = df[df["process"].isin(sel)]
    cur = view[view["year"] == year]

    # --- 핵심 지표 ---
    prev = view[view["year"] == year - 1]
    avg = cur["rnd_ratio"].mean()
    delta = None if prev.empty else f"{avg - prev['rnd_ratio'].mean():+.1f}%p (전년 대비)"
    top = cur.loc[cur["rnd_ratio"].idxmax()]
    low = cur.loc[cur["rnd_ratio"].idxmin()]
    k1, k2, k3 = st.columns(3)
    k1.metric(f"{year}년 평균 R&D 비중", f"{avg:.1f}%", delta, delta_color="off")
    k2.metric("가장 높은 기업", f"{top['rnd_ratio']:.1f}%", top["name"], delta_color="off", delta_arrow="off")
    k3.metric("가장 낮은 기업", f"{low['rnd_ratio']:.1f}%", low["name"], delta_color="off", delta_arrow="off")

    flagged = sorted(df.loc[df["basis_flag"], "name"].unique())
    note = "\\* 표시: 보고서의 R&D와 매출 기준이 달라 보고서에 적힌 공시 비율을 사용한 기업"

    tab1, tab2, tab3, tab4 = st.tabs(["기업 비교", "연도별 추이", "규모 vs 투자 강도", "데이터·한계"])

    with tab1:
        st.subheader(f"{year}년 매출 대비 R&D 투자 비중")
        show(bar_chart(cur, c))
        if flagged:
            st.caption(note)
        st.markdown("**공정별 평균 R&D 비중 (%)**")
        tbl = (view.pivot_table(index="process", columns="year", values="rnd_ratio", aggfunc="mean")
               .reindex([p for p in PROCESS_ORDER if p in sel]).round(1))
        tbl.columns = [f"{y}년" for y in tbl.columns]
        tbl.index.name = "공정"  # 맨 왼쪽 머리글이 영어(process)로 나오지 않게 한다
        st.dataframe(tbl.style.format("{:.1f}", na_rep="-"), **DF_KW)
        st.caption("공정별 기업 수가 1~3개라 평균은 참고용입니다 (단순 평균).")

    with tab2:
        st.subheader("연도별 R&D 투자 비중 추이")
        show(trend_chart(view, c))
        st.caption("색은 구분(장비·부품/후공정·소재), 마커 모양과 오른쪽 이름은 기업을 나타냅니다.")
        if flagged:
            st.caption(note)

    with tab3:
        st.subheader(f"{year}년 매출 규모와 R&D 투자 강도")
        show(scatter_chart(cur, c))
        st.caption("매출이 커도 R&D 비중은 낮을 수 있고, 매출이 작아도 높을 수 있습니다.")
        if flagged:
            st.caption(note)

    with tab4:
        st.subheader("데이터 표")
        table = view[["name", "process", "year", "revenue_eok", "rnd_eok", "rnd_ratio",
                       "ratio_source", "revenue_yoy", "rnd_yoy", "vs_process"]].copy()
        table["revenue_eok"] = table["revenue_eok"].round(0)
        table["rnd_eok"] = table["rnd_eok"].round(1)
        table["rnd_ratio"] = table["rnd_ratio"].round(1)
        table["year"] = table["year"].astype(str)
        table["ratio_source"] = table["ratio_source"].map(
            {"계산(R&D/매출)": "계산", "공시비율(기준상이)": "공시비율"})
        # 첫 해(2023)는 전년 값이 없어 비어 있다 -> 'None' 대신 '-'로 표시
        for col in ["revenue_yoy", "rnd_yoy", "vs_process"]:
            table[col] = table[col].map(lambda v: "-" if pd.isna(v) else f"{v:+.1f}")
        table = table.rename(columns={
            "name": "기업", "process": "공정", "year": "연도",
            "revenue_eok": "매출(억원)", "rnd_eok": "R&D(억원)", "rnd_ratio": "R&D 비중(%)",
            "ratio_source": "산출", "revenue_yoy": "매출 YoY(%)",
            "rnd_yoy": "R&D YoY(%)", "vs_process": "중앙값 대비(%p)",
        }).sort_values(["기업", "연도"])
        cc = st.column_config
        widths = {
            "기업": cc.Column(width=125), "공정": cc.Column(width=85), "연도": cc.Column(width=55),
            "매출(억원)": cc.NumberColumn(width=85, format="localized"),  # 10,097처럼 천 단위 쉼표
            "R&D(억원)": cc.NumberColumn(width=85, format="%.1f"),
            "R&D 비중(%)": cc.NumberColumn(width=85, format="%.1f"),
            "산출": cc.Column(width=60), "매출 YoY(%)": cc.Column(width=85),
            "R&D YoY(%)": cc.Column(width=85),
            "중앙값 대비(%p)": cc.Column(),  # 너비를 정하지 않아 남는 칸을 채운다(오른쪽 끝이 잘리지 않게)
        }
        # 표 높이를 행 수에 맞춰 키워서 행이 중간에 잘리거나 안쪽 스크롤이 생기지 않게 한다 (행 35px + 머리글)
        tbl_height = (len(table) + 1) * 35 + 3
        try:
            st.dataframe(table, hide_index=True, column_config=widths, height=tbl_height, **DF_KW)
        except Exception:
            st.dataframe(table, hide_index=True, height=tbl_height, **DF_KW)
        st.download_button("CSV 다운로드", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name="semi_rnd_tracker.csv", mime="text/csv")

        st.subheader("데이터 출처와 한계")
        st.markdown(
            "- **출처**: 금융감독원 Open DART. 매출액은 재무제표 API, R&D는 사업보고서 '연구개발활동' 표에서 확인했습니다.\n"
            "- **R&D 비중** = 연구개발비용 ÷ 매출액 × 100. 직접 계산한 값이 보고서 공시 비율과 0.5%p 넘게 다르면 공시 비율을 사용했습니다. "
            "(ISC처럼 보고서가 정수(%)로만 적은 경우는 더 정밀한 계산값을 사용했습니다.)\n"
            f"- **기준이 다른 기업(*)**: {', '.join(flagged) if flagged else '없음'}. "
            "R&D는 지배회사 단독(별도) 기준인데 매출은 연결 기준인 경우 등이 있어 직접 계산한 값과 차이가 납니다.\n"
            "- **티에스이 2025년 R&D 증가율(+85.9%)**: 지배회사 단독 R&D가 2,031 → 3,776백만원으로 실제로 늘어난 값입니다. "
            "금액 자체가 작아(약 38억원) 조금만 늘어도 증가율이 크게 보이며, 자회사 R&D는 포함되지 않았습니다.\n"
            "- **표본 한계**: 12개사만 분석했고 평균은 단순 평균이라 업계 전체를 대표하지 않습니다.\n"
            "- R&D 비중이 높다고 반드시 좋은 것은 아닙니다. 매출이 급증하면 R&D 금액이 늘어도 비중은 낮아질 수 있습니다."
        )


main()
