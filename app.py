"""
Hidden Signals — Northfield & Co.
An interactive Streamlit dashboard on daily e-commerce revenue, marketing spend,
and the hidden calendar factors (day-of-week, BFCM, a coordinated media-budget
rhythm) that confound the raw correlation between spend and revenue.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy: push this folder to a GitHub repo and connect it at share.streamlit.io
(Streamlit Community Cloud) — set the main file to app.py. No secrets needed;
the workbook ships in data/Northfield_Co_Case_Study.xlsx and is read with a
path relative to this file, so it works unchanged on any host.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

# ============================================================================
# CONFIG & CONSTANTS
# ============================================================================
st.set_page_config(
    page_title="Hidden Signals — Northfield & Co.",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "Northfield_Co_Case_Study.xlsx"

# Validated categorical palette (fixed hue order) + chart chrome
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
CAT_HEX = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK_SOFT, INK_MUTE, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
ACCENT = "#b8721f"

SPEND_COLS = [
    "influencer_spend", "Awin_spend", "microsoft_spend", "criteo_spend", "cost_outbrain",
    "spend_Google_Branded", "spend_Google_Non_Branded", "spend_Google_PMAX",
    "spend_Google_Demand_Gen", "spend_Google_Others",
    "spend_Meta_ASC", "spend_Meta_Retargeting", "spend_Meta_Prospecting",
]
FLAG_COLS = ["UWG_Mailing", "BFCM_Promo_Effect", "holiday_list", "Offline_Promo", "Promotion_Discount"]
KPI_COLS = ["Total_Revenue", "Revenue_New_Customer"]
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

CHANNEL_LABEL = {c: c.replace("spend_", "").replace("_", " ") for c in SPEND_COLS}
FLAG_LABEL = {c: c.replace("_", " ") for c in FLAG_COLS}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_SOFT, size=13),
    plot_bgcolor=SURFACE,
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
    hoverlabel=dict(bgcolor=INK, font_color=SURFACE, font_size=12),
)


def style(fig, height=380, showlegend=True):
    fig.update_layout(**PLOTLY_LAYOUT, height=height, showlegend=showlegend)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# ============================================================================
# DATA LOADING & CACHED COMPUTATION
# ============================================================================
@st.cache_data(show_spinner="Loading Northfield workbook…")
def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name="Master data")
    glossary = pd.read_excel(DATA_PATH, sheet_name="Glossary").dropna(how="all")
    df = df.sort_values("Date").reset_index(drop=True)
    df["dow"] = df["Date"].dt.day_name()
    df["dow_n"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["t"] = np.arange(len(df))
    return df, glossary


@st.cache_data(show_spinner=False)
def compute_stl(df):
    ts = df.set_index("Date")["Total_Revenue"]
    stl = STL(ts, period=7, robust=True).fit()
    return stl.trend.values, stl.seasonal.values, stl.resid.values


@st.cache_data(show_spinner=False)
def compute_confounding(df):
    """Raw vs. partial (calendar-adjusted) correlation of each spend channel with revenue."""
    dow_dummies = pd.get_dummies(df["dow_n"], prefix="dow", drop_first=True).astype(float)
    controls = sm.add_constant(pd.concat([dow_dummies, df[FLAG_COLS], df[["t"]]], axis=1).astype(float))

    def residualize(y):
        return sm.OLS(y.astype(float), controls).fit().resid

    resid_rev = residualize(df["Total_Revenue"])
    rows = []
    for c in SPEND_COLS:
        raw = df[c].corr(df["Total_Revenue"])
        partial = np.corrcoef(resid_rev, residualize(df[c]))[0, 1]
        rows.append({"channel": c, "raw_corr": raw, "partial_corr": partial, "gap": raw - partial})
    out = pd.DataFrame(rows).sort_values("raw_corr", ascending=False).reset_index(drop=True)
    out["pct_explained_by_calendar"] = (out["gap"] / out["raw_corr"]).clip(-2, 2)
    return out


@st.cache_data(show_spinner=False)
def compute_pca(df):
    X = StandardScaler().fit_transform(df[SPEND_COLS])
    pca = PCA(n_components=5).fit(X)
    scores = pca.transform(X)[:, 0]
    loadings = pd.DataFrame(
        {"channel": SPEND_COLS, "loading": pca.components_[0]}
    ).sort_values("loading", ascending=False).reset_index(drop=True)
    corrs = {
        "Total_Revenue": np.corrcoef(scores, df["Total_Revenue"])[0, 1],
        "BFCM_Promo_Effect": np.corrcoef(scores, df["BFCM_Promo_Effect"])[0, 1],
        "Promotion_Discount": np.corrcoef(scores, df["Promotion_Discount"])[0, 1],
        "UWG_Mailing": np.corrcoef(scores, df["UWG_Mailing"])[0, 1],
    }
    return loadings, scores, pca.explained_variance_ratio_[0], corrs


@st.cache_data(show_spinner=False)
def compute_event_study(df):
    rows = []
    for flag in FLAG_COLS:
        g1, g0 = df.loc[df[flag] == 1, "Total_Revenue"], df.loc[df[flag] == 0, "Total_Revenue"]
        t, p = stats.ttest_ind(g1, g0, equal_var=False)
        rows.append({
            "flag": flag, "n_on": len(g1), "mean_off": g0.mean(), "mean_on": g1.mean(),
            "uplift_pct": (g1.mean() / g0.mean() - 1) * 100, "t_stat": t, "p_value": p,
        })
    return pd.DataFrame(rows).sort_values("uplift_pct", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_lag(df, channels, max_lag=7):
    out = {}
    for ch in channels:
        out[ch] = [df[ch].shift(lag).corr(df["Total_Revenue"]) for lag in range(max_lag + 1)]
    return out


@st.cache_data(show_spinner=False)
def compute_rolling(df, channels, window):
    roll_df = df.set_index("Date")
    out = {}
    for ch in channels:
        out[ch] = roll_df["Total_Revenue"].rolling(window).corr(roll_df[ch])
    return out


# ============================================================================
# LOAD
# ============================================================================
df, glossary = load_data()
trend, seasonal, resid = compute_stl(df)
df["stl_trend"] = trend
df["stl_seasonal"] = seasonal

DATE_MIN, DATE_MAX = df["Date"].min().date(), df["Date"].max().date()

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
with st.sidebar:
    st.markdown("### 📊 Hidden Signals")
    st.caption("Northfield & Co. — daily e-commerce, 1 Apr 2024 – 31 Jul 2026")

    st.markdown("---")
    st.markdown("**Date range** _(affects Overview only — the hidden-factor analysis below always uses the full 852-day history for statistical reliability)_")
    date_range = st.slider(
        "Date range", min_value=DATE_MIN, max_value=DATE_MAX, value=(DATE_MIN, DATE_MAX),
        format="DD MMM YYYY", label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Deep-dive channels**")
    default_channels = ["Awin_spend", "spend_Google_PMAX", "spend_Meta_ASC", "spend_Meta_Prospecting"]
    channels_sel = st.multiselect(
        "Channels for lag / rolling-correlation charts",
        options=SPEND_COLS, default=default_channels,
        format_func=lambda c: CHANNEL_LABEL[c],
    )
    scatter_channel = st.selectbox(
        "Channel for the Simpson's-paradox scatter", options=SPEND_COLS,
        index=SPEND_COLS.index("spend_Google_PMAX"), format_func=lambda c: CHANNEL_LABEL[c],
    )
    roll_window = st.slider("Rolling-correlation window (days)", 30, 180, 90, step=10)
    sig_level = st.slider("Significance threshold (p <)", 0.01, 0.10, 0.05, step=0.01)

    st.markdown("---")
    st.caption(
        "All relationships shown are **observational associations**, not proven causal effects. "
        "Correlations are Pearson; partial correlation residualises each series on day-of-week "
        "dummies, the 5 promo/calendar flags and a linear trend before correlating residuals."
    )

mask = (df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])
dfv = df.loc[mask].copy()
if len(dfv) < 14:
    st.warning("Selected range is very short — widen it for meaningful weekly patterns.")

if not channels_sel:
    channels_sel = default_channels

# ============================================================================
# HERO
# ============================================================================
st.title("Every Sunday, revenue jumps 67%. It isn't in any spreadsheet column.")
st.markdown(
    "Northfield & Co.'s marketing numbers show paid media correlating with revenue — some channels "
    "above 0.4. This dashboard goes looking for **why**, and finds a set of **hidden calendar factors** "
    "— day-of-week, the BFCM window, a coordinated media-budget rhythm — quietly moving spend and "
    "revenue together, inflating the story the raw numbers tell."
)

sunday_mean = df.loc[df["dow"] == "Sunday", "Total_Revenue"].mean()
weekday_mean = df.loc[df["dow"] != "Sunday", "Total_Revenue"].mean()
media_share = df[SPEND_COLS].sum(axis=1).sum() / df["Total_Revenue"].sum() * 100
bfcm_uplift = (
    df.loc[df["BFCM_Promo_Effect"] == 1, "Total_Revenue"].mean()
    / df.loc[df["BFCM_Promo_Effect"] == 0, "Total_Revenue"].mean() - 1
) * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg daily revenue (selected range)", f"${dfv['Total_Revenue'].mean():,.0f}")
k2.metric("Paid media as share of revenue", f"{media_share:.0f}%")
k3.metric("Sunday vs. weekday revenue", f"+{sunday_mean/weekday_mean-1:.0%}")
k4.metric("BFCM window uplift", f"+{bfcm_uplift:.0f}%")

st.markdown("")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① Overview — The Pattern", "② The Suspects", "③ The Reveal — Hidden Factors",
    "④ The Verdict", "🗂 Data Explorer",
])

# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================
with tab1:
    st.subheader("Revenue has a heartbeat the flag columns don't capture")
    st.caption(
        "Daily revenue with a 7-day STL trend overlaid, plus BFCM markers. Zoom, pan or hover for exact values."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfv["Date"], y=dfv["Total_Revenue"], name="Daily revenue",
        line=dict(color=INK_MUTE, width=1), opacity=0.75,
        hovertemplate="%{x|%d %b %Y}<br>Revenue: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dfv["Date"], y=dfv["stl_trend"], name="7-day trend (STL)",
        line=dict(color=BLUE, width=2.4),
        hovertemplate="%{x|%d %b %Y}<br>Trend: $%{y:,.0f}<extra></extra>",
    ))
    bfcm_pts = dfv[dfv["BFCM_Promo_Effect"] == 1]
    fig.add_trace(go.Scatter(
        x=bfcm_pts["Date"], y=bfcm_pts["Total_Revenue"], name="BFCM window", mode="markers",
        marker=dict(color=RED, size=7, symbol="circle"),
        hovertemplate="%{x|%d %b %Y}<br>BFCM day: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_yaxes(title="USD / day", tickprefix="$", tickformat=",.0f")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(style(fig, height=440), width='stretch')

    st.markdown(
        "Pull the weekly wave apart by day of week and one day stands alone: **Sunday averages "
        f"\\${sunday_mean:,.0f}** against a Mon–Sat average of **\\${weekday_mean:,.0f}** "
        f"(**+{sunday_mean/weekday_mean-1:.0%}**). Day-of-week isn't a labelled control anywhere "
        "in this workbook — it's derived purely from the calendar date, which is exactly why it's "
        "easy to miss and easy to confuse for something else."
    )

    c1, c2 = st.columns(2)
    with c1:
        dow_rev = dfv.groupby("dow")["Total_Revenue"].mean().reindex(DOW_ORDER)
        colors = [RED if d == "Sunday" else BLUE for d in DOW_ORDER]
        fig = go.Figure(go.Bar(
            x=DOW_ORDER, y=dow_rev.values, marker_color=colors,
            hovertemplate="%{x}<br>Avg revenue: $%{y:,.0f}<extra></extra>",
        ))
        fig.update_yaxes(title="Mean Total_Revenue", tickprefix="$", tickformat=",.0f")
        st.markdown("**Average revenue by day of week**")
        st.plotly_chart(style(fig, height=340, showlegend=False), width='stretch')
    with c2:
        mail_dow = (
            dfv.loc[dfv["UWG_Mailing"] == 1, "dow"].value_counts().reindex(DOW_ORDER).fillna(0).astype(int)
        )
        colors = [ACCENT if d in ("Sunday", "Thursday") else INK_MUTE for d in DOW_ORDER]
        fig = go.Figure(go.Bar(
            x=DOW_ORDER, y=mail_dow.values, marker_color=colors,
            hovertemplate="%{x}<br>Mailings: %{y}<extra></extra>",
        ))
        fig.update_yaxes(title="# of mailing days")
        st.markdown("**When do UWG mailings actually go out?**")
        st.plotly_chart(style(fig, height=340, showlegend=False), width='stretch')

    total_mail = mail_dow.sum()
    if total_mail > 0:
        pct = (mail_dow.get("Sunday", 0) + mail_dow.get("Thursday", 0)) / total_mail
        st.info(
            f"**Read together:** {pct:.0%} of mailings in the selected range land on a Sunday or "
            "Thursday — the loyalty-file mailing programme is not spread evenly across the week. "
            "Any uplift attributed to \"the mailing\" is, in part, really just Sunday."
        )

# ============================================================================
# TAB 2 — RAW CORRELATION ("THE SUSPECTS")
# ============================================================================
with tab2:
    st.subheader("On paper, several channels look like they're driving revenue")
    st.caption(
        "Rank every spend channel and calendar flag by raw Pearson correlation with Total_Revenue "
        "(always computed on the full 852-day history)."
    )

    corr_cols = KPI_COLS + FLAG_COLS + SPEND_COLS
    raw_corr = df[corr_cols].corr()["Total_Revenue"].drop("Total_Revenue").sort_values(ascending=False)

    def kind(c):
        if c in FLAG_COLS:
            return "Promo / calendar flag"
        if c == "Revenue_New_Customer":
            return "Revenue component (not a driver)"
        return "Paid media channel"

    kind_color = {"Promo / calendar flag": VIOLET, "Paid media channel": BLUE, "Revenue component (not a driver)": INK_MUTE}
    labels = [FLAG_LABEL.get(c, CHANNEL_LABEL.get(c, c)) for c in raw_corr.index]
    colors = [kind_color[kind(c)] for c in raw_corr.index]

    fig = go.Figure(go.Bar(
        x=raw_corr.values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.2f}" for v in raw_corr.values], textposition="outside",
        hovertemplate="%{y}<br>r = %{x:.2f}<extra></extra>", showlegend=False,
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Pearson correlation with Total_Revenue", range=[min(raw_corr.min() - 0.08, -0.05), raw_corr.max() + 0.12])
    for lbl, c in kind_color.items():
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=lbl))
    st.plotly_chart(style(fig, height=560), width='stretch')

    st.warning(
        "**Awin_spend (affiliate) tops every media channel at r = 0.76** — but Awin is paid on "
        "**commission**. Some of that correlation runs backwards: more sales this week can itself "
        "generate more affiliate commission, not the other way round. A high raw number here is a "
        "flag to investigate, not a result to act on — see the next tab."
    )

# ============================================================================
# TAB 3 — HIDDEN FACTORS ("THE REVEAL")
# ============================================================================
with tab3:
    st.subheader("Four checks, and the same hidden hand shows up in all of them")
    st.markdown(
        "If a spend channel really drives revenue, its relationship with revenue should **survive "
        "controlling for the calendar**, show up as **its own distinct pattern** (not a shared one), "
        "show a **stable slope** regardless of what else is happening that day, and **build up over "
        "the following days** rather than only landing same-day."
    )

    st.markdown("#### 1 — Strip out the calendar, and most of the \"media effect\" shrinks")
    st.caption(
        "Each channel is regressed on day-of-week, all 5 promo/calendar flags and a linear growth "
        "trend; the residual is correlated with revenue's own residual — the relationship "
        "*after* removing everything the calendar already explains."
    )
    conf = compute_confounding(df)
    fig = go.Figure()
    labels = [CHANNEL_LABEL[c] for c in conf["channel"]]
    fig.add_trace(go.Bar(x=conf["raw_corr"], y=labels, orientation="h", name="Raw correlation",
                          marker_color=BLUE, hovertemplate="%{y}<br>Raw r = %{x:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=conf["partial_corr"], y=labels, orientation="h", name="Partial (calendar removed)",
                          marker_color=ACCENT, hovertemplate="%{y}<br>Partial r = %{x:.2f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Correlation with Total_Revenue")
    st.plotly_chart(style(fig, height=520), width='stretch')

    biggest = conf.reindex(conf["gap"].abs().sort_values(ascending=False).index).iloc[0]
    st.warning(
        f"**{CHANNEL_LABEL[biggest['channel']]} drops the hardest** — {biggest['raw_corr']:.2f} → "
        f"{biggest['partial_corr']:.2f}, a {abs(biggest['pct_explained_by_calendar']):.0%} relative fall "
        "— meaning most of its raw correlation was calendar timing, not a direct link. **Awin_spend stays "
        "highest even adjusted**, consistent with its commission structure rather than a pure "
        "spend→revenue story."
    )

    st.markdown("#### 2 — The channels don't move independently — one hidden factor pushes most of them at once")
    st.caption(
        "First principal component (PCA) across all 13 standardised spend channels — a single latent "
        "\"push the budget today\" behaviour."
    )
    loadings, pc1_scores, explvar, pc1_corrs = compute_pca(df)
    df["_pc1"] = pc1_scores
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = go.Figure(go.Bar(
            x=loadings["loading"], y=[CHANNEL_LABEL[c] for c in loadings["channel"]], orientation="h",
            marker_color=BLUE, hovertemplate="%{y}<br>Loading: %{x:.2f}<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title=f"PC1 loading ({explvar:.1%} of variance)")
        st.plotly_chart(style(fig, height=420, showlegend=False), width='stretch')
    with c2:
        m1, m2 = st.columns(2)
        m1.metric("corr(PC1, Total_Revenue)", f"{pc1_corrs['Total_Revenue']:.2f}")
        m2.metric("corr(PC1, BFCM window)", f"{pc1_corrs['BFCM_Promo_Effect']:.2f}")
        m3, m4 = st.columns(2)
        m3.metric("corr(PC1, Promotion_Discount flag)", f"{pc1_corrs['Promotion_Discount']:.2f}")
        m4.metric("corr(PC1, UWG_Mailing flag)", f"{pc1_corrs['UWG_Mailing']:.2f}")
        pc1_dow = df.groupby("dow")["_pc1"].mean().reindex(DOW_ORDER)
        colors = [RED if d == "Sunday" else AQUA for d in DOW_ORDER]
        fig = go.Figure(go.Bar(x=DOW_ORDER, y=pc1_dow.values, marker_color=colors,
                                hovertemplate="%{x}<br>PC1: %{y:.2f}<extra></extra>"))
        fig.add_hline(y=0, line_color=INK_MUTE, line_width=1)
        st.plotly_chart(style(fig, height=260, showlegend=False), width='stretch')
        st.caption("Mean media-push factor score by weekday — peaks Sunday, same as revenue itself.")

    st.info(
        "The labelled **Promotion_Discount flag barely correlates with the push factor** — the binary "
        "column doesn't capture it. What the hidden factor actually tracks is **Sunday and the "
        "November/BFCM window**: the same calendar rhythm that independently drives revenue is also "
        "when the team happens to turn up several media channels together."
    )

    st.markdown("#### 3 — Promo days sit on a different shelf entirely")
    st.caption(f"{CHANNEL_LABEL[scatter_channel]} spend vs. revenue, coloured by promo status (pick a different channel in the sidebar).")
    fig = go.Figure()
    for flag_val, color, label in [(0, BLUE, "Full-price day"), (1, ACCENT, "Promo day")]:
        sub = df[df["Promotion_Discount"] == flag_val]
        fig.add_trace(go.Scatter(
            x=sub[scatter_channel], y=sub["Total_Revenue"], mode="markers", name=label,
            marker=dict(color=color, size=6, opacity=0.55),
            hovertemplate="Spend: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<extra></extra>",
        ))
    fig.update_xaxes(title=f"{CHANNEL_LABEL[scatter_channel]} (USD)")
    fig.update_yaxes(title="Total_Revenue (USD)")
    st.plotly_chart(style(fig, height=440), width='stretch')
    st.caption(
        "Promo-day points sit as a systematically higher band across the full range of spend — the "
        "flag shifts the whole cluster up rather than changing the slope. That's the confounding "
        "signature a same-day, unlagged correlation can't tell apart from a genuine media lift."
    )

    st.markdown("#### 4 — No channel shows a delayed payoff")
    st.caption("Correlation with revenue by lag (days spend leads revenue) — for the channels selected in the sidebar.")
    lag_res = compute_lag(df, channels_sel)
    fig = go.Figure()
    for i, (ch, vals) in enumerate(lag_res.items()):
        fig.add_trace(go.Scatter(
            x=list(range(len(vals))), y=vals, name=CHANNEL_LABEL[ch], mode="lines+markers",
            line=dict(color=CAT_HEX[i % len(CAT_HEX)], width=2.2), marker=dict(size=6),
            hovertemplate="Lag %{x}<br>r = %{y:.2f}<extra>" + CHANNEL_LABEL[ch] + "</extra>",
        ))
    fig.update_xaxes(title="Lag (days spend leads revenue)", dtick=1)
    fig.update_yaxes(title="Correlation with Total_Revenue")
    st.plotly_chart(style(fig, height=420), width='stretch')
    peak_lags = {ch: int(np.argmax(vals)) for ch, vals in lag_res.items()}
    st.caption("Peak lag by channel: " + " · ".join(f"{CHANNEL_LABEL[c]} = {l}" for c, l in peak_lags.items()) + " (0 = same day)")

    st.markdown(f"#### Even the confounded, same-day relationship isn't stable ({roll_window}-day rolling correlation)")
    roll_res = compute_rolling(df, channels_sel, roll_window)
    fig = go.Figure()
    for i, (ch, series) in enumerate(roll_res.items()):
        fig.add_trace(go.Scatter(
            x=df["Date"], y=series.values, name=CHANNEL_LABEL[ch], mode="lines",
            line=dict(color=CAT_HEX[i % len(CAT_HEX)], width=1.6),
            hovertemplate="%{x|%d %b %Y}<br>r = %{y:.2f}<extra>" + CHANNEL_LABEL[ch] + "</extra>",
        ))
    fig.add_hline(y=0, line_color=INK_MUTE, line_width=1)
    fig.update_yaxes(title=f"{roll_window}-day rolling correlation")
    st.plotly_chart(style(fig, height=380), width='stretch')
    st.caption(
        "Rolling correlations swing substantially through the series — including brief negative "
        "spells — rather than sitting flat. The full-sample correlation is an average over shifting "
        "promotional regimes, not a stable structural relationship."
    )

# ============================================================================
# TAB 4 — VERDICT
# ============================================================================
with tab4:
    st.subheader("What actually moves revenue, ranked and quantified")
    st.caption("Mean revenue on \"on\" days vs. \"off\" days for each flag (Welch's t-test), full 852-day history.")

    ev = compute_event_study(df)
    colors = [BLUE if p < sig_level else INK_MUTE for p in ev["p_value"]]
    fig = go.Figure(go.Bar(
        x=ev["uplift_pct"], y=[FLAG_LABEL[f] for f in ev["flag"]], orientation="h", marker_color=colors,
        text=[f"+{v:.0f}%" for v in ev["uplift_pct"]], textposition="outside",
        hovertemplate="%{y}<br>Uplift: %{x:.1f}%<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Revenue uplift vs. \"off\" days (%)")
    st.plotly_chart(style(fig, height=320, showlegend=False), width='stretch')
    st.caption(f"Grey bars are not significant at p < {sig_level:.2f}.")

    show = ev.copy()
    show["flag"] = show["flag"].map(FLAG_LABEL)
    show["mean_off"] = show["mean_off"].map(lambda v: f"${v:,.0f}")
    show["mean_on"] = show["mean_on"].map(lambda v: f"${v:,.0f}")
    show["uplift_pct"] = show["uplift_pct"].map(lambda v: f"+{v:.1f}%")
    show["significant"] = ev["p_value"].apply(lambda p: f"Yes (p={p:.1e})" if p < sig_level else f"No (p={p:.2f})")
    st.dataframe(
        show[["flag", "n_on", "mean_off", "mean_on", "uplift_pct", "significant"]].rename(columns={
            "flag": "Flag", "n_on": "Days on", "mean_off": "Mean off", "mean_on": "Mean on",
            "uplift_pct": "Uplift", "significant": "Significant?",
        }),
        hide_index=True, width='stretch',
    )

    sunday_promo = df[(df["dow"] == "Sunday") & (df["Promotion_Discount"] == 1)]
    g1 = sunday_promo.loc[sunday_promo["UWG_Mailing"] == 1, "Total_Revenue"]
    g0 = sunday_promo.loc[sunday_promo["UWG_Mailing"] == 0, "Total_Revenue"]
    if len(g1) >= 3 and len(g0) >= 3:
        _, p_lfl = stats.ttest_ind(g1, g0, equal_var=False)
        st.info(
            f"**Like-for-like check:** within Sunday + promo-discount days only, mailing mean "
            f"= \\${g1.mean():,.0f} (n={len(g1)}) vs. no-mailing mean = \\${g0.mean():,.0f} (n={len(g0)}) → "
            f"uplift = **{g1.mean()/g0.mean()-1:.1%}** (p={p_lfl:.3f}) — real, but smaller than the naive "
            f"+{ev.loc[ev['flag']=='UWG_Mailing','uplift_pct'].values[0]:.0f}% once the Sunday/promo "
            "overlap is stripped out."
        )

    st.markdown("### The hidden factors, mapped")
    factors = [
        ("01 — CALENDAR", "Day of week (Sunday)",
         "Not a column anywhere in the raw data. Drives revenue directly (+67%), drives the hidden "
         "media-push factor, and is where ~40% of mailings land."),
        ("02 — CALENDAR", "BFCM window",
         "Just 17 days (2% of the year) but the single largest uplift (+185%) and the strongest driver "
         "of the coordinated media-push factor."),
        ("03 — PLAYBOOK", "Site-wide promo discount",
         "Live ~67% of all days. Co-occurs with 92% of UWG mailing days, diluting any \"pure\" mailing "
         "read and lifting revenue +36% on its own."),
        ("04 — BEHAVIOUR", "Coordinated media \"push\"",
         "A latent factor, not a labelled column — ~31% of all spend co-movement. Tracks Sunday and "
         "BFCM almost as tightly as it tracks revenue itself."),
        ("05 — MECHANICS", "Affiliate commission structure",
         "Awin is paid a % of attributed sales — spend can be a consequence of revenue, not only a "
         "cause. Highest raw correlation of any channel, partly for this reason."),
        ("06 — GROWTH", "Underlying growth trend",
         "The business grew across the 2+ year window, slowly inflating cumulative spend and revenue "
         "together — a small amount of \"correlation for free.\""),
    ]
    cols = st.columns(3)
    for i, (tag, title, body) in enumerate(factors):
        with cols[i % 3]:
            st.markdown(f"**{tag}**")
            st.markdown(f"**{title}**")
            st.caption(body)
            st.markdown("")

    st.markdown("### What to do next")
    recs = [
        ("Budget decisions off calendar-adjusted numbers, not raw correlation",
         "Raw correlation overstates most channels, especially Meta Prospecting. Re-run the partial-"
         "correlation view before reallocating spend."),
        ("Get a clean read on the mailing programme",
         "92% of UWG mailings already coincide with a live discount. Test mailings on non-Sunday, "
         "non-promo days — or a holdout/suppression test — to separate its own contribution."),
        ("Audit the affiliate (Awin) attribution",
         "Decompose Awin's revenue correlation into genuinely incremental affiliate sales vs. "
         "commission paid on sales that would have happened anyway."),
        ("Test for a real (lagged) media effect before trusting same-day numbers",
         "Every channel peaks at lag 0 with no build-up. Worth a proper incrementality/holdout test "
         "on 1–2 channels before writing off a lasting media effect."),
        ("Treat the holiday flag with caution",
         "+21% uplift, but only borderline significant (n=40) and overlapping other flags. Needs a "
         "cleaner definition — major vs. minor holidays — before it's treated as a lever."),
    ]
    for i, (title, body) in enumerate(recs, 1):
        st.markdown(f"**{i}. {title}**")
        st.caption(body)

# ============================================================================
# TAB 5 — DATA EXPLORER
# ============================================================================
with tab5:
    st.subheader("Raw data explorer")
    st.caption("Filtered to the sidebar date range. Use the column headers to sort.")
    show_cols = ["Date", "dow"] + FLAG_COLS + KPI_COLS + SPEND_COLS
    st.dataframe(dfv[show_cols], width='stretch', height=420)
    st.download_button(
        "Download filtered data as CSV", dfv[show_cols].to_csv(index=False).encode(),
        file_name="northfield_filtered.csv", mime="text/csv",
    )
    st.markdown("### Field glossary")
    st.dataframe(glossary, width='stretch', hide_index=True)

st.markdown("---")
st.caption(
    "Methodology: all relationships described are observational associations in 852 days of daily "
    "data (1 Apr 2024 – 31 Jul 2026), not proven causal effects. Correlations are Pearson; \"partial "
    "correlation\" residualises each series on day-of-week dummies, the five promo/calendar flags and "
    "a linear time trend (OLS) before correlating residuals. The hidden media-push factor is PC1 of a "
    "standardised PCA across the 13 spend channels. Event-study uplifts use Welch's t-test. "
    "Source: Northfield_Co_Case_Study.xlsx, Master data tab."
)
