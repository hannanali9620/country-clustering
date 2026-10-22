import inspect
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Global Development Clusters", page_icon="🌍",
                   layout="wide", initial_sidebar_state="expanded")
BASE = Path(__file__).parent


# ---------------------------------------------------------------- helpers
def _stretch(fn):
    """Streamlit ke naye aur purane versions dono me full-width chalane ke liye."""
    p = inspect.signature(fn).parameters
    if "width" in p and p["width"].default == "stretch":
        return {"width": "stretch"}
    if "use_container_width" in p:
        return {"use_container_width": True}
    return {}


PLOT_KW = _stretch(st.plotly_chart)
DF_KW = _stretch(st.dataframe)

SHORT = {"child_mort": "Child mortality", "exports": "Exports", "health": "Health spend",
         "imports": "Imports", "income": "Income", "inflation": "Inflation",
         "life_expec": "Life expectancy", "total_fer": "Fertility", "gdpp": "GDP per capita"}
HIGHER_IS_BAD = {"child_mort", "inflation", "total_fer"}

# dataset ke kuch naam Plotly ke map se match nahi hote
NAME_FIX = {
    "Congo, Dem. Rep.": "Democratic Republic of the Congo", "Congo, Rep.": "Republic of the Congo",
    "Cote d'Ivoire": "Ivory Coast", "Kyrgyz Republic": "Kyrgyzstan", "Lao": "Laos",
    "Micronesia, Fed. Sts.": "Micronesia", "Slovak Republic": "Slovakia",
    "Macedonia, FYR": "North Macedonia",
}

ANCHORS = ["#ef4444", "#f59e0b", "#22c55e", "#06b6d4"]   # red (high need) -> cyan (low need)
BRANCH_COLORS = ["#818cf8", "#34d399", "#fbbf24", "#f472b6", "#38bdf8",
                 "#fb7185", "#a3e635", "#c084fc", "#2dd4bf", "#fdba74"]


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def ramp(n):
    if n == 1:
        return [ANCHORS[0]]
    out = []
    for i in range(n):
        t = i / (n - 1) * (len(ANCHORS) - 1)
        lo = int(np.floor(t))
        hi = min(lo + 1, len(ANCHORS) - 1)
        f = t - lo
        a, b = _rgb(ANCHORS[lo]), _rgb(ANCHORS[hi])
        c = tuple(round(a[j] + (b[j] - a[j]) * f) for j in range(3))
        out.append("#%02x%02x%02x" % c)
    return out


def rgba(hex_color, alpha):
    r, g, b = _rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def style(fig, height=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e5e7eb"),
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    if height:
        fig.update_layout(height=height)
    return fig


def show(fig):
    st.plotly_chart(fig, theme=None, **PLOT_KW)


def kpi(col, label, value, sub, color):
    col.markdown(f'<div class="kpi" style="--accent:{color}"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                 unsafe_allow_html=True)


# ---------------------------------------------------------------- css
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }
.hero { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
  border-radius: 22px; padding: 30px 36px; margin-bottom: 22px;
  box-shadow: 0 12px 40px rgba(124,58,237,.30); }
.hero-title { font-size: 2.1rem; font-weight: 800; color: #fff; line-height: 1.2; }
.hero-sub { color: rgba(255,255,255,.88); font-size: 1.02rem; margin-top: 8px; max-width: 760px; }
.hero-tags { margin-top: 16px; }
.hero-tags span { display: inline-block; background: rgba(255,255,255,.18); color: #fff;
  padding: 4px 14px; border-radius: 999px; font-size: .8rem; font-weight: 600; margin-right: 8px; }
.kpi { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
  border-top: 4px solid var(--accent); border-radius: 16px; padding: 16px 20px;
  transition: transform .2s, box-shadow .2s; }
.kpi:hover { transform: translateY(-3px); box-shadow: 0 10px 28px rgba(0,0,0,.35); }
.kpi-label { color: #94a3b8; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
.kpi-value { color: #fff; font-size: 2rem; font-weight: 800; margin: 2px 0; }
.kpi-sub { color: #94a3b8; font-size: .8rem; }
.pills { margin: 6px 0 12px 0; }
.pill { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px;
  background: rgba(255,255,255,.05); border: 1px solid var(--c); margin: 0 8px 8px 0; font-size: .85rem; }
.pill i { width: 10px; height: 10px; border-radius: 50%; background: var(--c); display: inline-block; }
.gcard { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
  border-left: 5px solid var(--accent); border-radius: 14px; padding: 14px 18px; margin-bottom: 14px; }
.gname { font-weight: 700; font-size: 1rem; color: #fff; }
.gcount { color: var(--accent); font-weight: 700; font-size: .85rem; margin-bottom: 6px; }
.gstat { color: #94a3b8; font-size: .82rem; }
.gstat b { color: #e5e7eb; }
.country-head { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px; padding: 18px 24px; margin-bottom: 14px; }
.country-name { font-size: 1.7rem; font-weight: 800; color: #fff; }
.badge { display: inline-block; padding: 4px 14px; border-radius: 999px; font-size: .8rem;
  font-weight: 700; color: #0b1020; margin-top: 6px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,.04); border-radius: 12px; padding: 8px 18px; height: auto; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #4f46e5, #7c3aed); color: #fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    d = pd.read_csv(BASE / "countries_clustered.csv")
    m = json.loads((BASE / "models" / "metrics.json").read_text())
    return d, m


@st.cache_resource
def load_art():
    return joblib.load(BASE / "models" / "artifacts.joblib")


df, metrics = load()
art = load_art()
Z, Xs, FEATURES = art["linkage"], art["X_scaled"], art["features"]

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🎛️ Controls")
k = st.sidebar.slider("Number of clusters (k)", 2, 10, int(metrics["k"]))
st.sidebar.caption("Dendrogram ko k jagah se kaat kar groups banate hain. "
                   f"Training me k = {metrics['k']} chuna gaya tha.")

df = df.copy()
df["raw"] = fcluster(Z, k, "maxclust")
order = df.groupby("raw")["need_score"].mean().sort_values(ascending=False).index.tolist()
names = {}
for i, c in enumerate(order):
    label = f"Group {i + 1}"
    if i == 0:
        label += " (highest need)"
    if i == k - 1:
        label += " (lowest need)"
    names[c] = label
df["group"] = df["raw"].map(names)
group_order = [names[c] for c in order]
cmap = dict(zip(group_order, ramp(k)))
df["map_name"] = df["country"].replace(NAME_FIX)
mn, mx = df["need_score"].min(), df["need_score"].max()
df["need_idx"] = (df["need_score"] - mn) / (mx - mn) * 100

shown = st.sidebar.multiselect("Groups to display", group_order, default=group_order)
if not shown:
    st.warning("Sidebar se kam az kam ek group chuno.")
    st.stop()
view = df[df["group"].isin(shown)]

with st.sidebar.expander("ℹ️ How to read this"):
    st.write("**Group 1** = sab se zyada need (bachon ki maut zyada, income aur life expectancy kam). "
             "Last group = sab se kam need. Need score 6 indicators ka simple average hai.")

sil = float(silhouette_score(Xs[FEATURES].values, df["raw"]))
top_group = group_order[0]
top_df = df[df["group"] == top_group]

# ---------------------------------------------------------------- hero + KPIs
st.markdown(
    '<div class="hero"><div class="hero-title">🌍 Global Development Clusters</div>'
    f'<div class="hero-sub">Hierarchical clustering of {len(df)} countries on 9 socio-economic '
    'indicators, to show where aid is needed most.</div>'
    f'<div class="hero-tags"><span>Ward linkage</span><span>k = {k}</span>'
    f'<span>Silhouette {sil:.2f}</span></div></div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Countries", len(df), "in the dataset", "#818cf8")
kpi(k2, "Clusters", k, "groups found", "#f472b6")
kpi(k3, "Highest need", len(top_df), f"avg child mortality {top_df['child_mort'].mean():.0f} / 1000", "#ef4444")
kpi(k4, "Silhouette", f"{sil:.3f}", "higher = better separated", "#22c55e")
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗺️ World map", "🌳 Dendrogram", "📊 Cluster profiles", "🔎 Country explorer", "🎯 Aid priority"])

# ================================================================ TAB 1
with tab1:
    pills = "".join(f'<span class="pill" style="--c:{cmap[g]}"><i></i>{g} · {int((df.group == g).sum())}</span>'
                    for g in group_order)
    st.markdown(f'<div class="pills">{pills}</div>', unsafe_allow_html=True)

    fig = px.choropleth(view, locations="map_name", locationmode="country names", color="group",
                        hover_name="country", color_discrete_map=cmap,
                        category_orders={"group": group_order},
                        hover_data={"child_mort": ":.1f", "income": ":,.0f", "life_expec": ":.1f",
                                    "group": False, "map_name": False})
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                    bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#1e293b",
                    showcountries=True, countrycolor="#0b1020")
    fig.update_layout(legend_title_text="")
    show(style(fig, 520))
    st.caption("Kuch chhote countries ke naam Plotly ke map se match nahi hote, wo map par nahi dikhte.")

    a, b = st.columns(2)
    with a:
        f2 = px.scatter(view, x="income", y="child_mort", color="group", hover_name="country",
                        log_x=True, color_discrete_map=cmap, category_orders={"group": group_order},
                        title="Income vs child mortality", opacity=0.85,
                        labels={"income": "Income per person ($, log)", "child_mort": "Child mortality"})
        f2.update_traces(marker=dict(size=10, line=dict(width=0.5, color="#0b1020")))
        show(style(f2, 420))
    with b:
        f3 = px.box(view, x="group", y="need_score", color="group", points="all", hover_name="country",
                    color_discrete_map=cmap, category_orders={"group": group_order},
                    title="Need score by group", labels={"need_score": "Need score", "group": ""})
        f3.update_layout(showlegend=False)
        f3.update_xaxes(showticklabels=False)
        show(style(f3, 420))

# ================================================================ TAB 2
with tab2:
    mode = st.radio("View", ["Compact (top merges)", "Full (all countries)"], horizontal=True)
    thr = (Z[-k, 2] + Z[-(k - 1), 2]) / 2
    hierarchy.set_link_color_palette(BRANCH_COLORS)
    with plt.style.context("dark_background"):
        if mode.startswith("Compact"):
            fig, ax = plt.subplots(figsize=(11, 5.5))
            dendrogram(Z, truncate_mode="lastp", p=30, show_contracted=True, color_threshold=thr,
                       above_threshold_color="#64748b", leaf_rotation=90, ax=ax)
            ax.axhline(thr, color="#f43f5e", linestyle="--", linewidth=1.2)
            ax.set_ylabel("Ward distance")
        else:
            fig, ax = plt.subplots(figsize=(9, max(10, len(df) * 0.17)))
            dendrogram(Z, labels=df["country"].values, orientation="left", color_threshold=thr,
                       above_threshold_color="#64748b", leaf_font_size=7, ax=ax)
            ax.axvline(thr, color="#f43f5e", linestyle="--", linewidth=1.2)
            ax.set_xlabel("Ward distance")
        fig.patch.set_facecolor("#0b1020")
        ax.set_facecolor("#0b1020")
        for s in ax.spines.values():
            s.set_color("#334155")
        st.pyplot(fig)
        plt.close(fig)
    hierarchy.set_link_color_palette(None)
    st.caption("Red dotted line = jahan tree kata gaya. Sidebar me k badlo to line upar neeche hogi.")

    st.markdown("#### Which linkage method is best?")
    c1, c2 = st.columns(2)
    comp = metrics["comparison"]
    with c1:
        ct = pd.DataFrame({m: {"Cophenetic": round(v["cophenetic"], 3),
                               "Best silhouette": round(max(v["silhouette_by_k"].values()), 3)}
                           for m, v in comp.items()}).T
        st.dataframe(ct, **DF_KW)
        st.caption("Average linkage ke numbers behtar hain, lekin woh bohat unbalanced clusters "
                   "banata hai (jaise 161 / 5 / 1), isliye Ward chuna gaya.")
    with c2:
        sw = pd.Series({int(kk): vv for kk, vv in comp["ward"]["silhouette_by_k"].items()}).sort_index()
        f4 = px.line(x=sw.index, y=sw.values, markers=True, title="Ward: silhouette vs k",
                     labels={"x": "k", "y": "Silhouette"})
        f4.update_traces(line_color="#818cf8", marker=dict(size=9))
        f4.add_vline(x=k, line_dash="dash", line_color="#f43f5e")
        show(style(f4, 300))

# ================================================================ TAB 3
with tab3:
    gm = df.groupby("group")[FEATURES].mean()
    cnt = df["group"].value_counts()
    for chunk in [group_order[i:i + 4] for i in range(0, len(group_order), 4)]:
        cols = st.columns(len(chunk))
        for col, g in zip(cols, chunk):
            col.markdown(
                f'<div class="gcard" style="--accent:{cmap[g]}"><div class="gname">{g}</div>'
                f'<div class="gcount">{cnt[g]} countries</div>'
                f'<div class="gstat">Child mortality <b>{gm.loc[g, "child_mort"]:.0f}</b> / 1000</div>'
                f'<div class="gstat">Income <b>${gm.loc[g, "income"]:,.0f}</b></div>'
                f'<div class="gstat">Life expectancy <b>{gm.loc[g, "life_expec"]:.0f} yrs</b></div></div>',
                unsafe_allow_html=True)

    z = Xs[FEATURES].groupby(df["group"].values).mean().loc[group_order]
    z = z.rename(columns=SHORT)
    left, right = st.columns(2)
    with left:
        hm = px.imshow(z, color_continuous_scale="RdBu_r", zmin=-2.5, zmax=2.5, text_auto=".1f",
                       aspect="auto", title="Feature averages (red = high, blue = low)")
        hm.update_layout(coloraxis_showscale=False)
        show(style(hm, 460))
    with right:
        rz = (z - z.min()) / (z.max() - z.min()).replace(0, 1)
        rd = go.Figure()
        for g in group_order:
            vals = list(rz.loc[g]) + [rz.loc[g].iloc[0]]
            rd.add_trace(go.Scatterpolar(r=vals, theta=list(rz.columns) + [rz.columns[0]], name=g,
                                         fill="toself", fillcolor=rgba(cmap[g], 0.20),
                                         line=dict(color=cmap[g], width=2)))
        rd.update_layout(title="Group fingerprint (radar)",
                         polar=dict(bgcolor="rgba(0,0,0,0)",
                                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                                                    gridcolor="#334155"),
                                    angularaxis=dict(gridcolor="#334155")))
        show(style(rd, 460))

    for g in group_order:
        with st.expander(f"Countries in {g}"):
            sub = df[df["group"] == g].sort_values("need_score", ascending=False)
            st.write(", ".join(sub["country"]))

# ================================================================ TAB 4
with tab4:
    country = st.selectbox("Select a country", sorted(df["country"]))
    idx = df.index[df["country"] == country][0]
    row = df.loc[idx]
    g = row["group"]
    rank = int(df["need_score"].rank(ascending=False)[idx])
    st.markdown(f'<div class="country-head"><div class="country-name">{country}</div>'
                f'<span class="badge" style="background:{cmap[g]}">{g}</span></div>', unsafe_allow_html=True)

    gavg = df[df["group"] == g][FEATURES].mean()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Need rank", f"#{rank} of {len(df)}")
    m2.metric("Child mortality", f"{row['child_mort']:.1f}",
              f"{row['child_mort'] - gavg['child_mort']:+.1f} vs group", delta_color="inverse")
    m3.metric("Income", f"${row['income']:,.0f}", f"{row['income'] - gavg['income']:+,.0f} vs group")
    m4.metric("Life expectancy", f"{row['life_expec']:.1f}",
              f"{row['life_expec'] - gavg['life_expec']:+.1f} vs group")

    pct = df[FEATURES].rank(pct=True) * 100
    gp = pct[df["group"] == g].mean()
    labels = [SHORT[f] for f in FEATURES]
    bar = go.Figure()
    bar.add_bar(y=labels, x=[pct.loc[idx, f] for f in FEATURES], orientation="h",
                name=country, marker_color=cmap[g])
    bar.add_bar(y=labels, x=[gp[f] for f in FEATURES], orientation="h",
                name=f"{g} average", marker_color="#64748b")
    bar.update_layout(barmode="group", title="Percentile among all countries (100 = highest value)",
                      xaxis=dict(range=[0, 100]), yaxis=dict(autorange="reversed"))
    show(style(bar, 480))

    X = Xs[FEATURES].values
    d = np.linalg.norm(X - X[idx], axis=1)
    near = np.argsort(d)[1:6]
    sim = pd.DataFrame({"Country": df.loc[near, "country"].values,
                        "Group": df.loc[near, "group"].values,
                        "Similarity": (1 - d[near] / d.max()) * 100})
    st.markdown("#### Most similar countries")
    st.dataframe(sim, hide_index=True,
                 column_config={"Similarity": st.column_config.ProgressColumn(
                     "Similarity", min_value=0, max_value=100, format="%.0f%%")}, **DF_KW)

# ================================================================ TAB 5
with tab5:
    n = st.slider("How many countries to show?", 5, 40, 15)
    top = df.sort_values("need_score", ascending=False).head(n)
    f5 = px.bar(top.iloc[::-1], x="need_idx", y="country", color="group", orientation="h",
                color_discrete_map=cmap, category_orders={"group": group_order},
                title=f"Top {n} countries by need", labels={"need_idx": "Need index (0 to 100)", "country": ""})
    show(style(f5, max(380, n * 26)))

    tbl = top[["country", "group", "need_idx", "child_mort", "income", "life_expec"]].copy()
    tbl.insert(0, "rank", range(1, len(tbl) + 1))
    st.dataframe(tbl, hide_index=True, column_config={
        "rank": "#", "country": "Country", "group": "Group",
        "need_idx": st.column_config.ProgressColumn("Need index", min_value=0, max_value=100, format="%.0f"),
        "child_mort": st.column_config.NumberColumn("Child mortality", format="%.1f"),
        "income": st.column_config.NumberColumn("Income ($)", format="%d"),
        "life_expec": st.column_config.NumberColumn("Life expectancy", format="%.1f"),
    }, **DF_KW)
    st.download_button("⬇️ Download full results (CSV)",
                       df.drop(columns=["raw", "map_name"]).to_csv(index=False),
                       "country_clusters.csv", "text/csv")

st.markdown("---")
st.caption("Built with SciPy, scikit-learn, Plotly and Streamlit · Data: Kaggle, Unsupervised Learning on Country Data")