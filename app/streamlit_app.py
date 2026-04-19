import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append("utils")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy | Profitability Dashboard",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .block-container { padding: 2rem 2.5rem 2rem 2.5rem; }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 600;
        color: #111;
        line-height: 1.1;
        font-family: 'DM Mono', monospace;
    }
    .kpi-delta {
        font-size: 12px;
        color: #27ae60;
        margin-top: 4px;
    }
    .kpi-delta.neg { color: #e74c3c; }

    /* Section headers */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #555;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 8px;
        margin: 1.5rem 0 1rem 0;
    }

    /* Action badges */
    .badge-healthy      { background:#e8f8f0; color:#1a7a46; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500; }
    .badge-reprice      { background:#fff4e0; color:#b35900; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500; }
    .badge-cost         { background:#fde8e8; color:#b31b1b; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500; }
    .badge-discontinue  { background:#f0e8ff; color:#5a1b8e; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #f7f7f5; border-right: 1px solid #e8e8e8; }
    [data-testid="stSidebar"] .stMarkdown h3 { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: #888; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #111 !important; }

    /* Hide default Streamlit header */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA LOADING & CLEANING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("notebook/Nassau Candy Distributor.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  format="%d-%m-%Y")
    df["Product Name"] = df["Product Name"].str.replace(
        "Wonka Bar -Scrumdiddlyumptious",
        "Wonka Bar - Scrumdiddlyumptious"
    )
    df = df[df["Sales"] > 0]
    df = df[df["Units"] > 0]
    df["Gross_Margin"] = df["Gross Profit"] / df["Sales"]
    return df


# ─────────────────────────────────────────────
#  METRICS FUNCTIONS  (metrics.py logic inline)
# ─────────────────────────────────────────────
def calculate_product_metrics(df):
    pm = df.groupby("Product Name").agg(
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Units=("Units", "sum"),
        Cost=("Cost", "sum")
    ).reset_index()
    pm["Gross_Margin_%"]         = (pm["Gross_Profit"] / pm["Sales"]) * 100
    pm["Profit_per_Unit"]        = pm["Gross_Profit"] / pm["Units"]
    pm["Revenue_Contribution_%"] = (pm["Sales"] / pm["Sales"].sum()) * 100
    pm["Profit_Contribution_%"]  = (pm["Gross_Profit"] / pm["Gross_Profit"].sum()) * 100
    pm["Cost_Ratio"]             = pm["Cost"] / pm["Sales"]

    avg_profit = pm["Gross_Profit"].mean()
    avg_margin = pm["Gross_Margin_%"].mean()
    avg_sales  = pm["Sales"].mean()

    def segment(row):
        if row["Gross_Profit"] > avg_profit and row["Gross_Margin_%"] > avg_margin:
            return "High Profit / High Margin"
        elif row["Sales"] > avg_sales and row["Gross_Margin_%"] < avg_margin:
            return "High Sales / Low Margin"
        elif row["Sales"] < avg_sales and row["Gross_Profit"] < avg_profit:
            return "Low Sales / Low Profit"
        else:
            return "Average"

    pm["Segment"] = pm.apply(segment, axis=1)

    avg_cost_ratio = pm["Cost_Ratio"].mean()

    def action_flag(row):
        if row["Gross_Margin_%"] / 100 < avg_margin / 100 and row["Cost_Ratio"] > avg_cost_ratio:
            return "Cost Reduction / Renegotiation"
        elif row["Sales"] > avg_sales and row["Gross_Margin_%"] < avg_margin:
            return "Repricing Needed"
        elif row["Sales"] < avg_sales and row["Gross_Margin_%"] < avg_margin:
            return "Discontinuation Review"
        else:
            return "Healthy"

    pm["Action"] = pm.apply(action_flag, axis=1)
    return pm


def calculate_division_metrics(df):
    dm = df.groupby("Division").agg(
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Units=("Units", "sum"),
        Avg_Margin=("Gross_Margin", "mean")
    ).reset_index()
    dm["Gross_Margin_%"]  = dm["Avg_Margin"] * 100
    dm["Profit_Ratio"]    = dm["Gross_Profit"] / dm["Sales"]

    avg_margin = dm["Avg_Margin"].mean()
    avg_sales  = dm["Sales"].mean()

    def classify(row):
        if row["Avg_Margin"] >= avg_margin and row["Sales"] >= avg_sales:
            return "Strong Financial Efficiency"
        elif row["Avg_Margin"] < avg_margin and row["Sales"] >= avg_sales:
            return "Structural Margin Issues"
        else:
            return "Underutilized / Low Scale"

    dm["Category"] = dm.apply(classify, axis=1)
    return dm


def calculate_pareto(df):
    ps = df.groupby("Product Name").agg(
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum")
    ).reset_index()

    sp = ps.sort_values("Sales", ascending=False).copy().reset_index(drop=True)
    sp["Cum_Sales_%"] = sp["Sales"].cumsum() / sp["Sales"].sum() * 100
    sp["Product_Rank"] = sp.index + 1

    pp = ps.sort_values("Gross_Profit", ascending=False).copy().reset_index(drop=True)
    pp["Cum_Profit_%"] = pp["Gross_Profit"].cumsum() / pp["Gross_Profit"].sum() * 100
    pp["Product_Rank"] = pp.index + 1

    return sp, pp


def margin_volatility(df):
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)
    mv = df.groupby(["Month", "Product Name"])["Gross_Margin"].mean().reset_index()
    vol = mv.groupby("Product Name")["Gross_Margin"].std().reset_index()
    vol.columns = ["Product Name", "Margin_Volatility"]
    return vol.sort_values("Margin_Volatility", ascending=False)


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
try:
    raw_df = load_data()
except FileNotFoundError:
    st.error("⚠️  `Nassau Candy Distributor.csv` not found. Place it in the same folder as `app.py`.")
    st.stop()


# ─────────────────────────────────────────────
#  SIDEBAR — FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🍬 Nassau Candy")
    st.markdown("**Profitability Intelligence**")
    st.divider()

    st.markdown("### Date Range")
    min_date = raw_df["Order Date"].min().date()
    max_date = raw_df["Order Date"].max().date()
    date_range = st.date_input(
        "Select period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed"
    )

    st.markdown("### Division")
    all_divisions = sorted(raw_df["Division"].unique().tolist())
    selected_divisions = st.multiselect(
        "Select divisions",
        options=all_divisions,
        default=all_divisions,
        label_visibility="collapsed"
    )

    st.markdown("### Margin Threshold")
    margin_threshold = st.slider(
        "Min Gross Margin %",
        min_value=0, max_value=100, value=0, step=5,
        label_visibility="collapsed"
    )

    st.markdown("### Product Search")
    product_search = st.text_input(
        "Search product name",
        placeholder="e.g. Wonka Bar",
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Data: Nassau Candy Distributor")


# ─────────────────────────────────────────────
#  APPLY FILTERS
# ─────────────────────────────────────────────
if len(date_range) == 2:
    start_date, end_date = date_range
    df = raw_df[
        (raw_df["Order Date"].dt.date >= start_date) &
        (raw_df["Order Date"].dt.date <= end_date)
    ]
else:
    df = raw_df.copy()

if selected_divisions:
    df = df[df["Division"].isin(selected_divisions)]

product_df   = calculate_product_metrics(df)
division_df  = calculate_division_metrics(df)
sales_pareto, profit_pareto = calculate_pareto(df)
vol_df       = margin_volatility(df)

# Apply margin threshold filter to product_df
product_df_filtered = product_df[product_df["Gross_Margin_%"] >= margin_threshold]

# Apply product search
if product_search:
    product_df_filtered = product_df_filtered[
        product_df_filtered["Product Name"].str.contains(product_search, case=False)
    ]


# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────
COLORS = {
    "primary":   "#1a1a2e",
    "accent":    "#e63946",
    "green":     "#2ecc71",
    "amber":     "#f39c12",
    "purple":    "#8e44ad",
    "blue":      "#2980b9",
    "light_bg":  "#fafafa",
    "grid":      "#f0f0f0",
}
SEQ_PALETTE = ["#1a1a2e","#2c3e7a","#2980b9","#27ae60","#f39c12","#e63946","#8e44ad"]

PLOT_LAYOUT = dict(
    font_family="DM Sans",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=40, b=40, l=10, r=10),
    hoverlabel=dict(font_family="DM Sans", font_size=12),
    xaxis=dict(showgrid=False, linecolor="#e8e8e8"),
    yaxis=dict(gridcolor="#f0f0f0", linecolor="#e8e8e8"),
)


# ─────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("## 🍬 Nassau Candy — Profitability Dashboard")
st.caption(f"Showing data from **{df['Order Date'].min().strftime('%d %b %Y')}** to **{df['Order Date'].max().strftime('%d %b %Y')}**  ·  {len(df):,} records  ·  {df['Product Name'].nunique()} products  ·  {df['Division'].nunique()} divisions")
st.divider()


# ─────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────
total_revenue  = df["Sales"].sum()
total_profit   = df["Gross Profit"].sum()
avg_margin_pct = (total_profit / total_revenue) * 100
avg_ppu        = product_df["Profit_per_Unit"].mean()
top_product    = product_df.sort_values("Gross_Profit", ascending=False).iloc[0]["Product Name"]
vol_avg        = vol_df["Margin_Volatility"].mean()

k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, delta=None):
    delta_html = f'<div class="kpi-delta {"neg" if delta and delta.startswith("-") else ""}">{delta}</div>' if delta else ""
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

kpi(k1, "Total Revenue",     f"${total_revenue:,.0f}")
kpi(k2, "Total Gross Profit", f"${total_profit:,.0f}")
kpi(k3, "Avg Gross Margin",  f"{avg_margin_pct:.1f}%")
kpi(k4, "Avg Profit / Unit", f"${avg_ppu:.2f}")
kpi(k5, "Margin Volatility", f"{vol_avg:.3f}", delta="Std Dev across products")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📦  Product Profitability",
    "🏢  Division Performance",
    "🔍  Cost & Margin Diagnostics",
    "📊  Profit Concentration"
])


# ══════════════════════════════════════════════
#  TAB 1 — PRODUCT PROFITABILITY
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Product Margin Leaderboard</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])

    with c1:
        # Horizontal bar — top products by gross profit
        top_n = product_df_filtered.sort_values("Gross_Profit", ascending=True).tail(15)
        fig = go.Figure(go.Bar(
            x=top_n["Gross_Profit"],
            y=top_n["Product Name"],
            orientation="h",
            marker_color=top_n["Gross_Margin_%"],
            marker_colorscale=[[0,"#fde8e8"],[0.5,"#f39c12"],[1,"#27ae60"]],
            marker_showscale=True,
            marker_colorbar=dict(title="GM %", thickness=10, len=0.6),
            hovertemplate="<b>%{y}</b><br>Profit: $%{x:,.0f}<extra></extra>"
        ))
        fig.update_layout(**PLOT_LAYOUT, title="Gross Profit by Product (colour = Margin %)",
                        height=420, yaxis_title="", xaxis_title="Gross Profit ($)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Profit contribution donut
        top5 = product_df_filtered.sort_values("Gross_Profit", ascending=False).head(6)
        fig2 = go.Figure(go.Pie(
            labels=top5["Product Name"],
            values=top5["Profit_Contribution_%"],
            hole=0.55,
            marker_colors=SEQ_PALETTE,
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value:.1f}% of profit<extra></extra>"
        ))
        fig2.update_layout(**PLOT_LAYOUT, title="Profit Contribution — Top 6", height=420,
                        showlegend=True, legend=dict(font_size=11, orientation="v"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Segment Classification</div>', unsafe_allow_html=True)

    # Scatter — margin vs profit coloured by segment
    fig3 = px.scatter(
        product_df_filtered,
        x="Gross_Margin_%", y="Gross_Profit",
        size="Sales", color="Segment",
        hover_name="Product Name",
        color_discrete_map={
            "High Profit / High Margin": "#27ae60",
            "High Sales / Low Margin":   "#f39c12",
            "Low Sales / Low Profit":    "#e63946",
            "Average":                   "#95a5a6"
        },
        size_max=40,
        labels={"Gross_Margin_%": "Gross Margin (%)", "Gross_Profit": "Gross Profit ($)"}
    )
    fig3.update_layout(**PLOT_LAYOUT, title="Margin vs Profit (bubble size = Revenue)", height=380)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Full Product Table</div>', unsafe_allow_html=True)

    display_cols = ["Product Name","Sales","Gross_Profit","Gross_Margin_%","Profit_per_Unit",
                    "Revenue_Contribution_%","Profit_Contribution_%","Segment"]
    show_df = product_df_filtered[display_cols].sort_values("Gross_Profit", ascending=False)
    show_df = show_df.rename(columns={
        "Gross_Profit": "Gross Profit",
        "Gross_Margin_%": "Gross Margin %",
        "Profit_per_Unit": "Profit/Unit",
        "Revenue_Contribution_%": "Rev Contrib %",
        "Profit_Contribution_%": "Profit Contrib %"
    })
    st.dataframe(
        show_df.style.format({
            "Sales": "${:,.0f}", "Gross Profit": "${:,.0f}",
            "Gross Margin %": "{:.1f}%", "Profit/Unit": "${:.2f}",
            "Rev Contrib %": "{:.1f}%", "Profit Contrib %": "{:.1f}%"
        }).background_gradient(subset=["Gross Margin %"], cmap="RdYlGn"),
        use_container_width=True, height=380
    )


# ══════════════════════════════════════════════
#  TAB 2 — DIVISION PERFORMANCE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Revenue vs Profit by Division</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Grouped bar
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Revenue", x=division_df["Division"], y=division_df["Sales"],
                            marker_color=COLORS["blue"], opacity=0.85))
        fig.add_trace(go.Bar(name="Gross Profit", x=division_df["Division"], y=division_df["Gross_Profit"],
                            marker_color=COLORS["green"], opacity=0.85))
        fig.update_layout(**PLOT_LAYOUT, barmode="group", title="Revenue vs Gross Profit",
                        height=360, legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Margin by division — horizontal bar
        div_sorted = division_df.sort_values("Gross_Margin_%", ascending=True)
        colors = ["#e63946" if c == "Structural Margin Issues" else
                "#f39c12" if c == "Underutilized / Low Scale" else
                "#27ae60" for c in div_sorted["Category"]]
        fig2 = go.Figure(go.Bar(
            x=div_sorted["Gross_Margin_%"],
            y=div_sorted["Division"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Margin: %{x:.1f}%<extra></extra>"
        ))
        fig2.update_layout(**PLOT_LAYOUT, title="Average Gross Margin % by Division",
                        height=360, xaxis_title="Gross Margin (%)", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Division Classification</div>', unsafe_allow_html=True)

    for _, row in division_df.iterrows():
        badge = {
            "Strong Financial Efficiency": "🟢",
            "Structural Margin Issues":    "🔴",
            "Underutilized / Low Scale":   "🟡"
        }.get(row["Category"], "⚪")
        st.markdown(f"{badge} **{row['Division']}** — {row['Category']}  |  "
                    f"Revenue: ${row['Sales']:,.0f}  |  "
                    f"Profit: ${row['Gross_Profit']:,.0f}  |  "
                    f"Margin: {row['Gross_Margin_%']:.1f}%")

    st.markdown('<div class="section-header">Division Summary Table</div>', unsafe_allow_html=True)
    st.dataframe(
        division_df[["Division","Sales","Gross_Profit","Gross_Margin_%","Profit_Ratio","Category"]]
        .rename(columns={"Gross_Profit":"Gross Profit","Gross_Margin_%":"Gross Margin %","Profit_Ratio":"Profit Ratio"})
        .style.format({"Sales":"${:,.0f}","Gross Profit":"${:,.0f}","Gross Margin %":"{:.1f}%","Profit Ratio":"{:.2f}"}),
        use_container_width=True
    )

    st.markdown('<div class="section-header">Margin Volatility by Product</div>', unsafe_allow_html=True)
    fig3 = px.bar(
        vol_df.head(15).sort_values("Margin_Volatility", ascending=True),
        x="Margin_Volatility", y="Product Name",
        orientation="h", color="Margin_Volatility",
        color_continuous_scale=["#27ae60","#f39c12","#e63946"],
        labels={"Margin_Volatility": "Std Dev of Margin"}
    )
    fig3.update_layout(**PLOT_LAYOUT, title="Margin Volatility — Top 15 Most Volatile Products",
                    height=380, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 — COST & MARGIN DIAGNOSTICS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Cost vs Sales Scatter</div>', unsafe_allow_html=True)

    fig = px.scatter(
        product_df_filtered,
        x="Sales", y="Cost",
        color="Action",
        hover_name="Product Name",
        size="Units",
        size_max=35,
        color_discrete_map={
            "Healthy":                      "#27ae60",
            "Repricing Needed":             "#f39c12",
            "Cost Reduction / Renegotiation":"#e63946",
            "Discontinuation Review":        "#8e44ad"
        },
        labels={"Sales": "Total Sales ($)", "Cost": "Total Cost ($)"}
    )
    # 45° reference line
    max_val = max(product_df_filtered["Sales"].max(), product_df_filtered["Cost"].max())
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color="#aaa", dash="dot", width=1))
    fig.add_annotation(x=max_val*0.85, y=max_val*0.9,
                    text="Cost = Sales (0% margin)", showarrow=False,
                    font=dict(size=10, color="#aaa"))
    fig.update_layout(**PLOT_LAYOUT, title="Cost vs Sales (bubble size = Units sold)", height=440,
                    legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Margin Risk Flags</div>', unsafe_allow_html=True)
        action_counts = product_df_filtered["Action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        fig2 = px.pie(action_counts, names="Action", values="Count", hole=0.5,
                    color="Action",
                    color_discrete_map={
                        "Healthy":                       "#27ae60",
                        "Repricing Needed":              "#f39c12",
                        "Cost Reduction / Renegotiation":"#e63946",
                        "Discontinuation Review":        "#8e44ad"
                    })
        fig2.update_layout(**PLOT_LAYOUT, height=320, showlegend=True,
                        legend=dict(font_size=11))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Cost Ratio vs Gross Margin</div>', unsafe_allow_html=True)
        fig3 = px.scatter(
            product_df_filtered,
            x="Cost_Ratio", y="Gross_Margin_%",
            text="Product Name",
            color="Action",
            color_discrete_map={
                "Healthy":                      "#27ae60",
                "Repricing Needed":             "#f39c12",
                "Cost Reduction / Renegotiation":"#e63946",
                "Discontinuation Review":        "#8e44ad"
            }
        )
        fig3.update_traces(textposition="top center", textfont_size=9)
        fig3.update_layout(**PLOT_LAYOUT, height=320,
                        xaxis_title="Cost Ratio (Cost/Sales)",
                        yaxis_title="Gross Margin (%)", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Product Action Table</div>', unsafe_allow_html=True)

    def badge_html(action):
        classes = {
            "Healthy":                       "badge-healthy",
            "Repricing Needed":              "badge-reprice",
            "Cost Reduction / Renegotiation":"badge-cost",
            "Discontinuation Review":        "badge-discontinue"
        }
        return f'<span class="{classes.get(action, "")}">{action}</span>'

    action_df = product_df_filtered[["Product Name","Sales","Cost","Gross_Margin_%","Cost_Ratio","Action"]].copy()
    action_df = action_df.sort_values("Gross_Margin_%")
    action_df["Action Badge"] = action_df["Action"].apply(badge_html)

    st.dataframe(
        action_df[["Product Name","Sales","Cost","Gross_Margin_%","Cost_Ratio","Action"]]
        .rename(columns={"Gross_Margin_%":"Gross Margin %","Cost_Ratio":"Cost Ratio"})
        .style.format({"Sales":"${:,.0f}","Cost":"${:,.0f}","Gross Margin %":"{:.1f}%","Cost Ratio":"{:.2f}"}),
        use_container_width=True, height=340
    )


# ══════════════════════════════════════════════
#  TAB 4 — PROFIT CONCENTRATION (PARETO)
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Pareto Analysis — Profit Concentration</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Profit Pareto
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=profit_pareto["Product Name"], y=profit_pareto["Gross_Profit"],
            name="Gross Profit", marker_color=COLORS["blue"], opacity=0.75
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=profit_pareto["Product Name"], y=profit_pareto["Cum_Profit_%"],
            name="Cumulative %", mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2),
            marker=dict(size=5)
        ), secondary_y=True)
        fig.add_hline(y=80, line_dash="dot", line_color="#aaa", secondary_y=True,
                    annotation_text="80%", annotation_position="right")
        fig.update_layout(**PLOT_LAYOUT, title="Profit Pareto Chart", height=400,
                        legend=dict(orientation="h", y=1.1),
                        xaxis_tickangle=45)
        fig.update_yaxes(title_text="Gross Profit ($)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Revenue Pareto
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(
            x=sales_pareto["Product Name"], y=sales_pareto["Sales"],
            name="Revenue", marker_color=COLORS["purple"], opacity=0.75
        ), secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=sales_pareto["Product Name"], y=sales_pareto["Cum_Sales_%"],
            name="Cumulative %", mode="lines+markers",
            line=dict(color=COLORS["amber"], width=2),
            marker=dict(size=5)
        ), secondary_y=True)
        fig2.add_hline(y=80, line_dash="dot", line_color="#aaa", secondary_y=True,
                    annotation_text="80%", annotation_position="right")
        fig2.update_layout(**PLOT_LAYOUT, title="Revenue Pareto Chart", height=400,
                        legend=dict(orientation="h", y=1.1),
                        xaxis_tickangle=45)
        fig2.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig2.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])
        st.plotly_chart(fig2, use_container_width=True)

    # Dependency indicators
    st.markdown('<div class="section-header">Dependency Indicators</div>', unsafe_allow_html=True)

    top80_profit_products = profit_pareto[profit_pareto["Cum_Profit_%"] <= 80]
    top80_sales_products  = sales_pareto[sales_pareto["Cum_Sales_%"] <= 80]
    total_products        = len(profit_pareto)

    d1, d2, d3 = st.columns(3)
    d1.metric("Products driving 80% of Profit",
            f"{len(top80_profit_products)} of {total_products}",
            f"{len(top80_profit_products)/total_products*100:.0f}% of portfolio")
    d2.metric("Products driving 80% of Revenue",
            f"{len(top80_sales_products)} of {total_products}",
            f"{len(top80_sales_products)/total_products*100:.0f}% of portfolio")
    d3.metric("Top product profit share",
            f"{profit_pareto.iloc[0]['Cum_Profit_%']:.1f}%",
            profit_pareto.iloc[0]["Product Name"])

    st.markdown('<div class="section-header">Products Driving 80% of Profit</div>', unsafe_allow_html=True)
    st.dataframe(
        top80_profit_products[["Product_Rank","Product Name","Gross_Profit","Cum_Profit_%"]]
        .rename(columns={"Product_Rank":"Rank","Gross_Profit":"Gross Profit","Cum_Profit_%":"Cumulative Profit %"})
        .style.format({"Gross Profit":"${:,.0f}","Cumulative Profit %":"{:.1f}%"}),
        use_container_width=True, height=320
    )
