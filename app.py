import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSense AI · Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --navy:  #0A1628;
    --blue:  #1565C0;
    --sky:   #1E88E5;
    --gold:  #FFB300;
    --green: #00C853;
    --red:   #FF1744;
    --paper: #F0F4F8;
    --card:  #FFFFFF;
    --ink:   #1A2332;
    --muted: #607D8B;
    --border:#DDE6ED;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--paper);
    color: var(--ink);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 1.5rem 4rem !important; max-width: 1280px !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0A1628 0%, #0D47A1 55%, #1565C0 100%);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:'';
    position:absolute; top:-80px; right:-80px;
    width:320px; height:320px; border-radius:50%;
    background:rgba(255,179,0,0.08);
}
.hero::after {
    content:'';
    position:absolute; bottom:-60px; left:30%;
    width:200px; height:200px; border-radius:50%;
    background:rgba(255,255,255,0.04);
}
.hero-tag {
    display:inline-block;
    background:rgba(255,179,0,0.2);
    color:#FFB300;
    border:1px solid rgba(255,179,0,0.4);
    border-radius:100px;
    padding:0.25rem 0.9rem;
    font-size:0.72rem;
    font-weight:600;
    letter-spacing:0.12em;
    text-transform:uppercase;
    margin-bottom:1rem;
}
.hero h1 {
    font-family:'DM Serif Display',serif;
    font-size:2.8rem; font-weight:400;
    color:#FFFFFF;
    line-height:1.15;
    margin:0 0 0.9rem;
}
.hero h1 em { color:#FFB300; font-style:normal; }
.hero p { color:rgba(255,255,255,0.75); font-size:1rem; font-weight:300; max-width:520px; line-height:1.75; margin:0; }
.hero-stats { display:flex; gap:2rem; margin-top:1.8rem; }
.hero-stat-val { font-family:'DM Serif Display',serif; font-size:1.6rem; color:#fff; }
.hero-stat-lbl { font-size:0.72rem; color:rgba(255,255,255,0.55); letter-spacing:0.08em; text-transform:uppercase; margin-top:0.1rem; }

/* ── Cards ── */
.card {
    background:var(--card);
    border-radius:16px;
    padding:1.6rem 1.8rem;
    box-shadow:0 2px 16px rgba(10,22,40,0.08);
    border:1px solid var(--border);
    margin-bottom:1.2rem;
    height: 100%;
}
.card-title {
    font-size:0.7rem; font-weight:700;
    letter-spacing:0.12em; text-transform:uppercase;
    color:var(--muted); margin-bottom:1.2rem;
    padding-bottom:0.6rem;
    border-bottom:2px solid var(--paper);
}

/* ── Predict button ── */
div.stButton > button {
    background: linear-gradient(135deg, #0D47A1 0%, #1976D2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.85rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 20px rgba(13,71,161,0.35);
    margin-top:0.5rem;
}
div.stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 28px rgba(13,71,161,0.4); }

/* ── Result banner ── */
.result-box {
    border-radius:18px;
    padding:2.5rem 2rem;
    text-align:center;
    margin-top:0.5rem;
    position:relative;
    overflow:hidden;
}
.result-approved { background: linear-gradient(135deg,#004D2C,#00796B); box-shadow:0 12px 40px rgba(0,121,107,0.4); }
.result-rejected { background: linear-gradient(135deg,#7B0000,#C62828); box-shadow:0 12px 40px rgba(198,40,40,0.4); }
.result-box .icon { font-size:3.2rem; margin-bottom:0.5rem; }
.result-box .verdict {
    font-family:'DM Serif Display',serif;
    font-size:2.2rem; color:#fff;
    margin:0 0 0.4rem;
}
.result-box .sub { color:rgba(255,255,255,0.8); font-size:0.9rem; font-weight:300; margin:0; }

/* confidence bar */
.conf-wrap { margin:1.4rem auto 0; max-width:320px; }
.conf-track { background:rgba(255,255,255,0.2); border-radius:100px; height:10px; overflow:hidden; margin:0.5rem 0; }
.conf-fill   { height:10px; border-radius:100px; background:#fff; }
.conf-lbl    { font-size:0.75rem; color:rgba(255,255,255,0.65); text-align:center; }

/* chips */
.chips { display:flex; flex-wrap:wrap; gap:0.5rem; justify-content:center; margin-top:1.2rem; }
.chip  { background:rgba(255,255,255,0.15); border-radius:100px; padding:0.3rem 0.9rem; font-size:0.78rem; color:#fff; font-weight:500; }

/* ── Metric tiles ── */
.metric-row { display:flex; gap:0.8rem; flex-wrap:wrap; margin-bottom:1rem; }
.metric-tile {
    flex:1; min-width:130px;
    background:var(--card);
    border:1px solid var(--border);
    border-radius:12px;
    padding:1rem 1.2rem;
    box-shadow:0 1px 8px rgba(10,22,40,0.06);
}
.metric-tile .mt-lbl { font-size:0.7rem; color:var(--muted); font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.3rem; }
.metric-tile .mt-val { font-family:'DM Serif Display',serif; font-size:1.5rem; color:var(--ink); }
.metric-tile .mt-sub { font-size:0.72rem; color:var(--muted); margin-top:0.15rem; }

/* ── Section divider ── */
.divider { border:none; border-top:2px solid var(--border); margin:2rem 0; }

/* ── Tab label ── */
.sec-lbl {
    font-size:0.68rem; font-weight:700;
    letter-spacing:0.14em; text-transform:uppercase;
    color:var(--muted); margin:0 0 0.8rem;
}

/* slider label override */
div[data-testid="stSlider"] > label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stNumberInput"] > label {
    font-size:0.82rem !important;
    font-weight:600 !important;
    color:var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt_inr(val):
    """Format number as Indian Rupees with L/Cr suffix."""
    if val >= 1_00_00_000:
        return f"₹{val/1_00_00_000:.1f} Cr"
    elif val >= 1_00_000:
        return f"₹{val/1_00_000:.1f} L"
    else:
        return f"₹{val:,.0f}"

PLOTLY_LAYOUT = dict(
    font_family="Inter",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=30, b=10),
)
COLORS = dict(approved="#00897B", rejected="#E53935", gold="#FFB300", blue="#1565C0", sky="#42A5F5")


# ── Load model & data ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load("ada_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()
    return df

try:
    model = load_model()
    df_raw = load_data()
    data_ok = True
except Exception as e:
    data_ok = False
    st.error(f"Error loading files: {e}")
    st.stop()


# ── HERO ───────────────────────────────────────────────────────────────────────
approved_count = (df_raw["loan_status"] == "Approved").sum()
total_count    = len(df_raw)
approval_rate  = approved_count / total_count * 100

st.markdown(f"""
<div class="hero">
  <div class="hero-tag">🏦 AdaBoost · ML Powered</div>
  <h1>Will your loan<br>be <em>Approved?</em></h1>
  <p>Enter your complete financial profile. Our model analyses 15 key indicators — including CIBIL score, assets, income, and loan structure — to predict your outcome instantly.</p>
  <div class="hero-stats">
    <div>
      <div class="hero-stat-val">{total_count:,}</div>
      <div class="hero-stat-lbl">Training Records</div>
    </div>
    <div>
      <div class="hero-stat-val">{approval_rate:.1f}%</div>
      <div class="hero-stat-lbl">Historical Approval Rate</div>
    </div>
    <div>
      <div class="hero-stat-val">15</div>
      <div class="hero-stat-lbl">Features Analysed</div>
    </div>
    <div>
      <div class="hero-stat-val">200</div>
      <div class="hero-stat-lbl">Boosted Estimators</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT FORM
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📋 Your Loan Application")

col_a, col_b, col_c = st.columns([1, 1, 1], gap="medium")

with col_a:
    st.markdown('<p class="sec-lbl">👤 Personal Details</p>', unsafe_allow_html=True)
    no_of_dependents = st.selectbox("Number of Dependents", [0,1,2,3,4,5], index=1)
    education        = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed    = st.selectbox("Employment", ["Salaried", "Self-Employed"])
    cibil_score      = st.slider("CIBIL Credit Score", 300, 900, 650, 1,
                                  help="300 = poor  |  600 = fair  |  750+ = excellent")

with col_b:
    st.markdown('<p class="sec-lbl">💰 Loan Details (₹)</p>', unsafe_allow_html=True)
    income_annum = st.number_input("Annual Income (₹)", 200_000, 9_900_000, 3_000_000, 50_000,
                                    format="%d", help="Your yearly gross income")
    loan_amount  = st.number_input("Loan Amount (₹)",   300_000, 39_500_000, 8_000_000, 100_000,
                                    format="%d")
    loan_term    = st.slider("Loan Term (Years)", 2, 20, 10)

    st.markdown('<p class="sec-lbl" style="margin-top:1rem">🏘️ Residential Assets (₹)</p>', unsafe_allow_html=True)
    residential_assets = st.number_input("Residential Assets (₹)", 0, 29_100_000, 5_000_000, 100_000, format="%d")

with col_c:
    st.markdown('<p class="sec-lbl">🏢 Other Assets (₹)</p>', unsafe_allow_html=True)
    commercial_assets = st.number_input("Commercial Assets (₹)",  0, 19_400_000, 2_000_000, 100_000, format="%d")
    luxury_assets     = st.number_input("Luxury Assets (₹)",      300_000, 39_200_000, 1_500_000, 100_000, format="%d")
    bank_asset        = st.number_input("Bank / Liquid Assets (₹)", 0, 14_700_000, 3_000_000, 100_000, format="%d")

    # Live quick metrics
    total_assets = residential_assets + commercial_assets + luxury_assets + bank_asset
    dti          = (loan_amount / loan_term) / income_annum * 100
    loan_to_inc  = loan_amount / income_annum

    st.markdown(f"""
    <div class="metric-row" style="margin-top:1.4rem">
      <div class="metric-tile">
        <div class="mt-lbl">Total Assets</div>
        <div class="mt-val">{fmt_inr(total_assets)}</div>
        <div class="mt-sub">All asset classes</div>
      </div>
      <div class="metric-tile">
        <div class="mt-lbl">Loan / Income</div>
        <div class="mt-val">{loan_to_inc:.1f}×</div>
        <div class="mt-sub">Ratio</div>
      </div>
    </div>
    <div class="metric-row">
      <div class="metric-tile">
        <div class="mt-lbl">Annual Repayment</div>
        <div class="mt-val">{fmt_inr(loan_amount/loan_term)}</div>
        <div class="mt-sub">Est. per year</div>
      </div>
      <div class="metric-tile">
        <div class="mt-lbl">Debt-to-Income</div>
        <div class="mt-val">{dti:.1f}%</div>
        <div class="mt-sub">DTI ratio</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("🔍  Predict My Loan Approval", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if predict_btn:
    dep_ohe     = [1 if no_of_dependents == i else 0 for i in range(1, 6)]
    edu_ng      = 1 if education == "Not Graduate" else 0
    self_emp_y  = 1 if self_employed == "Self-Employed" else 0

    features = np.array([[
        income_annum, loan_amount, loan_term, cibil_score,
        residential_assets, commercial_assets, luxury_assets, bank_asset,
        *dep_ohe, edu_ng, self_emp_y
    ]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred  = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

    approved = (pred == 1)
    conf     = float(proba[1] if approved else proba[0]) * 100
    prob_app = float(proba[1]) * 100
    prob_rej = float(proba[0]) * 100

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Prediction Result")

    res_col, gap_col = st.columns([1, 0.02])
    with res_col:
        css_cls = "result-approved" if approved else "result-rejected"
        icon    = "✅" if approved else "❌"
        verdict = "Loan Approved" if approved else "Loan Rejected"
        sub     = "Your profile meets the approval criteria." if approved else "Your profile does not meet the current approval threshold."

        st.markdown(f"""
        <div class="result-box {css_cls}">
          <div class="icon">{icon}</div>
          <p class="verdict">{verdict}</p>
          <p class="sub">{sub}</p>
          <div class="conf-wrap">
            <div class="conf-lbl">Model Confidence</div>
            <div class="conf-track"><div class="conf-fill" style="width:{conf:.0f}%"></div></div>
            <div class="conf-lbl">{conf:.1f}%</div>
          </div>
          <div class="chips">
            <span class="chip">CIBIL {cibil_score}</span>
            <span class="chip">{fmt_inr(income_annum)} income</span>
            <span class="chip">{fmt_inr(loan_amount)} loan</span>
            <span class="chip">{loan_term} yr term</span>
            <span class="chip">{fmt_inr(total_assets)} assets</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # CHARTS SECTION
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📊 Full Analytics Dashboard")

    # ── Row 1: Approval probability gauge + feature importance ────────────────
    r1c1, r1c2 = st.columns(2, gap="medium")

    with r1c1:
        st.markdown('<div class="card"><div class="card-title">Approval Probability Gauge</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_app,
            delta={"reference": 50, "valueformat": ".1f", "suffix": "%"},
            number={"suffix": "%", "font": {"size": 38, "family": "DM Serif Display"}},
            title={"text": "Approval Probability", "font": {"size": 14, "color": "#607D8B"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 11}},
                "bar":  {"color": COLORS["approved"] if approved else COLORS["rejected"], "thickness": 0.28},
                "steps": [
                    {"range": [0, 40],  "color": "rgba(229,57,53,0.12)"},
                    {"range": [40, 60], "color": "rgba(255,179,0,0.12)"},
                    {"range": [60, 100],"color": "rgba(0,137,123,0.12)"},
                ],
                "threshold": {"line": {"color": COLORS["gold"], "width": 3}, "thickness": 0.75, "value": 50},
            }
        ))
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="card"><div class="card-title">Feature Importance (Model Weights)</div>', unsafe_allow_html=True)
        feat_names = [
            'Annual Income','Loan Amount','Loan Term','CIBIL Score',
            'Residential Assets','Commercial Assets','Luxury Assets','Bank Assets',
            'Dependents 1','Dependents 2','Dependents 3','Dependents 4','Dependents 5',
            'Not Graduate','Self Employed'
        ]
        imps = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": imps}) \
                  .sort_values("Importance", ascending=True).tail(10)

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0,"#1565C0"],[0.5,"#FFB300"],[1,"#00897B"]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in fi_df["Importance"]],
            textposition="outside",
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, height=280,
            xaxis_title="Importance", yaxis_title="",
            xaxis=dict(showgrid=True, gridcolor="#EEF2F5"))
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Input values radar + Approved vs Rejected donut ───────────────
    r2c1, r2c2 = st.columns(2, gap="medium")

    with r2c1:
        st.markdown('<div class="card"><div class="card-title">Your Financial Profile (₹ Normalised)</div>', unsafe_allow_html=True)
        # Normalise to 0-100 for radar
        radar_vals = {
            "Annual Income":     income_annum / 9_900_000 * 100,
            "Loan Amount":       loan_amount  / 39_500_000 * 100,
            "CIBIL Score":       (cibil_score - 300) / 600 * 100,
            "Residential Assets":residential_assets / 29_100_000 * 100,
            "Commercial Assets": commercial_assets  / 19_400_000 * 100,
            "Luxury Assets":     luxury_assets      / 39_200_000 * 100,
            "Bank Assets":       bank_asset         / 14_700_000 * 100,
        }
        cats = list(radar_vals.keys())
        vals = list(radar_vals.values())
        vals_closed = vals + [vals[0]]
        cats_closed = cats + [cats[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed, fill="toself",
            fillcolor=f"rgba(21,101,192,0.18)",
            line=dict(color=COLORS["blue"], width=2.5),
            name="Your Profile",
            hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
        ))
        fig_radar.update_layout(**PLOTLY_LAYOUT, height=300,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[0,100], showticklabels=True, tickfont=dict(size=9), gridcolor="#DDE6ED"),
                angularaxis=dict(tickfont=dict(size=10)),
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="card"><div class="card-title">Approval vs Rejection — Dataset Overview</div>', unsafe_allow_html=True)
        app_count = int((df_raw["loan_status"] == "Approved").sum())
        rej_count = int((df_raw["loan_status"] == "Rejected").sum())
        fig_donut = go.Figure(go.Pie(
            labels=["Approved", "Rejected"],
            values=[app_count, rej_count],
            hole=0.6,
            marker_colors=[COLORS["approved"], COLORS["rejected"]],
            textinfo="percent+label",
            textfont_size=13,
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{approval_rate:.0f}%</b><br>Approved",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, family="DM Serif Display", color=COLORS["approved"]),
            align="center",
        )
        fig_donut.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5))
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: CIBIL score distribution + Income vs Loan Amount scatter ───────
    r3c1, r3c2 = st.columns(2, gap="medium")

    with r3c1:
        st.markdown('<div class="card"><div class="card-title">CIBIL Score Distribution by Outcome</div>', unsafe_allow_html=True)
        df_app = df_raw[df_raw["loan_status"] == "Approved"]["cibil_score"]
        df_rej = df_raw[df_raw["loan_status"] == "Rejected"]["cibil_score"]
        fig_cibil = go.Figure()
        fig_cibil.add_trace(go.Histogram(x=df_app, name="Approved", nbinsx=30,
            marker_color=COLORS["approved"], opacity=0.75))
        fig_cibil.add_trace(go.Histogram(x=df_rej, name="Rejected", nbinsx=30,
            marker_color=COLORS["rejected"], opacity=0.75))
        fig_cibil.add_vline(x=cibil_score, line_dash="dash", line_color=COLORS["gold"],
            line_width=2.5, annotation_text=f"You: {cibil_score}",
            annotation_font_color=COLORS["gold"], annotation_font_size=12)
        fig_cibil.update_layout(**PLOTLY_LAYOUT, height=280, barmode="overlay",
            xaxis_title="CIBIL Score", yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="#EEF2F5"),
            yaxis=dict(showgrid=True, gridcolor="#EEF2F5"),
        )
        st.plotly_chart(fig_cibil, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r3c2:
        st.markdown('<div class="card"><div class="card-title">Income vs Loan Amount (₹) — Scatter</div>', unsafe_allow_html=True)
        sample = df_raw.sample(min(600, len(df_raw)), random_state=42)
        colors_scatter = [COLORS["approved"] if s == "Approved" else COLORS["rejected"] for s in sample["loan_status"]]
        fig_sc = go.Figure()
        for status, color in [("Approved", COLORS["approved"]), ("Rejected", COLORS["rejected"])]:
            sub = sample[sample["loan_status"] == status]
            fig_sc.add_trace(go.Scatter(
                x=sub["income_annum"], y=sub["loan_amount"],
                mode="markers", name=status,
                marker=dict(color=color, size=5, opacity=0.6),
                hovertemplate=f"<b>{status}</b><br>Income: ₹%{{x:,.0f}}<br>Loan: ₹%{{y:,.0f}}<extra></extra>",
            ))
        # Your position
        fig_sc.add_trace(go.Scatter(
            x=[income_annum], y=[loan_amount],
            mode="markers+text", name="You",
            marker=dict(color=COLORS["gold"], size=14, symbol="star", line=dict(color="#fff", width=2)),
            text=["  You"], textfont=dict(color=COLORS["gold"], size=12, family="Inter"),
            textposition="middle right",
            hovertemplate=f"<b>You</b><br>Income: ₹{income_annum:,.0f}<br>Loan: ₹{loan_amount:,.0f}<extra></extra>",
        ))
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=280,
            xaxis_title="Annual Income (₹)", yaxis_title="Loan Amount (₹)",
            xaxis=dict(showgrid=True, gridcolor="#EEF2F5", tickformat=",.0f"),
            yaxis=dict(showgrid=True, gridcolor="#EEF2F5", tickformat=",.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 4: Asset breakdown waterfall + Loan term approval bar ─────────────
    r4c1, r4c2 = st.columns(2, gap="medium")

    with r4c1:
        st.markdown('<div class="card"><div class="card-title">Your Asset Breakdown (₹)</div>', unsafe_allow_html=True)
        asset_labels  = ["Residential", "Commercial", "Luxury", "Bank / Liquid", "Total"]
        asset_values  = [residential_assets, commercial_assets, luxury_assets, bank_asset, total_assets]
        asset_colors  = [COLORS["blue"], COLORS["sky"], COLORS["gold"], COLORS["approved"], "#7B1FA2"]

        fig_assets = go.Figure(go.Bar(
            x=asset_labels, y=asset_values,
            marker_color=asset_colors,
            text=[fmt_inr(v) for v in asset_values],
            textposition="outside",
            hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
        ))
        fig_assets.update_layout(**PLOTLY_LAYOUT, height=280,
            yaxis_title="Value (₹)", yaxis_tickformat=",.0f",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#EEF2F5"),
        )
        st.plotly_chart(fig_assets, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r4c2:
        st.markdown('<div class="card"><div class="card-title">Loan Term vs Approval Rate (Dataset)</div>', unsafe_allow_html=True)
        term_df = df_raw.groupby("loan_term")["loan_status"] \
                    .apply(lambda x: (x == "Approved").mean() * 100).reset_index()
        term_df.columns = ["Loan Term", "Approval Rate (%)"]

        fig_term = go.Figure()
        fig_term.add_trace(go.Bar(
            x=term_df["Loan Term"], y=term_df["Approval Rate (%)"],
            marker=dict(
                color=term_df["Approval Rate (%)"],
                colorscale=[[0, COLORS["rejected"]], [0.5, COLORS["gold"]], [1, COLORS["approved"]]],
                showscale=False,
            ),
            text=[f"{v:.0f}%" for v in term_df["Approval Rate (%)"]],
            textposition="outside",
            hovertemplate="Term: %{x} yrs<br>Approval Rate: %{y:.1f}%<extra></extra>",
        ))
        fig_term.add_vline(x=loan_term, line_dash="dash", line_color=COLORS["gold"],
            line_width=2, annotation_text=f"Your term: {loan_term}yr",
            annotation_font_color=COLORS["gold"], annotation_font_size=11)
        fig_term.update_layout(**PLOTLY_LAYOUT, height=280,
            xaxis_title="Loan Term (Years)", yaxis_title="Approval Rate (%)",
            xaxis=dict(showgrid=False, dtick=2),
            yaxis=dict(showgrid=True, gridcolor="#EEF2F5", range=[0, 110]),
        )
        st.plotly_chart(fig_term, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 5: Full applicant summary table ───────────────────────────────────
    st.markdown("#### 📋 Full Application Summary")
    summary_data = {
        "Parameter": [
            "Annual Income", "Loan Amount Requested", "Loan Term",
            "CIBIL Score", "Residential Assets", "Commercial Assets",
            "Luxury Assets", "Bank Assets", "Total Assets",
            "Loan-to-Income Ratio", "Annual Repayment (Est.)", "Debt-to-Income",
            "Dependents", "Education", "Employment",
            "🎯 Prediction", "📊 Approval Probability", "🔴 Rejection Probability"
        ],
        "Value": [
            fmt_inr(income_annum), fmt_inr(loan_amount), f"{loan_term} Years",
            f"{cibil_score} / 900", fmt_inr(residential_assets), fmt_inr(commercial_assets),
            fmt_inr(luxury_assets), fmt_inr(bank_asset), fmt_inr(total_assets),
            f"{loan_to_inc:.2f}×", fmt_inr(loan_amount / loan_term), f"{dti:.1f}%",
            str(no_of_dependents), education, self_employed,
            "✅ Approved" if approved else "❌ Rejected",
            f"{prob_app:.1f}%", f"{prob_rej:.1f}%"
        ]
    }
    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
            "Value":     st.column_config.TextColumn("Value",     width="medium"),
        }
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;font-size:0.74rem;color:#90A4AE;">
  LoanSense AI &nbsp;·&nbsp; Powered by AdaBoostClassifier (sklearn) &nbsp;·&nbsp;
  Trained on 4,269 loan records &nbsp;·&nbsp; For educational purposes only.<br>
  This tool does not constitute financial or banking advice.
</div>
""", unsafe_allow_html=True)