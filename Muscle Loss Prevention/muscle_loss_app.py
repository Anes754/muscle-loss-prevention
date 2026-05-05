import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Muscle Loss Risk Predictor",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0a0c0f;
    --surface:   #111318;
    --surface2:  #1a1d24;
    --border:    #2a2d36;
    --accent:    #f97316;
    --accent2:   #fb923c;
    --green:     #22c55e;
    --yellow:    #eab308;
    --red:       #ef4444;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --font-head: 'Bebas Neue', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* ── Reset ── */
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f1117 0%, #1a1320 50%, #0f1117 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(249,115,22,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: var(--font-head);
    font-size: 3.6rem;
    letter-spacing: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), #fcd34d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Step indicator ── */
.stepper {
    display: flex; align-items: center; gap: 0;
    margin-bottom: 2rem;
}
.step {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.55rem 1.2rem;
    border-radius: 99px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    transition: all 0.2s;
}
.step-active {
    background: var(--accent);
    color: #fff;
}
.step-done {
    background: rgba(34,197,94,0.15);
    color: var(--green);
    border: 1px solid rgba(34,197,94,0.3);
}
.step-inactive {
    background: var(--surface2);
    color: var(--muted);
    border: 1px solid var(--border);
}
.step-line {
    flex: 1; height: 1px;
    background: var(--border);
    margin: 0 0.4rem;
}

/* ── Section card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.card-title {
    font-family: var(--font-head);
    font-size: 1.6rem;
    letter-spacing: 1px;
    color: var(--accent);
    margin-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
}

/* ── Streamlit input overrides ── */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div:first-child,
div[data-baseweb="textarea"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="select"] input {
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}
label, .stSlider label, .stSelectbox label, .stNumberInput label {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    font-weight: 500 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-head) !important;
    font-size: 1.1rem !important;
    letter-spacing: 1.5px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(249,115,22,0.3) !important;
}
.stButton > button:hover {
    background: var(--accent2) !important;
    box-shadow: 0 6px 28px rgba(249,115,22,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: var(--font-mono);
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--accent);
}
.metric-unit {
    font-size: 0.75rem;
    color: var(--muted);
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.4rem 1.4rem;
    border-radius: 99px;
    font-family: var(--font-head);
    font-size: 1.4rem;
    letter-spacing: 2px;
}
.risk-low    { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid rgba(34,197,94,0.4); }
.risk-mod    { background: rgba(234,179,8,0.15);   color: #eab308; border: 1px solid rgba(234,179,8,0.4); }
.risk-high   { background: rgba(239,68,68,0.15);   color: #ef4444; border: 1px solid rgba(239,68,68,0.4); }

/* ── Alerts ── */
.stSuccess, .stInfo, .stWarning {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Slider ── */
.stSlider > div > div > div { background: var(--accent) !important; }

/* ── Selectbox dropdown ── */
ul[data-baseweb="menu"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
li[role="option"] { color: var(--text) !important; }
li[role="option"]:hover { background: var(--surface) !important; }

/* ── Plotly chart backgrounds ── */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ── Day selector tabs ── */
div[data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border-radius: 10px !important;
    gap: 4px !important;
    padding: 4px !important;
}
div[data-baseweb="tab"] {
    color: var(--muted) !important;
    border-radius: 8px !important;
}
div[aria-selected="true"][data-baseweb="tab"] {
    background: var(--accent) !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("muscle_loss_model.pkl")

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = 1
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "weekly_data" not in st.session_state:
    st.session_state.weekly_data = {
        "calories":   [0]*7,
        "duration":   [0]*7,
        "heart_rate": [0]*7,
        "intensity":  [0]*7
    }

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def step_html(current):
    steps = [("01","Profile"), ("02","Workouts"), ("03","Results"), ("04","Workout Plan")]
    parts = ['<div class="stepper">']
    for i, (num, label) in enumerate(steps, 1):
        if i < current:   cls = "step-done"
        elif i == current: cls = "step-active"
        else:             cls = "step-inactive"
        parts.append(f'<div class="step {cls}">{"✓" if i<current else num} {label}</div>')
        if i < 4:
            parts.append('<div class="step-line"></div>')
    parts.append('</div>')
    return "".join(parts)

def gauge_chart(value, title, color="#f97316"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#9ca3af", "size": 13, "family": "DM Sans"}},
        number={"font": {"color": "#e8eaf0", "size": 28, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#374151",
                     "tickfont": {"color": "#6b7280", "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1d24",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33],  "color": "rgba(34,197,94,0.12)"},
                {"range": [33, 66], "color": "rgba(234,179,8,0.12)"},
                {"range": [66, 100],"color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.8, "value": value}
        }
    ))
    fig.update_layout(
        height=220, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def weekly_bar(metric, values, color, ylab):
    fig = go.Figure(go.Bar(
        x=DAYS, y=values,
        marker=dict(
            color=values,
            colorscale=[[0,"#1a1d24"],[0.5, color],[1,"#fcd34d"]],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(color="#9ca3af", size=11, family="JetBrains Mono"),
    ))
    fig.update_layout(
        title=dict(text=metric, font=dict(color="#9ca3af", size=13, family="DM Sans")),
        xaxis=dict(tickfont=dict(color="#9ca3af", size=11), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title=ylab, tickfont=dict(color="#6b7280", size=10),
                   gridcolor="#1f2430", title_font=dict(color="#6b7280", size=11)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(t=40, b=10, l=10, r=10),
        showlegend=False
    )
    return fig

def radar_chart(categories, values, max_vals):
    norm = [v/m*100 for v, m in zip(values, max_vals)]
    fig = go.Figure(go.Scatterpolar(
        r=norm + [norm[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(249,115,22,0.15)',
        line=dict(color='#f97316', width=2),
        marker=dict(color='#f97316', size=6)
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100],
                            gridcolor="#2a2d36", tickfont=dict(color="#6b7280", size=9)),
            angularaxis=dict(tickfont=dict(color="#9ca3af", size=11), gridcolor="#2a2d36")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(t=30, b=30, l=30, r=30)
    )
    return fig

def bmi_zone_chart(bmi):
    zones = ["Underweight\n< 18.5","Normal\n18.5–24.9","Overweight\n25–29.9","Obese\n≥ 30"]
    colors = ["#3b82f6","#22c55e","#eab308","#ef4444"]
    ranges = [18.5, 24.9, 29.9, 40]
    bmi_clamped = min(max(bmi, 10), 40)
    fig = go.Figure()
    x_start = 10
    for i, (label, color, end) in enumerate(zip(zones, colors, ranges)):
        fig.add_shape(type="rect", x0=x_start, x1=end, y0=0, y1=1,
                      fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:],16)},0.18)",
                      line=dict(color=color, width=1))
        fig.add_annotation(x=(x_start+end)/2, y=0.5, text=label.replace("\n","<br>"),
                           showarrow=False, font=dict(color=color, size=10, family="DM Sans"),
                           align="center")
        x_start = end
    fig.add_vline(x=bmi_clamped, line=dict(color="#f97316", width=3, dash="dot"))
    fig.add_annotation(x=bmi_clamped, y=1.15,
                       text=f"<b>BMI {bmi:.1f}</b>",
                       showarrow=False, font=dict(color="#f97316", size=13, family="JetBrains Mono"))
    fig.update_layout(
        xaxis=dict(range=[10,40], showgrid=False, showticklabels=False),
        yaxis=dict(range=[-0.1,1.4], showgrid=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=120, margin=dict(t=30, b=10, l=10, r=10)
    )
    return fig

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">💪 MUSCLE LOSS RISK PREDICTOR</div>
    <div class="hero-sub">AI-Powered Fitness Analysis · Weekly Workout Intelligence</div>
</div>
""", unsafe_allow_html=True)

st.markdown(step_html(st.session_state.page), unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 1 — USER PROFILE
# ══════════════════════════════════════════════
if st.session_state.page == 1:

    st.markdown('<div class="card"><div class="card-title">USER PROFILE</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        user_id = st.number_input("User ID", min_value=0, step=1)
        name    = st.text_input("Full Name")
    with c2:
        age     = st.number_input("Age", 15, 80, 25)
        gender  = st.selectbox("Gender", ["Male","Female"])
    with c3:
        height  = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
        weight  = st.number_input("Current Weight (kg)", 30.0, 250.0, 70.0)

    goal_weight = st.slider("🎯 Goal Weight (kg)", float(int(weight*0.6)), float(int(weight*1.3)), float(weight))

    # Live BMI preview
    bmi_live = weight / ((height/100)**2)
    bmi_color = "#22c55e" if 18.5 <= bmi_live <= 24.9 else ("#eab308" if bmi_live < 30 else "#ef4444")
    st.markdown(f"""
    <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;
                padding:0.8rem 1.4rem;display:inline-flex;gap:1.5rem;align-items:center;margin-top:0.5rem;">
        <span style="color:var(--muted);font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;">Live BMI</span>
        <span style="font-family:'JetBrains Mono';font-size:1.6rem;color:{bmi_color};font-weight:600;">{bmi_live:.1f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Continue to Workout Data  ➜"):
        st.session_state.user_data = {
            "id": user_id, "name": name, "age": age,
            "height": height, "weight": weight,
            "goal_weight": goal_weight, "gender": gender
        }
        st.session_state.page = 2
        st.rerun()

# ══════════════════════════════════════════════
# PAGE 2 — WORKOUT DATA
# ══════════════════════════════════════════════
elif st.session_state.page == 2:

    user = st.session_state.user_data
    st.markdown(f"""
    <div style="color:var(--muted);margin-bottom:1rem;">
        <span style="color:var(--accent);font-family:'Bebas Neue';font-size:1.1rem;">
            {user.get('name','—').upper()}
        </span>
        &nbsp;·&nbsp; Age {user.get('age')} &nbsp;·&nbsp; {user.get('gender')}
        &nbsp;·&nbsp; {user.get('weight')} kg → {user.get('goal_weight')} kg
    </div>
    """, unsafe_allow_html=True)

    # ─── Config row ───
    col_a, col_b, col_c = st.columns([2,2,1])
    with col_a:
        exercise = st.selectbox("Exercise Type", ["Cardio","Strength","Mixed"])
    with col_b:
        weather  = st.selectbox("Weather Condition", ["Sunny","Cloudy","Rainy"])
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        day_idx  = st.number_input("Day #", 1, 7, 1) - 1

    # ─── Day input card ───
    st.markdown(f'<div class="card"><div class="card-title">DAY {day_idx+1} — {DAYS[day_idx].upper()}</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        cal  = st.number_input("Calories Burned", 0.0, 2000.0, float(st.session_state.weekly_data["calories"][day_idx]), key="cal")
    with d2:
        dur  = st.number_input("Duration (min)", 0.0, 300.0, float(st.session_state.weekly_data["duration"][day_idx]), key="dur")
    with d3:
        hr   = st.number_input("Heart Rate (bpm)", 0.0, 220.0, float(st.session_state.weekly_data["heart_rate"][day_idx]), key="hr")
    with d4:
        st.markdown("<br>", unsafe_allow_html=True)
        inten = st.slider("Intensity (1–10)", 1, 10, max(1, int(st.session_state.weekly_data["intensity"][day_idx])), key="int")

    if st.button(f"💾  Save Day {day_idx+1}"):
        st.session_state.weekly_data["calories"][day_idx]   = cal
        st.session_state.weekly_data["duration"][day_idx]   = dur
        st.session_state.weekly_data["heart_rate"][day_idx] = hr
        st.session_state.weekly_data["intensity"][day_idx]  = inten
        st.success(f"✅  {DAYS[day_idx]} data saved!")

    st.markdown('</div>', unsafe_allow_html=True)

    # ─── Live weekly charts ───
    data = st.session_state.weekly_data
    has_data = any(v > 0 for v in data["calories"])

    if has_data:
        st.markdown('<div class="card"><div class="card-title">WEEKLY OVERVIEW</div>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(weekly_bar("🔥 Calories Burned", data["calories"], "#f97316", "kcal"),
                            use_container_width=True, config={"displayModeBar":False})
        with ch2:
            st.plotly_chart(weekly_bar("⏱️ Duration", data["duration"], "#3b82f6", "min"),
                            use_container_width=True, config={"displayModeBar":False})
        ch3, ch4 = st.columns(2)
        with ch3:
            st.plotly_chart(weekly_bar("❤️ Heart Rate", data["heart_rate"], "#ef4444", "bpm"),
                            use_container_width=True, config={"displayModeBar":False})
        with ch4:
            st.plotly_chart(weekly_bar("💪 Intensity", data["intensity"], "#a855f7", "level"),
                            use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ─── Action buttons ───
    btn1, btn2, _, btn3 = st.columns([1.5, 1.5, 3, 2])
    with btn1:
        if st.button("⬅  Back"):
            st.session_state.page = 1
            st.rerun()
    with btn3:
        if st.button("🚀  Run AI Prediction"):
            st.session_state.exercise = exercise
            st.session_state.weather  = weather
            st.session_state.page = 3
            st.rerun()

# ══════════════════════════════════════════════
# PAGE 3 — RESULTS
# ══════════════════════════════════════════════
elif st.session_state.page == 3:

    user = st.session_state.user_data
    data = st.session_state.weekly_data
    exercise = st.session_state.get("exercise","Mixed")
    weather  = st.session_state.get("weather","Sunny")

    # ── Compute inputs ──
    avg_cal  = np.mean(data["calories"])
    avg_dur  = np.mean(data["duration"])
    avg_hr   = np.mean(data["heart_rate"])
    avg_int  = np.mean(data["intensity"])
    bmi      = user["weight"] / ((user["height"]/100) ** 2)
    wlr      = (user["weight"] - user["goal_weight"]) / 7
    protein  = user["weight"] * 1.6

    ex_map = {"Cardio":0,"Strength":1,"Mixed":2}
    gn_map = {"Male":0,"Female":1}
    wt_map = {"Sunny":0,"Cloudy":1,"Rainy":2}

    input_df = pd.DataFrame({
        'id':[user["id"]], 'exercise':[ex_map[exercise]],
        'calories_burn':[avg_cal], 'dream_weight':[user["goal_weight"]],
        'actual_weight':[user["weight"]], 'age':[user["age"]],
        'gender':[gn_map[user["gender"]]], 'duration':[avg_dur],
        'heart_rate':[avg_hr], 'bmi':[bmi],
        'weather_conditions':[wt_map[weather]],
        'exercise_intensity':[avg_int], 'weight_loss_rate':[wlr],
        'protein_intake':[protein]
    })

    if model_loaded:
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
    else:
        # Demo values when model not found
        prediction  = 1
        probability = [0.25, 0.55, 0.20]

    labels   = {0:"LOW", 1:"MODERATE", 2:"HIGH"}
    risk_cls = {0:"risk-low", 1:"risk-mod", 2:"risk-high"}
    risk_col = {0:"#22c55e", 1:"#eab308", 2:"#ef4444"}
    # Save prediction so page 4 can access it
    st.session_state["last_prediction"] = int(prediction)

    # ── Banner ──
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);
                border-left:4px solid {risk_col[prediction]};
                border-radius:14px;padding:1.6rem 2rem;margin-bottom:1.5rem;
                display:flex;align-items:center;justify-content:space-between;">
        <div>
            <div style="color:var(--muted);font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">
                Muscle Loss Risk Assessment · {user.get('name','User').upper()}
            </div>
            <div style="font-family:'Bebas Neue';font-size:2rem;color:var(--text);">
                AI ANALYSIS COMPLETE
            </div>
        </div>
        <div class="risk-badge {risk_cls[prediction]}">{labels[prediction]} RISK</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ──
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">BMI</div>
            <div class="metric-value">{bmi:.1f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Protein Target</div>
            <div class="metric-value">{protein:.0f}</div>
            <div class="metric-unit">g / day</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Daily Weight Loss</div>
            <div class="metric-value">{abs(wlr):.3f}</div>
            <div class="metric-unit">kg / day</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Calories/Day</div>
            <div class="metric-value">{avg_cal:.0f}</div>
            <div class="metric-unit">kcal</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Heart Rate</div>
            <div class="metric-value">{avg_hr:.0f}</div>
            <div class="metric-unit">bpm</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Intensity</div>
            <div class="metric-value">{avg_int:.1f}</div>
            <div class="metric-unit">/ 10</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row 1 ──
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        st.plotly_chart(gauge_chart(probability[0]*100, "Low Risk %", "#22c55e"),
                        use_container_width=True, config={"displayModeBar":False})
    with gc2:
        st.plotly_chart(gauge_chart(probability[1]*100, "Moderate Risk %", "#eab308"),
                        use_container_width=True, config={"displayModeBar":False})
    with gc3:
        st.plotly_chart(gauge_chart(probability[2]*100, "High Risk %", "#ef4444"),
                        use_container_width=True, config={"displayModeBar":False})

    # ── BMI Zone ──
    st.markdown('<div class="card"><div class="card-title">BMI ZONE</div>', unsafe_allow_html=True)
    st.plotly_chart(bmi_zone_chart(bmi), use_container_width=True, config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Radar + Weekly trend ──
    r1, r2 = st.columns([1, 1.6])
    with r1:
        st.markdown('<div class="card"><div class="card-title">FITNESS PROFILE</div>', unsafe_allow_html=True)
        radar_cats = ["Calories","Duration","Heart Rate","Intensity","Protein"]
        radar_vals = [avg_cal, avg_dur, avg_hr, avg_int*10, protein]
        radar_max  = [700, 120, 180, 100, 200]
        st.plotly_chart(radar_chart(radar_cats, radar_vals, radar_max),
                        use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="card"><div class="card-title">WEEKLY CALORIE TREND</div>', unsafe_allow_html=True)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=DAYS, y=data["calories"],
            mode="lines+markers",
            line=dict(color="#f97316", width=2.5),
            marker=dict(size=8, color="#f97316",
                        line=dict(color="#0a0c0f", width=2)),
            fill="tozeroy",
            fillcolor="rgba(249,115,22,0.08)",
            name="Calories"
        ))
        fig_trend.update_layout(
            xaxis=dict(tickfont=dict(color="#9ca3af"), gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(tickfont=dict(color="#6b7280"), gridcolor="#1f2430"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Risk probability bar ──
    st.markdown('<div class="card"><div class="card-title">RISK PROBABILITY BREAKDOWN</div>', unsafe_allow_html=True)
    fig_bar = go.Figure(go.Bar(
        x=["Low Risk","Moderate Risk","High Risk"],
        y=[p*100 for p in probability],
        marker=dict(color=["#22c55e","#eab308","#ef4444"],
                    opacity=0.85,
                    line=dict(color="rgba(0,0,0,0)")),
        text=[f"{p*100:.1f}%" for p in probability],
        textposition="outside",
        textfont=dict(color="#e8eaf0", size=14, family="JetBrains Mono"),
        width=0.5
    ))
    fig_bar.update_layout(
        xaxis=dict(tickfont=dict(color="#9ca3af", size=13), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(range=[0,110], tickfont=dict(color="#6b7280"), gridcolor="#1f2430",
                   title="Probability (%)", title_font=dict(color="#6b7280")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(t=20, b=10, l=10, r=10),
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Recommendations ──
    st.markdown('<div class="card"><div class="card-title">RECOMMENDATIONS</div>', unsafe_allow_html=True)
    rec1, rec2, rec3 = st.columns(3)

    with rec1:
        if wlr > 0.2:
            st.warning("⚡ **Weight Loss Too Aggressive**\n\nSlowing deficit reduces muscle catabolism risk.")
        elif wlr < 0.07:
            st.info("🐢 **Conservative Deficit**\n\nSafe pace — increase activity if desired.")
        else:
            st.success("✅ **Healthy Fat Loss Range**\n\nMaintain current deficit and monitor weekly.")

    with rec2:
        if avg_int < 4:
            st.info("📈 **Increase Workout Intensity**\n\nHigher intensity stimulates muscle retention.")
        elif avg_int >= 8:
            st.warning("🔄 **Consider Recovery Days**\n\nHigh intensity every day risks overtraining.")
        else:
            st.success("💪 **Good Intensity Balance**\n\nStay consistent across the week.")

    with rec3:
        bmi_note = ""
        if bmi < 18.5:
            st.warning(f"⚠️ **Underweight (BMI {bmi:.1f})**\n\nPrioritize calorie surplus before cutting.")
        elif bmi <= 24.9:
            st.success(f"✅ **Healthy BMI ({bmi:.1f})**\n\nFocus on body composition, not just weight.")
        elif bmi <= 29.9:
            st.info(f"📊 **Overweight (BMI {bmi:.1f})**\n\nStrength training + moderate deficit recommended.")
        else:
            st.warning(f"🔴 **Obese Range (BMI {bmi:.1f})**\n\nConsult a physician before intense training.")

    st.markdown('</div>', unsafe_allow_html=True)

        # ── Navigation buttons ──
    nb1, nb2 = st.columns([1, 1])
    with nb1:
        if st.button("⬅  Start Over"):
            st.session_state.page = 1
            st.session_state.user_data = {}
            st.session_state.weekly_data = {
                "calories":[0]*7,"duration":[0]*7,
                "heart_rate":[0]*7,"intensity":[0]*7
            }
            st.rerun()
    with nb2:
        if st.button("🏋️  View Workout Plan  ➜"):
            st.session_state.page = 4
            st.rerun()

# ══════════════════════════════════════════════
# PAGE 4 — WORKOUT PLAN
# ══════════════════════════════════════════════
elif st.session_state.page == 4:

    prediction = st.session_state.get("last_prediction", 1)
    user       = st.session_state.user_data
    exercise   = st.session_state.get("exercise", "Mixed")

    risk_labels = {0:"LOW", 1:"MODERATE", 2:"HIGH"}
    risk_colors = {0:"#22c55e", 1:"#eab308", 2:"#ef4444"}
    risk_cls    = {0:"risk-low", 1:"risk-mod", 2:"risk-high"}

    # ── Workout plans keyed by (risk, exercise_type) ──
    PLANS = {
        # ─── LOW RISK ───────────────────────────────────────────────────────
        (0, "Cardio"): [
            {"day":"Monday",    "focus":"Cardio Endurance",  "exercises":[
                ("Treadmill Run","3 sets × 20 min"), ("Jump Rope","3 sets × 5 min"),
                ("High Knees","3 sets × 30 reps")]},
            {"day":"Tuesday",   "focus":"Active Recovery",   "exercises":[
                ("Brisk Walk","1 set × 30 min"), ("Stretching","1 set × 15 min")]},
            {"day":"Wednesday", "focus":"HIIT Cardio",       "exercises":[
                ("Burpees","4 sets × 15 reps"), ("Mountain Climbers","4 sets × 20 reps"),
                ("Box Jumps","3 sets × 12 reps")]},
            {"day":"Thursday",  "focus":"Rest Day",          "exercises":[
                ("Light Yoga","1 set × 20 min")]},
            {"day":"Friday",    "focus":"Cardio + Core",     "exercises":[
                ("Cycling","3 sets × 20 min"), ("Plank","3 sets × 60 sec"),
                ("Leg Raises","3 sets × 15 reps")]},
            {"day":"Saturday",  "focus":"Long Cardio",       "exercises":[
                ("Outdoor Run","1 set × 45 min"), ("Cool-down Walk","1 set × 10 min")]},
            {"day":"Sunday",    "focus":"Rest Day",          "exercises":[
                ("Foam Rolling","1 set × 15 min")]},
        ],
        (0, "Strength"): [
            {"day":"Monday",    "focus":"Upper Body Push",   "exercises":[
                ("Push-ups","3 sets × 15 reps"), ("Dumbbell Press","3 sets × 12 reps"),
                ("Lateral Raises","3 sets × 12 reps")]},
            {"day":"Tuesday",   "focus":"Lower Body",        "exercises":[
                ("Bodyweight Squats","4 sets × 15 reps"), ("Lunges","3 sets × 12 reps"),
                ("Calf Raises","3 sets × 20 reps")]},
            {"day":"Wednesday", "focus":"Active Recovery",   "exercises":[
                ("Stretching","1 set × 20 min"), ("Brisk Walk","1 set × 20 min")]},
            {"day":"Thursday",  "focus":"Upper Body Pull",   "exercises":[
                ("Dumbbell Rows","3 sets × 12 reps"), ("Bicep Curls","3 sets × 12 reps"),
                ("Face Pulls","3 sets × 15 reps")]},
            {"day":"Friday",    "focus":"Full Body",         "exercises":[
                ("Deadlifts","4 sets × 10 reps"), ("Goblet Squats","3 sets × 12 reps"),
                ("Plank","3 sets × 45 sec")]},
            {"day":"Saturday",  "focus":"Core & Mobility",   "exercises":[
                ("Crunches","3 sets × 20 reps"), ("Russian Twists","3 sets × 16 reps"),
                ("Hip Flexor Stretch","2 sets × 30 sec")]},
            {"day":"Sunday",    "focus":"Rest Day",          "exercises":[
                ("Light Walk","1 set × 20 min")]},
        ],
        (0, "Mixed"): [
            {"day":"Monday",    "focus":"Strength — Upper",  "exercises":[
                ("Push-ups","3 sets × 15 reps"), ("Dumbbell Shoulder Press","3 sets × 12 reps"),
                ("Tricep Dips","3 sets × 12 reps")]},
            {"day":"Tuesday",   "focus":"Cardio",            "exercises":[
                ("Treadmill Run","3 sets × 15 min"), ("Jump Rope","3 sets × 3 min")]},
            {"day":"Wednesday", "focus":"Strength — Lower",  "exercises":[
                ("Squats","4 sets × 15 reps"), ("Lunges","3 sets × 12 reps"),
                ("Glute Bridges","3 sets × 15 reps")]},
            {"day":"Thursday",  "focus":"Rest Day",          "exercises":[
                ("Yoga / Stretching","1 set × 25 min")]},
            {"day":"Friday",    "focus":"HIIT",              "exercises":[
                ("Burpees","4 sets × 12 reps"), ("Box Jumps","3 sets × 10 reps"),
                ("Mountain Climbers","4 sets × 20 reps")]},
            {"day":"Saturday",  "focus":"Full Body Strength","exercises":[
                ("Deadlifts","3 sets × 10 reps"), ("Bent-over Rows","3 sets × 12 reps"),
                ("Plank","3 sets × 60 sec")]},
            {"day":"Sunday",    "focus":"Rest Day",          "exercises":[
                ("Foam Rolling","1 set × 15 min")]},
        ],

        # ─── MODERATE RISK ───────────────────────────────────────────────────
        (1, "Cardio"): [
            {"day":"Monday",    "focus":"Moderate Cardio",   "exercises":[
                ("Treadmill Jog","3 sets × 15 min"), ("Cycling","2 sets × 10 min"),
                ("Step-ups","3 sets × 15 reps")]},
            {"day":"Tuesday",   "focus":"Strength Insert",   "exercises":[
                ("Bodyweight Squats","3 sets × 15 reps"), ("Push-ups","3 sets × 12 reps"),
                ("Plank","3 sets × 40 sec")]},
            {"day":"Wednesday", "focus":"Active Recovery",   "exercises":[
                ("Walking","1 set × 30 min"), ("Stretching","1 set × 15 min")]},
            {"day":"Thursday",  "focus":"Interval Cardio",   "exercises":[
                ("Sprint Intervals","6 sets × 1 min on / 1 min off"),
                ("Jump Rope","3 sets × 3 min")]},
            {"day":"Friday",    "focus":"Strength + Cardio", "exercises":[
                ("Lunges","3 sets × 12 reps"), ("Dumbbell Rows","3 sets × 12 reps"),
                ("Rowing Machine","2 sets × 10 min")]},
            {"day":"Saturday",  "focus":"Light Cardio",      "exercises":[
                ("Brisk Walk","1 set × 40 min"), ("Core Stretches","1 set × 10 min")]},
            {"day":"Sunday",    "focus":"Rest Day",          "exercises":[
                ("Foam Rolling","1 set × 15 min")]},
        ],
        (1, "Strength"): [
            {"day":"Monday",    "focus":"Push Day",          "exercises":[
                ("Bench Press","4 sets × 10 reps"), ("Overhead Press","3 sets × 10 reps"),
                ("Tricep Pushdowns","3 sets × 12 reps")]},
            {"day":"Tuesday",   "focus":"Pull Day",          "exercises":[
                ("Barbell Rows","4 sets × 10 reps"), ("Lat Pulldowns","3 sets × 12 reps"),
                ("Hammer Curls","3 sets × 12 reps")]},
            {"day":"Wednesday", "focus":"Active Recovery",   "exercises":[
                ("Light Walk","1 set × 20 min"), ("Mobility Work","1 set × 20 min")]},
            {"day":"Thursday",  "focus":"Leg Day",           "exercises":[
                ("Barbell Squats","4 sets × 10 reps"), ("Romanian Deadlift","3 sets × 10 reps"),
                ("Leg Press","3 sets × 12 reps")]},
            {"day":"Friday",    "focus":"Full Body",         "exercises":[
                ("Deadlifts","4 sets × 8 reps"), ("Pull-ups","3 sets × 8 reps"),
                ("Dumbbell Lunges","3 sets × 12 reps")]},
            {"day":"Saturday",  "focus":"Core & Arms",       "exercises":[
                ("Plank","4 sets × 60 sec"), ("Bicep Curls","3 sets × 15 reps"),
                ("Tricep Dips","3 sets × 15 reps")]},
            {"day":"Sunday",    "focus":"Rest Day",          "exercises":[
                ("Stretching","1 set × 20 min")]},
        ],
        (1, "Mixed"): [
            {"day":"Monday",    "focus":"Upper Strength",    "exercises":[
                ("Bench Press","4 sets × 10 reps"), ("Dumbbell Rows","3 sets × 12 reps"),
                ("Lateral Raises","3 sets × 12 reps")]},
            {"day":"Tuesday",   "focus":"Cardio",            "exercises":[
                ("Cycling","3 sets × 15 min"), ("Jump Rope","3 sets × 3 min")]},
            {"day":"Wednesday", "focus":"Leg Strength",      "exercises":[
                ("Barbell Squats","4 sets × 10 reps"), ("Leg Curls","3 sets × 12 reps"),
                ("Calf Raises","3 sets × 20 reps")]},
            {"day":"Thursday",  "focus":"Active Recovery",   "exercises":[
                ("Yoga","1 set × 30 min")]},
            {"day":"Friday",    "focus":"HIIT + Core",       "exercises":[
                ("Burpees","4 sets × 12 reps"), ("Plank","3 sets × 60 sec"),
                ("Leg Raises","3 sets × 15 reps")]},
            {"day":"Saturday",  "focus":"Full Body Strength","exercises":[
                ("Deadlifts","4 sets × 8 reps"), ("Overhead Press","3 sets × 10 reps"),
                ("Pull-ups","3 sets × 8 reps")]},
            {"day":"Sunday",    "focus":"Rest Day",          "exercises":[
                ("Foam Rolling","1 set × 15 min")]},
        ],

        # ─── HIGH RISK ────────────────────────────────────────────────────────
        (2, "Cardio"): [
            {"day":"Monday",    "focus":"Low-Impact Cardio", "exercises":[
                ("Walking","2 sets × 20 min"), ("Light Cycling","2 sets × 10 min")]},
            {"day":"Tuesday",   "focus":"Muscle Retention",  "exercises":[
                ("Bodyweight Squats","3 sets × 15 reps"), ("Push-ups","3 sets × 10 reps"),
                ("Plank","3 sets × 30 sec")]},
            {"day":"Wednesday", "focus":"Rest Day",          "exercises":[
                ("Stretching","1 set × 20 min")]},
            {"day":"Thursday",  "focus":"Gentle Cardio",     "exercises":[
                ("Elliptical","2 sets × 15 min"), ("Step-ups","2 sets × 12 reps")]},
            {"day":"Friday",    "focus":"Strength Focus",    "exercises":[
                ("Dumbbell Squats","3 sets × 12 reps"), ("Dumbbell Rows","3 sets × 10 reps"),
                ("Glute Bridges","3 sets × 15 reps")]},
            {"day":"Saturday",  "focus":"Light Activity",    "exercises":[
                ("Brisk Walk","1 set × 30 min"), ("Foam Rolling","1 set × 10 min")]},
            {"day":"Sunday",    "focus":"Rest Day",          "exercises":[
                ("Light Yoga","1 set × 20 min")]},
        ],
        (2, "Strength"): [
            {"day":"Monday",    "focus":"Upper Body — Light","exercises":[
                ("Push-ups","3 sets × 10 reps"), ("Dumbbell Curls","3 sets × 10 reps"),
                ("Overhead Press","3 sets × 8 reps")]},
            {"day":"Tuesday",   "focus":"Rest Day",          "exercises":[
                ("Stretching","1 set × 20 min")]},
            {"day":"Wednesday", "focus":"Lower Body — Light","exercises":[
                ("Goblet Squats","3 sets × 10 reps"), ("Glute Bridges","3 sets × 15 reps"),
                ("Calf Raises","3 sets × 15 reps")]},
            {"day":"Thursday",  "focus":"Active Recovery",   "exercises":[
                ("Walking","1 set × 25 min"), ("Mobility Work","1 set × 15 min")]},
            {"day":"Friday",    "focus":"Full Body — Light",  "exercises":[
                ("Deadlifts","3 sets × 8 reps"), ("Dumbbell Rows","3 sets × 10 reps"),
                ("Plank","3 sets × 40 sec")]},
            {"day":"Saturday",  "focus":"Core Stability",    "exercises":[
                ("Dead Bug","3 sets × 10 reps"), ("Bird Dog","3 sets × 10 reps"),
                ("Side Plank","3 sets × 30 sec each")]},
            {"day":"Sunday",    "focus":"Rest Day",          "exercises":[
                ("Foam Rolling","1 set × 15 min")]},
        ],
        (2, "Mixed"): [
            {"day":"Monday",    "focus":"Gentle Strength",   "exercises":[
                ("Push-ups","3 sets × 10 reps"), ("Dumbbell Rows","3 sets × 10 reps"),
                ("Plank","3 sets × 30 sec")]},
            {"day":"Tuesday",   "focus":"Low-Impact Cardio", "exercises":[
                ("Walking","1 set × 30 min"), ("Light Cycling","1 set × 15 min")]},
            {"day":"Wednesday", "focus":"Rest Day",          "exercises":[
                ("Stretching / Yoga","1 set × 25 min")]},
            {"day":"Thursday",  "focus":"Lower Body — Light","exercises":[
                ("Bodyweight Squats","3 sets × 12 reps"), ("Glute Bridges","3 sets × 15 reps"),
                ("Calf Raises","3 sets × 15 reps")]},
            {"day":"Friday",    "focus":"Cardio + Core",     "exercises":[
                ("Elliptical","2 sets × 15 min"), ("Crunches","3 sets × 15 reps"),
                ("Leg Raises","3 sets × 12 reps")]},
            {"day":"Saturday",  "focus":"Full Body — Light",  "exercises":[
                ("Deadlifts","3 sets × 8 reps"), ("Overhead Press","3 sets × 8 reps"),
                ("Lat Pulldowns","3 sets × 10 reps")]},
            {"day":"Sunday",    "focus":"Rest Day",          "exercises":[
                ("Foam Rolling","1 set × 15 min")]},
        ],
    }

    plan = PLANS.get((prediction, exercise), PLANS[(1, "Mixed")])

    focus_colors = {
        "Rest Day":"#374151", "Active Recovery":"#1e3a5f",
        "Cardio":"#1e3a5f", "HIIT":"#7c2d12", "HIIT Cardio":"#7c2d12",
        "Strength":"#1a3a2a", "Push Day":"#1a3a2a", "Pull Day":"#1a3a2a",
        "Leg Day":"#1a3a2a", "Full Body":"#2d1a3a",
    }
    def focus_color(f):
        for k, v in focus_colors.items():
            if k.lower() in f.lower(): return v
        return "#1a1d24"

    # ── Banner ──
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);
                border-left:4px solid {risk_colors[prediction]};
                border-radius:14px;padding:1.6rem 2rem;margin-bottom:1.5rem;
                display:flex;align-items:center;justify-content:space-between;">
        <div>
            <div style="color:var(--muted);font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">
                7-Day Workout Schedule · {user.get('name','User').upper()} · {exercise}
            </div>
            <div style="font-family:'Bebas Neue';font-size:2rem;color:var(--text);">
                MUSCLE PRESERVATION PLAN
            </div>
        </div>
        <div class="risk-badge {risk_cls[prediction]}">{risk_labels[prediction]} RISK</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Info strip ──
    tip = {
        0: "✅ You are at low risk. Maintain consistency and keep protein intake high.",
        1: "⚠️ Moderate risk detected. Add more strength training and avoid skipping meals.",
        2: "🔴 High risk! Prioritize recovery, reduce cardio volume, and maximize protein.",
    }
    st.info(tip[prediction])

    # ── 7-day cards grid ──
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    cols = [col1, col2]

    for i, day_plan in enumerate(plan):
        with cols[i % 2]:
            ex_rows = "".join([
                f"""<tr>
                    <td style="padding:0.35rem 0.5rem;color:#e8eaf0;font-size:0.88rem;">{ex}</td>
                    <td style="padding:0.35rem 0.5rem;color:#f97316;font-size:0.82rem;
                               font-family:'JetBrains Mono';text-align:right;">{scheme}</td>
                </tr>"""
                for ex, scheme in day_plan["exercises"]
            ])
            bg = focus_color(day_plan["focus"])
            is_rest = "rest" in day_plan["focus"].lower() or "recovery" in day_plan["focus"].lower()
            border_col = "#374151" if is_rest else risk_colors[prediction]

            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border_col};
                        border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:1rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.7rem;">
                    <span style="font-family:'Bebas Neue';font-size:1.3rem;
                                 color:{'#6b7280' if is_rest else '#e8eaf0'};letter-spacing:1px;">
                        {day_plan['day'].upper()}
                    </span>
                    <span style="background:{'rgba(55,65,81,0.6)' if is_rest else 'rgba(249,115,22,0.15)'};
                                 color:{'#6b7280' if is_rest else risk_colors[prediction]};
                                 font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;
                                 padding:0.2rem 0.7rem;border-radius:99px;">
                        {day_plan['focus']}
                    </span>
                </div>
                <table style="width:100%;border-collapse:collapse;">
                    {ex_rows}
                </table>
            </div>
            """, unsafe_allow_html=True)

    # ── Weekly volume chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">📊 WEEKLY EXERCISE VOLUME</div>', unsafe_allow_html=True)

    total_sets = []
    for day_plan in plan:
        sets = 0
        for _, scheme in day_plan["exercises"]:
            import re
            m = re.search(r"(\d+)\s*set", scheme, re.IGNORECASE)
            sets += int(m.group(1)) if m else 0
        total_sets.append(sets)

    day_names  = [d["day"][:3] for d in plan]
    focus_list = [d["focus"] for d in plan]
    bar_colors = [risk_colors[prediction] if "rest" not in f.lower() and "recovery" not in f.lower()
                  else "#374151" for f in focus_list]

    fig_vol = go.Figure(go.Bar(
        x=day_names, y=total_sets,
        marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
        text=total_sets,
        textposition="outside",
        textfont=dict(color="#9ca3af", size=11, family="JetBrains Mono"),
    ))
    fig_vol.update_layout(
        xaxis=dict(tickfont=dict(color="#9ca3af"), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Total Sets", tickfont=dict(color="#6b7280"),
                   gridcolor="#1f2430", title_font=dict(color="#6b7280")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(t=20,b=10,l=10,r=10), showlegend=False
    )
    st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Back buttons ──
    pb1, pb2 = st.columns([1,1])
    with pb1:
        if st.button("⬅  Back to Results"):
            st.session_state.page = 3
            st.rerun()
    with pb2:
        if st.button("⬅  Start Over"):
            st.session_state.page = 1
            st.session_state.user_data = {}
            st.session_state.weekly_data = {
                "calories":[0]*7,"duration":[0]*7,
                "heart_rate":[0]*7,"intensity":[0]*7
            }
            st.rerun()

if not model_loaded:
    st.caption("ℹ️  muscle_loss_model.pkl not found — running in demo mode with placeholder predictions.")