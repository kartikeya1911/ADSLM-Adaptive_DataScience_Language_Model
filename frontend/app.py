"""
frontend/app.py
================
ADSLM — Premium Streamlit Frontend
Run: streamlit run frontend/app.py
"""

import io
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADSLM | ABB AI Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0a0a0f; }

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
    color: #e2e8f0;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #cc0000 0%, #8b0000 40%, #1a0000 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(204,0,0,0.3);
    box-shadow: 0 0 60px rgba(204,0,0,0.15);
}
.hero-title {
    font-size: 2.4rem; font-weight: 900;
    color: #ffffff; margin: 0; line-height: 1.1;
}
.hero-subtitle {
    font-size: 1rem; color: rgba(255,255,255,0.75);
    margin-top: 0.5rem; font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.75rem; color: #fff;
    margin-top: 0.8rem; margin-right: 6px;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #161b22, #1a2030);
    border: 1px solid rgba(204,0,0,0.25);
    border-radius: 12px; padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: rgba(204,0,0,0.6); }
.metric-value { font-size: 1.8rem; font-weight: 800; color: #ff4444; }
.metric-label { font-size: 0.78rem; color: #8892a4; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

/* Section Headers */
.section-header {
    font-size: 1.1rem; font-weight: 700;
    color: #ff4444;
    border-left: 4px solid #cc0000;
    padding-left: 12px; margin: 1.5rem 0 1rem;
    text-transform: uppercase; letter-spacing: 0.08em;
}

/* Insight Cards */
.insight-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px; padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #cc0000;
}
.insight-label { font-size: 0.7rem; color: #cc0000; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; }
.insight-text { font-size: 0.9rem; color: #c9d1d9; margin-top: 4px; line-height: 1.6; }

/* Recommendation Cards */
.rec-card {
    background: linear-gradient(135deg, #161b22, #1c2130);
    border: 1px solid #30363d;
    border-radius: 10px; padding: 1rem;
    margin-bottom: 0.6rem;
    display: flex; gap: 12px; align-items: flex-start;
}
.rec-rank {
    background: #cc0000; color: white;
    border-radius: 50%; width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; flex-shrink: 0;
}
.rec-name { font-weight: 700; color: #e2e8f0; font-size: 0.95rem; }
.rec-reason { font-size: 0.82rem; color: #8892a4; margin-top: 3px; line-height: 1.5; }

/* Action items */
.action-item {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 0.7rem 1rem;
    margin-bottom: 0.5rem; font-size: 0.88rem;
    color: #c9d1d9; display: flex; gap: 10px;
}
.action-num { color: #cc0000; font-weight: 700; flex-shrink: 0; }

/* Task badge */
.task-badge {
    display: inline-block;
    padding: 6px 20px; border-radius: 20px;
    font-weight: 700; font-size: 1rem;
    letter-spacing: 0.05em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #cc0000, #8b0000) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(204,0,0,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(204,0,0,0.5) !important;
}

/* DataFrame */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Divider */
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = "http://127.0.0.1:8000"

TASK_COLORS = {
    "Regression":     ("#ff6b6b", "#2d1515"),
    "Classification": ("#4ecdc4", "#0f2422"),
    "Clustering":     ("#ffe66d", "#2a2410"),
    "Time-Series":    ("#a29bfe", "#1a1530"),
}

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🤖 ADSLM</div>
  <div class="hero-subtitle">Adaptive Data Science Language Model — ABB Industrial AI Copilot</div>
  <span class="hero-badge">AutoML</span>
  <span class="hero-badge">Explainable AI</span>
  <span class="hero-badge">Adaptive Intelligence</span>
  <span class="hero-badge">Industrial Grade</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

    target_column   = None
    expertise_level = "intermediate"

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)

        st.markdown("### 🎯 Target Column")
        cols = [""] + list(df_preview.columns)
        sel  = st.selectbox("Select prediction target (leave blank for Clustering)", cols)
        if sel != "":
            target_column = sel

        st.markdown("### 👤 Expertise Level")
        expertise_level = st.select_slider(
            "Adapt explanations for:",
            options=["beginner", "intermediate", "expert"],
            value="intermediate",
        )

        st.markdown("---")
        run_btn = st.button("🚀 Run ADSLM Pipeline", use_container_width=True)
    else:
        run_btn = False
        st.info("👈 Upload a CSV dataset to begin.")
        st.markdown("---")
        st.markdown("### 💡 Supported Tasks")
        for t, emoji in [("Regression","📈"),("Classification","🏷️"),("Clustering","🔵"),("Time-Series","📅")]:
            st.markdown(f"- {emoji} **{t}**")

# ── Main Content ───────────────────────────────────────────────────────────────
if not uploaded_file:
    # Landing state
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("14", "Total Modules"),
        ("4",  "ML Task Types"),
        ("3",  "Expertise Levels"),
        ("∞",  "Dataset Support"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### How ADSLM Works
    1. **Upload** any CSV dataset (sensor data, maintenance logs, quality records)
    2. **Analyze** — ADSLM automatically profiles your data
    3. **Detect** — Identifies the right ML task (Regression / Classification / Clustering / Time-Series)
    4. **Train** — Trains & compares multiple models automatically
    5. **Explain** — Generates AI insights adapted to your expertise level
    6. **Report** — Saves a full PDF/TXT report for submission
    """)

elif uploaded_file and not run_btn:
    # Preview state
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_preview.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df_preview.shape[0]:,}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df_preview.shape[1]}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
    with c3:
        miss_pct = round(df_preview.isnull().sum().sum() / df_preview.size * 100, 1)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{miss_pct}%</div><div class="metric-label">Missing Values</div></div>', unsafe_allow_html=True)

elif run_btn:
    # Pipeline Execution
    with st.spinner("⚙️ Running full ADSLM pipeline — analyzing, training, explaining…"):
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        data  = {"expertise_level": expertise_level}
        if target_column:
            data["target_column"] = target_column

        try:
            res = requests.post(f"{BACKEND_URL}/orchestrate", files=files, data=data, timeout=300)
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the FastAPI backend. Run: `uvicorn main:app --reload`")
            st.stop()

    if res.status_code != 200:
        st.error(f"Pipeline failed (HTTP {res.status_code}): {res.json().get('detail','Unknown error')}")
        st.stop()

    R = res.json()
    meta     = R.get("metadata", {})
    insights = R.get("insights", {})
    metrics  = R.get("metrics", {})
    recs     = R.get("recommendations", [])
    fi       = R.get("feature_importances", {})
    top_feat = R.get("top_features", [])
    actions  = R.get("actionable_recommendations", [])
    analysis = R.get("analysis", {})
    all_res  = R.get("all_model_results", {})
    task     = meta.get("task_type","")
    tc, bg   = TASK_COLORS.get(task, ("#ff4444","#2d1515"))

    st.success("✅ Pipeline complete!")
    st.balloons()

    # ── Key Metrics Bar ──────────────────────────────────────────────────────
    cols = st.columns(4)
    kv_pairs = [
        ("Task Detected",  task),
        ("Best Model",     meta.get("best_model","N/A")),
        ("Best Score",     f"{meta.get('best_score',0):.4f}" if meta.get("best_score") else "N/A"),
        ("Expertise",      expertise_level.capitalize()),
    ]
    for col, (label, val) in zip(cols, kv_pairs):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.1rem">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column Layout ────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        # Insights
        st.markdown('<div class="section-header">🧠 AI Insights</div>', unsafe_allow_html=True)
        for key, label in [("data_profile","📊 Data Profile"),("task","🎯 Task Detection"),
                            ("preprocessing","🔧 Preprocessing"),("model","🏆 Model Selection"),
                            ("features","🔍 Feature Impact")]:
            text = insights.get(key,"")
            if text:
                st.markdown(f"""
                <div class="insight-card">
                  <div class="insight-label">{label}</div>
                  <div class="insight-text">{text}</div>
                </div>""", unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="section-header">💡 Model Recommendations</div>', unsafe_allow_html=True)
        for rec in recs:
            st.markdown(f"""
            <div class="rec-card">
              <div class="rec-rank">{rec.get('priority','')}</div>
              <div>
                <div class="rec-name">{rec['model']}</div>
                <div class="rec-reason">{rec['reason']}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Actionable Recommendations
        if actions:
            st.markdown('<div class="section-header">✅ Actionable Next Steps</div>', unsafe_allow_html=True)
            for i, act in enumerate(actions, 1):
                st.markdown(f'<div class="action-item"><span class="action-num">#{i}</span>{act}</div>', unsafe_allow_html=True)

    with right:
        # Performance Metrics
        st.markdown('<div class="section-header">📊 Performance Metrics</div>', unsafe_allow_html=True)

        display_metrics = {k:v for k,v in metrics.items() if k != "Confusion Matrix"}
        if display_metrics:
            df_met = pd.DataFrame(list(display_metrics.items()), columns=["Metric","Value"])
            df_met["Value"] = df_met["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
            st.dataframe(df_met, use_container_width=True, hide_index=True)

        if "Confusion Matrix" in metrics:
            st.markdown("**Confusion Matrix**")
            cm = metrics["Confusion Matrix"]
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale=[[0,"#0d1117"],[1,"#cc0000"]],
                labels=dict(x="Predicted", y="Actual"),
                aspect="auto",
            )
            fig_cm.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_color="#e2e8f0", margin=dict(l=20,r=20,t=30,b=20), height=280,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # All Models Comparison
        if len(all_res) > 1:
            st.markdown('<div class="section-header">⚡ Model Comparison</div>', unsafe_allow_html=True)
            comp_data = []
            for mname, mmetrics in all_res.items():
                if isinstance(mmetrics, dict) and "error" not in mmetrics:
                    score_key = "F1-score" if task=="Classification" else ("RMSE" if task in ("Regression","Time-Series") else "Silhouette Score")
                    score = mmetrics.get(score_key, 0)
                    comp_data.append({"Model": mname, "Score": score, "Metric": score_key})
            if comp_data:
                df_comp = pd.DataFrame(comp_data).sort_values("Score", ascending=(task!="Classification"))
                fig_bar = go.Figure(go.Bar(
                    x=df_comp["Score"], y=df_comp["Model"],
                    orientation="h",
                    marker=dict(color=df_comp["Score"], colorscale=[[0,"#8b0000"],[1,"#ff4444"]]),
                    text=[f"{s:.4f}" for s in df_comp["Score"]], textposition="auto",
                ))
                fig_bar.update_layout(
                    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    font_color="#e2e8f0", xaxis_title=comp_data[0]["Metric"],
                    margin=dict(l=10,r=10,t=10,b=10), height=220,
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Feature Importance
        if top_feat:
            st.markdown('<div class="section-header">🔍 Feature Importances</div>', unsafe_allow_html=True)
            df_fi = pd.DataFrame(top_feat)
            fig_fi = go.Figure(go.Bar(
                x=df_fi["importance"], y=df_fi["feature"],
                orientation="h",
                marker=dict(color=df_fi["importance"], colorscale=[[0,"#1a0000"],[0.5,"#8b0000"],[1,"#ff4444"]]),
                text=[f"{v:.4f}" for v in df_fi["importance"]], textposition="auto",
            ))
            fig_fi.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                font_color="#e2e8f0", xaxis_title="Importance Score",
                margin=dict(l=10,r=10,t=10,b=10), height=max(250, len(df_fi)*35),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    # ── Full Dataset Analysis Expander ──────────────────────────────────────
    with st.expander("🔬 Full Dataset Analysis", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Basic Info**")
            bi = analysis.get("basic_info", {})
            st.json({
                "Rows": bi.get("row_count"),
                "Columns": bi.get("column_count"),
                "Duplicates": bi.get("duplicate_rows"),
                "Memory (MB)": bi.get("memory_mb"),
            })
            st.markdown("**Column Types**")
            ct = analysis.get("column_types", {})
            st.json({k: len(v) for k, v in ct.items()})
        with c2:
            st.markdown("**Missing Values**")
            mv = {k:v for k,v in analysis.get("missing_values",{}).items() if k != "_summary"}
            if mv:
                df_mv = pd.DataFrame([{"Column":k,"Count":v["count"],"Percent (%)":v["percent"]} for k,v in mv.items()])
                st.dataframe(df_mv, use_container_width=True, hide_index=True)
            else:
                st.success("No missing values found!")
            st.markdown("**Outlier Info**")
            oi = {k:v for k,v in analysis.get("outlier_info",{}).items() if k != "_summary"}
            if oi:
                df_oi = pd.DataFrame([{"Column":k,"Outliers":v["outlier_count"]} for k,v in oi.items()])
                st.dataframe(df_oi, use_container_width=True, hide_index=True)
            else:
                st.success("No significant outliers detected.")

    # ── Report Download ──────────────────────────────────────────────────────
    if R.get("report_path"):
        st.markdown("---")
        st.markdown('<div class="section-header">📄 Report</div>', unsafe_allow_html=True)
        path = R["report_path"]
        txt_path = path.replace(".pdf", ".txt") if path.endswith(".pdf") else path
        try:
            txt_path_check = txt_path if os.path.exists(txt_path) else path
            if os.path.exists(txt_path_check):
                with open(txt_path_check, "r", encoding="utf-8") as f:
                    report_text = f.read()
                st.download_button(
                    "⬇️ Download Full Report (TXT)",
                    data=report_text.encode("utf-8"),
                    file_name="adslm_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        except Exception:
            pass

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#3d4a5c;font-size:0.78rem;padding:1rem">'
    '🤖 ADSLM v1.0 | Adaptive Data Science Language Model | ABB Innovation Evaluation 2026'
    '</div>',
    unsafe_allow_html=True,
)
