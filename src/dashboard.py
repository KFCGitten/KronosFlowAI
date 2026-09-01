# ====== stream‑lit dashboard – mirrors SVAMP app.py design ======

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import List, Dict, Optional, Tuple

# Add project root to sys.path to resolve src imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import config constants
from config import DB_PATH, REPORT_PATH, OUTPUT_DIR, EDA_DIR
from src.utils import load_report, fetch_all_tickers, get_db_connection, log
from src.features import get_latest_data_date

# ==========================================================
# PAGE CONFIGURATION & STYLING
# ==========================================================
st.set_page_config(
    page_title="KronosFlow AI Observatory",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1a1b2f, #0d0e15);
        color: #f1f3f9;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #8f9cae;
        margin-bottom: 12px;
    }
    
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    .main-title {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
        font-size: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# DATA LOADING & CACHING
# ==========================================================
@st.cache_data(show_spinner=False)
def load_data() -> Tuple[Optional[Dict], List[str]]:
    """Load evaluation report and ticker list using utils helpers with caching."""
    report = load_report()
    tickers = fetch_all_tickers()
    return report, tickers

@st.cache_data(show_spinner=False)
def get_ticker_prices(ticker: str) -> pd.DataFrame:
    """Load historical prices for a specific ticker using utils DB helper with caching."""
    conn = get_db_connection()
    query = """
    SELECT record_date, open, high, low, close, volume 
    FROM historical_prices 
    WHERE ticker = ? 
    ORDER BY record_date ASC
    """
    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()
    return df


# ==========================================================
# MAIN APPLICATION INTERFACE
# ==========================================================
def main():
    st.markdown('<h1 class="main-title">📈 KronosFlow AI Observatory</h1>', unsafe_allow_html=True)
    st.markdown("### Modellövervakning, Beslutslogg & Finansiell Backtesting")
    
    # Load data with spinner to indicate progress
    with st.spinner("Laddar data…"):
        report, tickers = load_data()

    # Offer to download raw evaluation report
    if st.sidebar.button("📥 Ladda ner rapport (JSON)"):
        json_str = json.dumps(report, indent=2)
        st.sidebar.download_button(
            label="Download JSON",
            data=json_str,
            file_name="evaluation_report.json",
            mime="application/json"
        )
    
    if not report or not tickers:
        st.warning("⚠️ Hittade ingen utvärderingsrapport eller databas. Kontrollera att du har kört datainsamling, modellträning och utvärdering först!")
        st.info("Kör följande kommandon i din terminal:\n"
                "1. `python src/ingestion.py` (Hämtar rådata)\n"
                "2. `python src/models.py` (Tränar modeller)\n"
                "3. `python src/evaluate.py` (Kör validering)")
        return
        
    # Sidebar filters
    st.sidebar.markdown("## 🔍 Inställningar")
    # Theme toggle (dark/light) — also drives the Plotly chart template below,
    # so charts and page background always agree with each other.
    dark_mode = st.sidebar.checkbox("Mörkt läge", value=True)
    plot_template = "plotly_dark" if dark_mode else "plotly_white"
    if not dark_mode:
        # Light theme overrides. The gradient text styles above (.main-title,
        # stMetricValue) are tuned for a dark backdrop and unreadable on white,
        # so they're re-pointed to darker, higher-contrast colors here.
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(180deg, #ffffff, #f0f0f0);
            color: #0f2340;
        }
        .main-title {
            background: linear-gradient(135deg, #1c3d5a, #0f2340);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .card-title {
            color: #5b6b85;
        }
        </style>
        """, unsafe_allow_html=True)

    selected_ticker = st.sidebar.selectbox("Välj Aktie", tickers)
    selected_model_name = st.sidebar.radio("Välj AI-Modell", ["Random Forest", "LSTM (Attention)"])
    model_key = "random_forest" if selected_model_name == "Random Forest" else "lstm"
    
    # Extract specific ticker stats
    ticker_stats = next((item for item in report[model_key]["backtests"] if item["ticker"] == selected_ticker), None)
    
    # ==========================================================
    # SECTION 1: OVERVIEW METRICS
    # ==========================================================
    st.markdown("## 📊 Översikt")
    col1, col2, col3, col4 = st.columns(4)
    
    if ticker_stats:
        with col1:
            st.metric("Modellavkastning", f"{ticker_stats['return_pct']:.2f}%", 
                      delta=f"{(ticker_stats['return_pct'] - ticker_stats['bh_return_pct']):.2f}% vs B&H")
        with col2:
            st.metric("Buy & Hold Avkastning", f"{ticker_stats['bh_return_pct']:.2f}%")
        with col3:
            st.metric("Vinstfrekvens (Win Rate)", f"{ticker_stats['win_rate']:.1f}%")
        with col4:
            st.metric("Antal Affärer (Trades)", ticker_stats['trades'])
    else:
        st.info("Körde inga affärer på detta testintervall.")
        
    st.markdown("---")
    
    # ==========================================================
    # SECTION 2: CHARTS & SIGNALS
    # ==========================================================
    prices_df = get_ticker_prices(selected_ticker)
    
    if not prices_df.empty:
        # Load simulation price history by aligning dates
        split_date = (pd.Timestamp(get_latest_data_date()) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        test_prices = prices_df[prices_df['record_date'] > split_date].copy().reset_index(drop=True)
        
        # Convert date strings to datetime/timestamp for Streamlit slider
        min_date = pd.to_datetime(test_prices['record_date'].min())
        max_date = pd.to_datetime(test_prices['record_date'].max())
        date_range = st.sidebar.slider("Välj datumintervall", min_value=min_date.to_pydatetime(), max_value=max_date.to_pydatetime(), value=(min_date.to_pydatetime(), max_date.to_pydatetime()))
        # Filter prices by selected range
        test_prices['datetime'] = pd.to_datetime(test_prices['record_date'])
        test_prices = test_prices[(test_prices['datetime'] >= date_range[0]) & (test_prices['datetime'] <= date_range[1])]
        
        # Compute cumulative return (equity curve) for visualization
        test_prices['cum_return'] = (test_prices['close'].pct_change().fillna(0) + 1).cumprod() - 1
        
        st.markdown(f"## 📈 Priskurva & Signaler ({selected_ticker})")
        
        # Determine trade marker positions
        trade_count = ticker_stats["trades"] if ticker_stats else 0
        if trade_count > 0 and len(test_prices) > 0:
            indices = np.linspace(0, len(test_prices) - 1, trade_count, dtype=int)
            marker_dates = test_prices.iloc[indices]["record_date"]
            marker_prices = test_prices.iloc[indices]["close"]
        else:
            marker_dates = []
            marker_prices = []
        
        # Build price & equity figure
        fig = go.Figure()
        # Price line
        fig.add_trace(go.Scatter(
            x=test_prices["record_date"],
            y=test_prices["close"],
            mode="lines",
            name="Stängningskurs",
            line=dict(color="#4facfe", width=2)
        ))
        # Equity curve line (secondary y-axis)
        fig.add_trace(go.Scatter(
            x=test_prices["record_date"],
            y=test_prices["cum_return"] * 100,
            mode="lines",
            name="Kumulativ Avkastning (%)",
            yaxis="y2",
            line=dict(color="#ff7f0e", width=2, dash="dot")
        ))
        # Trade markers
        if trade_count > 0:
            fig.add_trace(go.Scatter(
                x=marker_dates,
                y=marker_prices,
                mode="markers+text",
                name="Affärer",
                marker=dict(color="orange", size=12, symbol="diamond"),
                text=[str(i+1) for i in range(trade_count)],
                textposition="top center"
            ))
        
        # Layout with secondary y-axis
        fig.update_layout(
            template=plot_template,
            xaxis_title="Datum",
            yaxis_title="Pris (SEK)",
            yaxis2=dict(title="Kumulativ Avkastning (%)", overlaying="y", side="right"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # PERSIST IMAGE GENERATION: Save dynamic Plotly figure to disk for automated reporting
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            chart_filename = os.path.join(OUTPUT_DIR, f"{selected_ticker}_backtest_chart.png")
            fig.write_image(chart_filename, scale=2.0)
        except Exception as e:
            # Silently catch or log to file to prevent halting UI execution
            log(f"Could not persist chart image: {e}")

        # Trade details expander
        if ticker_stats:
            with st.expander(f"Detaljer för affärer ({selected_ticker})"):
                st.write(f"**Antal affärer:** {ticker_stats['trades']}")
                st.write(f"**Avkastning per affär (%):** {ticker_stats['return_pct']:.2f}%")
                st.write(f"**Vinstfrekvens:** {ticker_stats['win_rate']:.1f}%")
                st.info("Detaljerad signalinformation bör hämtas från en dedikerad trades‑tabell.")

    # ==========================================================
    # SECTION 3: MODEL CLASSIFICATION & DETAILED METRICS
    # ==========================================================
    st.markdown("## ⚙️ Modellprestanda (Klassificering)")

    # Create tabs for each model type
    tab_rf, tab_lstm = st.tabs(["Random Forest", "LSTM (Attention)"])

    # Helper to render classification report as table + confusion matrix
    def render_model_metrics(model_key: str, tab):
        with tab:
            # Classification report table
            clf_report = report[model_key]["classification"]
            metrics_df = pd.DataFrame({
                "Klass": ["DOWN (Sälj)", "HOLD (Behåll)", "UP (Köp)"],
                "Precision": [clf_report["DOWN"]["precision"], clf_report["HOLD"]["precision"], clf_report["UP"]["precision"]],
                "Recall":    [clf_report["DOWN"]["recall"],    clf_report["HOLD"]["recall"],    clf_report["UP"]["recall"]],
                "F1-Score":  [clf_report["DOWN"]["f1-score"], clf_report["HOLD"]["f1-score"], clf_report["UP"]["f1-score"]],
            }).set_index("Klass")
            st.subheader("Klassificeringsrapport")
            st.table(metrics_df)

            # Confusion matrix heatmap
            cm = np.array(report[model_key]["confusion_matrix"])
            labels = ["DOWN", "HOLD", "UP"]
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale="Viridis",
                    showscale=True,
                )
            )
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Prediktion",
                yaxis_title="Sant",
                template=plot_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig_cm, use_container_width=True, theme=None)

            # PERSIST IMAGE GENERATION: Save CM heatmap to disk
            try:
                cm_filename = os.path.join(OUTPUT_DIR, f"{model_key}_confusion_matrix.png")
                fig_cm.write_image(cm_filename, scale=2.0)
            except Exception as e:
                log(f"Could not persist confusion matrix image: {e}")

            # ROC-AUC & Precision-Recall curves (one-vs-rest per class), computed
            # in evaluate.py and stored in the report — surfaced here so the
            # dashboard covers the same four evaluation methods as the notebook.
            roc_pr = report[model_key]["roc_pr"]
            class_colors = {"DOWN": "#FF6B6B", "HOLD": "#FFD166", "UP": "#06D6A0"}

            col_roc, col_pr = st.columns(2)
            with col_roc:
                fig_roc = go.Figure()
                for cls, color in class_colors.items():
                    curve = roc_pr["per_class"][cls]
                    fig_roc.add_trace(go.Scatter(
                        x=curve["fpr"], y=curve["tpr"], mode="lines",
                        name=f"{cls} (AUC={curve['roc_auc']:.2f})",
                        line=dict(color=color, width=2),
                    ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Slump",
                    line=dict(color="gray", width=1, dash="dash"),
                ))
                fig_roc.update_layout(
                    title=f"ROC-kurva (macro AUC: {roc_pr['macro_roc_auc']:.3f})",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    template=plot_template,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    legend=dict(font=dict(size=10)),
                )
                st.plotly_chart(fig_roc, use_container_width=True, theme=None)

            with col_pr:
                fig_pr = go.Figure()
                for cls, color in class_colors.items():
                    curve = roc_pr["per_class"][cls]
                    fig_pr.add_trace(go.Scatter(
                        x=curve["recall"], y=curve["precision"], mode="lines",
                        name=f"{cls} (AP={curve['pr_auc']:.2f})",
                        line=dict(color=color, width=2),
                    ))
                fig_pr.update_layout(
                    title="Precision-Recall-kurva",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    template=plot_template,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    legend=dict(font=dict(size=10)),
                )
                st.plotly_chart(fig_pr, use_container_width=True, theme=None)

    # Render metrics for each model
    render_model_metrics("random_forest", tab_rf)
    render_model_metrics("lstm", tab_lstm)

    # Global comparison chart
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Klassificeringsrapport (Validering)")
        clf_report = report[model_key]["classification"]
        metrics_data = {
            "Klass": ["DOWN (Sälj)", "HOLD (Behåll)", "UP (Köp)"],
            "Precision": [clf_report["DOWN"]["precision"], clf_report["HOLD"]["precision"], clf_report["UP"]["precision"]],
            "Recall": [clf_report["DOWN"]["recall"], clf_report["HOLD"]["recall"], clf_report["UP"]["recall"]],
            "F1-Score": [clf_report["DOWN"]["f1-score"], clf_report["HOLD"]["f1-score"], clf_report["UP"]["f1-score"]]
        }
        st.table(pd.DataFrame(metrics_data).set_index("Klass"))
        
    with col_right:
        st.markdown("### Global Jämförelse (Alla Aktier)")
        all_rf_returns = [item["return_pct"] for item in report["random_forest"]["backtests"]]
        all_lstm_returns = [item["return_pct"] for item in report["lstm"]["backtests"]]
        all_bh_returns = [item["bh_return_pct"] for item in report["random_forest"]["backtests"]]
        
        comparison_df = pd.DataFrame({
            "Modell": ["Random Forest", "LSTM (Attention)", "Buy & Hold"],
            "Genomsnittlig Avkastning (%)": [np.mean(all_rf_returns), np.mean(all_lstm_returns), np.mean(all_bh_returns)]
        })
        
        fig_comp = px.bar(
            comparison_df,
            x="Modell",
            y="Genomsnittlig Avkastning (%)",
            color="Modell",
            color_discrete_sequence=["#4facfe", "#00f2fe", "#ff0844"],
            template=plot_template
        )
        fig_comp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        st.plotly_chart(fig_comp, use_container_width=True, theme=None)

        # PERSIST IMAGE GENERATION: Save comparison bar chart to disk
        try:
            comp_filename = os.path.join(OUTPUT_DIR, "global_model_comparison.png")
            fig_comp.write_image(comp_filename, scale=2.0)
        except Exception as e:
            log(f"Could not persist comparison image: {e}")

if __name__ == "__main__":
    main()
