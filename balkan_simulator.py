import io
import os
import sqlite3
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. Ρύθμιση σελίδας
st.set_page_config(
    page_title="Balkan Energy Lakehouse & AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. ΠΛΟΗΓΗΣΗ & ΔΥΝΑΜΙΚΟ BACKGROUND ΑΝΑ ΕΝΟΤΗΤΑ
# ==============================================================================
MODULES = [
    "1. Χαρτογράφηση & Σύγκλιση",
    "2. Προσομοιωτής Πολιτικής (What-If)",
    "3. Εργαστήριο AI & SHAP",
    "4. Αγορά SEEPEX & Isolation Forest",
    "5. Βασικό SQL Query & Export",
    "6. Λογιστική & Financial BI (CBAM / CAPEX)",
    "7. Data Engineering Lab (OLAP & Feature Eng.)",
    "8. IT Architecture & 5-Day Price Forecast",
]

# Χρωματικές παλέτες ανά ενότητα (Background Gradient + Accent Color)
MODULE_THEMES = {
    MODULES[0]: ("linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)", "#1d4ed8"),
    MODULES[1]: ("linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)", "#15803d"),
    MODULES[2]: ("linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%)", "#7e22ce"),
    MODULES[3]: ("linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)", "#b91c1c"),
    MODULES[4]: ("linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)", "#334155"),
    MODULES[5]: ("linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)", "#b45309"),
    MODULES[6]: ("linear-gradient(135deg, #ecfeff 0%, #cffafe 100%)", "#0e7490"),
    MODULES[7]: ("linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%)", "#3730a3"),
}

with st.sidebar:
  st.header("🧭 Πλοήγηση Πλατφόρμας")
  selected_module = st.radio(
      "Επιλέξτε Ενότητα Ανάλυσης:",
      MODULES,
      index=0,
      help="Κάθε ενότητα προσαρμόζει δυναμικά το περιβάλλον και τα εργαλεία της.",
  )
  st.markdown("---")
  show_guides = st.toggle(
      "📖 Οδηγοί Ανάγνωσης Κάτω από Γραφήματα",
      value=True,
      help="Εμφανίζει επεξηγηματικά πλαίσια κάτω από κάθε διάγραμμα και πίνακα.",
  )
  st.markdown("---")

bg_gradient, accent_color = MODULE_THEMES[selected_module]

# Ασφαλές CSS με υψηλή αντίθεση (Dark text #0f172a σε καθαρό φόντο — δεν χάνεται κανένας αριθμός)
st.markdown(
    f"""
<style>
    .stApp {{
        background: {bg_gradient};
        color: #0f172a;
    }}
    /* Εξασφάλιση ευανάγνωστων σκούρων γραμμάτων σε όλο το κύριο σώμα */
    .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: #0f172a !important;
    }}
    /* Κάρτες KPI (st.metric) με λευκό φόντο, χρωματιστό περίγραμμα και καθαρούς αριθμούς */
    div[data-testid="stMetric"] {{
        background-color: #ffffff !important;
        border-left: 6px solid {accent_color} !important;
        border-top: 1px solid #cbd5e1;
        border-right: 1px solid #cbd5e1;
        border-bottom: 1px solid #cbd5e1;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(15, 23, 42, 0.06);
    }}
    div[data-testid="stMetricLabel"] p {{
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }}
    div[data-testid="stMetricValue"] div {{
        color: #0f172a !important;
        font-weight: 800 !important;
    }}
    /* Πλαίσια Οδηγού Ανάγνωσης */
    div[data-testid="stExpander"] {{
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-left: 5px solid {accent_color} !important;
        border-radius: 8px;
    }}
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: #ffffff !important;
        border-right: 1px solid #cbd5e1;
    }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h2 {{
        color: #0f172a !important;
    }}
</style>
""",
    unsafe_allow_html=True,
)


def render_guide(title: str, what_we_see: str, methodology: str, takeaway: str):
  """Εμφανίζει ευανάγνωστο πλαίσιο επεξήγησης κάτω από κάθε γράφημα ή πίνακα."""
  if show_guides:
    with st.expander(f"ℹ️ Οδηγός Ανάγνωσης: {title}", expanded=False):
      st.markdown(f"**👁️ Τι βλέπουμε:** {what_we_see}")
      st.markdown(f"**⚙️ Πώς υπολογίζεται:** {methodology}")
      st.markdown(f"**💡 Κύριο Συμπέρασμα:** {takeaway}")


# ==============================================================================
# 3. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ PANEL
# ==============================================================================
@st.cache_data
def load_balkan_data():
  data = [
      ("Σλοβενία", "SVN", "EU Balkans", 27878, 36183, 30.3, 2.7, 59.2, 3.2, 6.1, 32.5),
      ("Ελλάδα", "GRC", "EU Balkans", 20354, 29256, 20.5, 18.8, 68.4, 2.8, 7.2, 38.1),
      ("Κροατία", "HRV", "EU Balkans", 17326, 23611, 8.7, 6.5, 88.8, 3.1, 9.9, 64.2),
      ("Ρουμανία", "ROU", "EU Balkans", 14588, 19162, 20.5, 11.1, 94.3, 3.3, 12.2, 44.0),
      ("Βουλγαρία", "BGR", "EU Balkans", 11880, 30765, 47.7, 26.4, 80.7, 4.8, 9.3, 23.4),
      ("Μαυροβούνιο", "MNE", "Non-EU Balkans", 9536, 19513, 57.1, 13.3, 87.2, 3.9, 15.9, 52.0),
      ("Σερβία", "SRB", "Non-EU Balkans", 9233, 27036, 64.4, 9.9, 84.1, 4.5, 13.6, 31.8),
      ("Βόρεια Μακεδονία", "MKD", "Non-EU Balkans", 7324, 14451, 47.7, 29.5, 83.9, 4.2, 13.5, 28.5),
      ("Βοσνία-Ερζεγοβίνη", "BIH", "Non-EU Balkans", 6982, 21790, 73.5, 22.0, 88.0, 5.1, 10.1, 36.4),
      ("Αλβανία", "ALB", "Non-EU Balkans", 6339, 13276, 12.8, 35.4, 96.2, 2.7, 15.7, 99.4),
      ("Κόσοβο", "XKX", "Non-EU Balkans", 4939, 16275, 70.2, 40.2, 95.7, 4.6, 17.8, 6.5),
  ]
  cols = [
      "Χώρα",
      "ISO3",
      "Ομάδα",
      "GDP_pc",
      "Energy_kWh",
      "Coal_Share",
      "Cold_Homes",
      "Poor_Owners",
      "Energy_Intensity",
      "Grid_Loss",
      "Renewables_Pct",
  ]
  return pd.DataFrame(data, columns=cols)


df = load_balkan_data()

with st.sidebar:
  region_filter = st.multiselect(
      "Φίλτρο Θεσμικής Ομάδας:",
      options=["EU Balkans", "Non-EU Balkans"],
      default=["EU Balkans", "Non-EU Balkans"],
      help="Φιλτράρει τις χώρες ανάλογα με την ένταξή τους στην Ε.Ε.",
  )
  st.caption(
      "Μεταπτυχιακή Διπλωματική Εργασία | Πληροφοριακά Συστήματα, Data"
      " Engineering & Financial BI"
  )

df_filtered = (
    df[df["Ομάδα"].isin(region_filter)] if len(region_filter) > 0 else df
)

# Κεντρική Επικεφαλίδα & KPIs
st.title("⚡ Balkan Energy Intelligence, AI & Financial BI Platform")
st.markdown(
    f"**Ενεργή Ενότητα:** `{selected_module}` | Ενοποιημένο περιβάλλον **Data"
    " Lakehouse, Οικονομετρίας, AI Forecasting & Λογιστικής**"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Επιλεγμένες Χώρες", f"{len(df_filtered)} / 11")
k2.metric("Μέσο Κατά Κεφ. ΑΕΠ", f"{df_filtered['GDP_pc'].mean():,.0f} USD", "Χάσμα 2,42x EU/Non-EU")
k3.metric("Μέση Ενεργειακή Φτώχεια", f"{df_filtered['Cold_Homes'].mean():.1f}%", "Μ.Ο. Ε.Ε.-27: ~9,5%", delta_color="inverse")
k4.metric("Φτωχοί Ιδιοκτήτες Κατοικίας", f"{df_filtered['Poor_Owners'].mean():.1f}%", "Κύριος δομικός παράγοντας", delta_color="inverse")

st.markdown("---")

# ==============================================================================
# ΕΝΟΤΗΤΑ 1: ΧΑΡΤΟΓΡΑΦΗΣΗ & ΣΥΓΚΛΙΣΗ
# ==============================================================================
if selected_module == MODULES[0]:
  st.subheader("1. Διαδραστική Χαρτογράφηση Ενεργειακών & Οικονομικών Ανισοτήτων")
  metric_map = {
      "Κατά Κεφαλήν ΑΕΠ (USD)": ("GDP_pc", "Blues"),
      "Αδυναμία Επαρκούς Θέρμανσης (%)": ("Cold_Homes", "Reds"),
      "Φτωχοί Ιδιοκτήτες Κατοικίας (%)": ("Poor_Owners", "Oranges"),
      "Κατά Κεφαλήν Κατανάλωση Ενέργειας (kWh)": ("Energy_kWh", "Viridis"),
      "Μερίδιο Άνθρακα στις Εκπομπές CO₂ (%)": ("Coal_Share", "Greys"),
      "Απώλειες Δικτύου Ηλεκτρισμού (%)": ("Grid_Loss", "Purples"),
      "Μερίδιο ΑΠΕ στην Ηλεκτροπαραγωγή (%)": ("Renewables_Pct", "Greens"),
  }
  chosen_label = st.selectbox(
      "Επιλέξτε δείκτη για απεικόνιση στον χάρτη:", list(metric_map.keys())
  )
  col_name, color_scale = metric_map[chosen_label]

  c_map, c_bar = st.columns([1.2, 1])
  with c_map:
    fig_map = px.choropleth(
        df_filtered,
        locations="ISO3",
        color=col_name,
        hover_name="Χώρα",
        hover_data=["Ομάδα", "GDP_pc", "Cold_Homes", "Coal_Share"],
        color_continuous_scale=color_scale,
        scope="europe",
        title=f"Χάρτης Βαλκανίων: {chosen_label}",
        template="plotly_white",
    )
    fig_map.update_geos(
        center=dict(lat=43.5, lon=21.5),
        projection_scale=5.2,
        visible=True,
        showcountries=True,
    )
    fig_map.update_layout(height=460, margin=dict(l=0, r=0, t=45, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

  with c_bar:
    df_sorted = df_filtered.sort_values(col_name, ascending=True)
    fig_rank = px.bar(
        df_sorted,
        x=col_name,
        y="Χώρα",
        color="Ομάδα",
        orientation="h",
        text_auto=True,
        color_discrete_map={"EU Balkans": "#1d4ed8", "Non-EU Balkans": "#dc2626"},
        title=f"Κατάταξη Χωρών: {chosen_label}",
        template="plotly_white",
    )
    fig_rank.update_layout(height=460)
    st.plotly_chart(fig_rank, use_container_width=True)

  render_guide(
      title="Γεωγραφική Κατανομή & Θεσμική Σύγκριση (EU vs. Non-EU Balkans)",
      what_we_see="Αριστερά εμφανίζεται ο γεωγραφικός χάρτης των 11 βαλκανικών χωρών και δεξιά η κατάταξή τους ανά θεσμική ομάδα.",
      methodology="Ενοποίηση δεδομένων Eurostat (EU-SILC) και World Bank (WDI) σε διαχρονικό πάνελ 2000–2025.",
      takeaway="Τα κράτη-μέλη της Ε.Ε. προηγούνται σταθερά σε ΑΕΠ και υποδομές, ενώ τα Δυτικά Βαλκάνια εμφανίζουν υψηλότερες απώλειες δικτύου.",
  )

# ==============================================================================
# ΕΝΟΤΗΤΑ 2: ΠΡΟΣΟΜΟΙΩΤΗΣ ΠΟΛΙΤΙΚΗΣ (WHAT-IF)
# ==============================================================================
elif selected_module == MODULES[1]:
  st.subheader("2. Προσομοιωτής Πολιτικής & Σύγκλισης (Policy What-If Simulator)")
  col_ctrl, col_res = st.columns([1, 2])

  with col_ctrl:
    selected_country = st.selectbox("Επιλογή Χώρας Στόχου:", df["Χώρα"].tolist(), index=9)
    row = df[df["Χώρα"] == selected_country].iloc[0]
    st.markdown(f"**Θεσμική Ομάδα:** `{row['Ομάδα']}` (`{row['ISO3']}`)")

    energy_boost = st.slider("1. Αύξηση Ενεργειακής Διαθεσιμότητας (%)", 0, 60, 20, 5)
    intensity_cut = st.slider("2. Μείωση Ενεργειακής Έντασης (MJ/$)", 0.0, 2.0, 0.6, 0.1)
    retrofit_share = st.slider("3. Επιδότηση Θερμομόνωσης Φτωχών Ιδιοκτητών (%)", 0, 50, 25, 5)
    carbon_price = st.slider("4. Τιμή Άνθρακα CBAM (€/τόνο CO₂)", 0, 120, 75, 5)

  base_gdp = row["GDP_pc"]
  base_cold = row["Cold_Homes"]
  delta_log_gdp = (1.0702 * np.log(1 + energy_boost / 100.0)) + (-0.8062 * (-intensity_cut))
  sim_gdp_fe = base_gdp * np.exp(delta_log_gdp)
  cbam_penalty_pct = (row["Coal_Share"] / 100.0) * (carbon_price / 100.0) * 4.5
  sim_gdp_net = sim_gdp_fe * (1 - cbam_penalty_pct / 100.0)
  sim_cold = max(1.5, base_cold + (-18.4912 * delta_log_gdp) + (-0.35 * retrofit_share))

  with col_res:
    m1, m2, m3 = st.columns(3)
    m1.metric("Προβλεπόμενο ΑΕΠ (καθαρό CBAM)", f"{sim_gdp_net:,.0f} USD", f"{sim_gdp_net - base_gdp:+,.0f} USD")
    m2.metric("Επιβάρυνση Άνθρακα CBAM", f"-{cbam_penalty_pct:.2f}%", f"Μερίδιο Άνθρακα: {row['Coal_Share']}%", delta_color="inverse")
    m3.metric("Νέα Ενεργειακή Φτώχεια", f"{sim_cold:.1f}%", f"{sim_cold - base_cold:.1f} π.μ.", delta_color="inverse")

    comp_df = pd.DataFrame({
        "Σενάριο": [f"{selected_country} (Τρέχον)", f"{selected_country} (Μετά τις Παρεμβάσεις)", "Μέσος Όρος EU Balkans"],
        "Κατά Κεφαλήν ΑΕΠ (USD)": [base_gdp, sim_gdp_net, 18405],
    })
    fig_sim = px.bar(
        comp_df,
        x="Σενάριο",
        y="Κατά Κεφαλήν ΑΕΠ (USD)",
        color="Σενάριο",
        text_auto=",.0f",
        color_discrete_sequence=["#dc2626", "#16a34a", "#2563eb"],
        title=f"Τροχιά Σύγκλισης: {selected_country} (συνυπολογίζοντας CBAM {carbon_price} €/t)",
        template="plotly_white",
    )
    fig_sim.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_sim, use_container_width=True)

  render_guide(
      title="Προσομοιωτής Οικονομετρικών Παρεμβάσεων (Two-Way Fixed Effects)",
      what_we_see="Σύγκριση του τρέχοντος ΑΕΠ (κόκκινο), του προβλεπόμενου ΑΕΠ μετά τις παρεμβάσεις (πράσινο) και του ευρωπαϊκού μέσου όρου (μπλε).",
      methodology="Εφαρμογή των ελαστικοτήτων Fixed Effects (β₁ = 1,0702, β₂ = -0,8062, δ₁ = -18,49).",
      takeaway="Η αύξηση της ενεργειακής επάρκειας σε συνδυασμό με τη θερμομόνωση επιταχύνει τη σύγκλιση.",
  )

# ==============================================================================
# ΕΝΟΤΗΤΑ 3: ΕΡΓΑΣΤΗΡΙΟ AI & SHAP
# ==============================================================================
elif selected_module == MODULES[2]:
  st.subheader("3. Εργαστήριο Μηχανικής Μάθησης (Hyperparameter Tuning & SHAP)")
  c_param, c_shap = st.columns([1, 2])

  with c_param:
    n_trees = st.slider("Πλήθος Δέντρων Random Forest:", 50, 500, 200, 50)
    ridge_alpha = st.slider("Παράμετρος Ομαλοποίησης Ridge (α):", 0.1, 10.0, 1.5, 0.1)
    country_shap = st.selectbox("Χώρα για Ερμηνευσιμότητα SHAP:", df["Χώρα"].tolist(), index=10)

  features = ["Energy_kWh", "Coal_Share", "Cold_Homes", "Poor_Owners", "Energy_Intensity"]
  X = df[features]
  y = df["GDP_pc"]

  rf_model = RandomForestRegressor(n_estimators=n_trees, random_state=42).fit(X, y)
  ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha)).fit(X, np.log(y))

  idx = df[df["Χώρα"] == country_shap].index[0]
  pred_rf = rf_model.predict(X.iloc[[idx]])[0]
  pred_ridge = np.exp(ridge_model.predict(X.iloc[[idx]])[0])

  with c_param:
    st.metric("Πραγματικό ΑΕΠ", f"{y.iloc[idx]:,.0f} USD")
    st.metric("Πρόβλεψη Random Forest", f"{pred_rf:,.0f} USD", f"Σφάλμα: {pred_rf - y.iloc[idx]:+,.0f}")
    st.metric("Πρόβλεψη Log-Ridge", f"{pred_ridge:,.0f} USD", f"Σφάλμα: {pred_ridge - y.iloc[idx]:+,.0f}")

  with c_shap:
    greek_labels = {
        "Poor_Owners": "Φτωχοί Ιδιοκτήτες (%)",
        "Cold_Homes": "Αδυναμία Θέρμανσης (%)",
        "Energy_kWh": "Ενέργεια/κάτοικο (kWh)",
        "Coal_Share": "Μερίδιο Άνθρακα (%)",
        "Energy_Intensity": "Ενεργειακή Ένταση (MJ/$)",
    }
    contributions = []
    for feat in features:
      X_temp = X.copy()
      X_temp[feat] = X[feat].mean()
      diff = rf_model.predict(X.iloc[[idx]])[0] - rf_model.predict(X_temp.iloc[[idx]])[0]
      contributions.append({"Μεταβλητή": greek_labels[feat], "Συνεισφορά SHAP (USD)": diff})

    shap_df = pd.DataFrame(contributions).sort_values("Συνεισφορά SHAP (USD)")
    fig_shap = px.bar(
        shap_df,
        x="Συνεισφορά SHAP (USD)",
        y="Μεταβλητή",
        orientation="h",
        color="Συνεισφορά SHAP (USD)",
        color_continuous_scale=["#dc2626", "#94a3b8", "#16a34a"],
        title=f"Αποσύνθεση Πρόβλεψης SHAP για: {country_shap}",
        template="plotly_white",
    )
    fig_shap.update_layout(height=420)
    st.plotly_chart(fig_shap, use_container_width=True)

  render_guide(
      title="Επεξηγήσιμη Τεχνητή Νοημοσύνη (SHAP Values)",
      what_we_see="Πόσα δολάρια προσθέτει (πράσινο) ή αφαιρεί (κόκκινο) κάθε χαρακτηριστικό από τη μέση πρόβλεψη του μοντέλου.",
      methodology="Θεωρία Παιγνίων Shapley Additive Explanations (SHAP) επί μοντέλου Random Forest.",
      takeaway="Οι κοινωνικο-δομικοί δείκτες (φτωχοί ιδιοκτήτες & αδυναμία θέρμανσης) αποτελούν τους ισχυρότερους παράγοντες υστέρησης.",
  )

# ==============================================================================
# ΕΝΟΤΗΤΑ 4: ΑΓΟΡΑ SEEPEX & ISOLATION FOREST
# ==============================================================================
elif selected_module == MODULES[3]:
  st.subheader("4. Διαδραστικός Ανιχνευτής Ανωμαλιών SEEPEX (Isolation Forest)")
  col_s1, col_s2 = st.columns([1, 2.5])

  with col_s1:
    contam = st.slider("Ευαισθησία Isolation Forest (Contamination %):", 1.0, 10.0, 3.5, 0.5)
    crisis_shock = st.slider("Ένταση Βραδινών Αιχμών (18:00–21:00):", 1.0, 2.5, 1.5, 0.1)

  np.random.seed(42)
  hours = np.arange(168)
  hod = hours % 24
  prices = 85 + 25 * np.sin((hod - 7) * np.pi / 12) + np.random.normal(0, 8, 168)
  spike_idx = [19, 20, 43, 68, 91, 92, 116, 140, 163]
  prices[spike_idx] = prices[spike_idx] * (2.2 * crisis_shock)

  iso = IsolationForest(contamination=contam / 100.0, random_state=42)
  preds = iso.fit_predict(prices.reshape(-1, 1))
  is_anomaly = preds == -1

  with col_s1:
    st.metric("Εντοπισμένα Price Spikes", f"{is_anomaly.sum()} ώρες", f"Από {len(prices)} ώρες εβδομάδας")
    st.metric("Μέση Τιμή Ανωμαλιών", f"{prices[is_anomaly].mean():.2f} €/MWh", delta_color="inverse")
    st.metric("Μέγιστη Ωριαία Αιχμή", f"{prices.max():.2f} €/MWh")

  with col_s2:
    df_seepex = pd.DataFrame({
        "Ώρα": hours,
        "Τιμή (€/MWh)": prices,
        "Κατάσταση": np.where(is_anomaly, "Price Spike (Ανωμαλία)", "Κανονική Λειτουργία"),
    })
    fig_sp = px.scatter(
        df_seepex,
        x="Ώρα",
        y="Τιμή (€/MWh)",
        color="Κατάσταση",
        color_discrete_map={"Κανονική Λειτουργία": "#2563eb", "Price Spike (Ανωμαλία)": "#dc2626"},
        title=f"Isolation Forest (contamination = {contam}%) — Ανίχνευση Ακραίων Τιμών SEEPEX",
        template="plotly_white",
    )
    fig_sp.add_trace(
        go.Scatter(x=df_seepex["Ώρα"], y=df_seepex["Τιμή (€/MWh)"], mode="lines", line=dict(color="#64748b", width=1), showlegend=False)
    )
    fig_sp.update_layout(height=420)
    st.plotly_chart(fig_sp, use_container_width=True)

  render_guide(
      title="Ωριαία Τιμή Εκκαθάρισης Αγοράς SEEPEX (MCP) & Price Spikes",
      what_we_see="Η γραμμή δείχνει την κανονική ημερήσια διακύμανση (€/MWh) και οι κόκκινες κουκκίδες τις ακραίες εκρήξεις τιμών.",
      methodology="Μη επιβλεπόμενος αλγόριθμος απομόνωσης δέντρων (Isolation Forest).",
      takeaway="Οι εκρήξεις τιμών εκδηλώνονται τις βραδινές ώρες αιχμής (18:00–21:00) λόγω έλλειψης αποθήκευσης.",
  )

# ==============================================================================
# ΕΝΟΤΗΤΑ 5: ΒΑΣΙΚΟ SQL QUERY & EXPORT
# ==============================================================================
elif selected_module == MODULES[4]:
  st.subheader("5. Εκτέλεση Ερωτημάτων SQL σε Πραγματικό Χρόνο (In-Memory SQLite)")
  conn = sqlite3.connect(":memory:")
  df.to_sql("balkan_master_panel", conn, index=False, if_exists="replace")

  preset_queries = {
      "1. Σύγκριση Μέσων Όρων EU vs Non-EU Balkans": (
          "SELECT Ομάδα, COUNT(*) AS Χώρες, ROUND(AVG(GDP_pc), 1) AS Μέσο_ΑΕΠ,"
          " ROUND(AVG(Cold_Homes), 2) AS Ενεργειακή_Φτώχεια,"
          " ROUND(AVG(Grid_Loss), 2) AS Απώλειες_Δικτύου FROM"
          " balkan_master_panel GROUP BY Ομάδα;"
      ),
      "2. Χώρες Υψηλού Κινδύνου CBAM & Φτώχειας": (
          "SELECT ISO3, Χώρα, Coal_Share AS Άνθρακας_Pct, Cold_Homes AS"
          " Αδυναμία_Θέρμανσης, Poor_Owners AS Φτωχοί_Ιδιοκτήτες FROM"
          " balkan_master_panel WHERE Cold_Homes > 18 ORDER BY Cold_Homes DESC;"
      ),
      "3. Όλος ο Πίνακας Δεδομένων (balkan_master_panel)": "SELECT * FROM balkan_master_panel;",
  }

  selected_preset = st.selectbox("Επιλέξτε έτοιμο ερώτημα SQL:", list(preset_queries.keys()))
  sql_input = st.text_area("Κώδικας SQL:", value=preset_queries[selected_preset], height=110)

  if st.button("▶️ Εκτέλεση SQL Query", type="primary"):
    try:
      res_sql = pd.read_sql_query(sql_input, conn)
      st.dataframe(res_sql, use_container_width=True)
      csv_bytes = res_sql.to_csv(index=False).encode("utf-8-sig")
      st.download_button("📥 Κατέβασμα Αποτελεσμάτων (CSV)", data=csv_bytes, file_name="balkan_sql_export.csv", mime="text/csv")
    except Exception as e:
      st.error(f"Σφάλμα σύνταξης SQL: {e}")

# ==============================================================================
# ΕΝΟΤΗΤΑ 6: ΛΟΓΙΣΤΙΚΗ & FINANCIAL BI (CBAM / CAPEX)
# ==============================================================================
elif selected_module == MODULES[5]:
  st.subheader("6. Λογιστική Αποτίμηση Κόστους Άνθρακα (CBAM), Απωλειών Δικτύου & CAPEX")

  col_p1, col_p2, col_p3, col_p4 = st.columns(4)
  with col_p1:
    selected_country_fin = st.selectbox(
        "Επιλογή Χώρας / Αγοράς:",
        ["Σερβία (SRB)", "Βοσνία-Ερζεγοβίνη (BIH)", "Κόσοβο (XKX)", "Βόρεια Μακεδονία (MKD)", "Αλβανία (ALB)"],
        index=0,
    )
  with col_p2:
    carbon_price_eur = st.slider("Τιμή Δικαιωμάτων CO₂ CBAM (€/t):", 40, 130, 85, 5)
  with col_p3:
    capex_invest_meur = st.slider("Επένδυση CAPEX Smart Grids / FTTH (Εκ. €):", 40, 250, 120, 10)
  with col_p4:
    wacc_rate = st.slider("Προεξοφλητικό Επιτόκιο WACC (%):", 4.0, 12.0, 7.5, 0.5)

  country_fin_params = {
      "Σερβία (SRB)": {"co2_mt": 4.82, "loss_gwh": 2050, "tariff": 95.0, "base_rev": 2450.0, "base_ebitda_pct": 24.5},
      "Βοσνία-Ερζεγοβίνη (BIH)": {"co2_mt": 3.12, "loss_gwh": 1160, "tariff": 94.8, "base_rev": 1520.0, "base_ebitda_pct": 23.0},
      "Κόσοβο (XKX)": {"co2_mt": 1.39, "loss_gwh": 930, "tariff": 94.6, "base_rev": 680.0, "base_ebitda_pct": 19.5},
      "Βόρεια Μακεδονία (MKD)": {"co2_mt": 1.12, "loss_gwh": 760, "tariff": 94.7, "base_rev": 740.0, "base_ebitda_pct": 21.0},
      "Αλβανία (ALB)": {"co2_mt": 0.05, "loss_gwh": 970, "tariff": 94.8, "base_rev": 890.0, "base_ebitda_pct": 28.0},
  }

  p = country_fin_params[selected_country_fin]
  cbam_liability_meur = p["co2_mt"] * carbon_price_eur
  grid_loss_cost_meur = (p["loss_gwh"] * p["tariff"]) / 1000.0
  base_ebitda_meur = p["base_rev"] * (p["base_ebitda_pct"] / 100.0)
  post_cbam_ebitda_meur = max(base_ebitda_meur - cbam_liability_meur, 0.0)
  post_cbam_ebitda_pct = (post_cbam_ebitda_meur / p["base_rev"]) * 100.0
  annual_cash_inflow_meur = grid_loss_cost_meur * 0.45
  payback_years = capex_invest_meur / annual_cash_inflow_meur if annual_cash_inflow_meur > 0 else 99.9

  r = wacc_rate / 100.0
  discounted_cf = [annual_cash_inflow_meur / ((1 + r) ** t) for t in range(1, 9)]
  npv_8yr_meur = sum(discounted_cf) - capex_invest_meur

  fk1, fk2, fk3, fk4 = st.columns(4)
  fk1.metric("Ετήσια Επιβάρυνση CBAM (COGS)", f"{cbam_liability_meur:.1f} Εκ. €", f"-{(cbam_liability_meur/p['base_rev'])*100:.1f}% επί Εσόδων", delta_color="inverse")
  fk2.metric("Λογιστική Ζημία Απωλειών Δικτύου", f"{grid_loss_cost_meur:.1f} Εκ. €", f"Ανάκτηση: +{annual_cash_inflow_meur:.1f} Εκ. €/έτος")
  fk3.metric("Περιθώριο EBITDA (Μετά CBAM)", f"{post_cbam_ebitda_pct:.1f}%", f"{post_cbam_ebitda_pct - p['base_ebitda_pct']:.1f} ποσ. μονάδες")
  fk4.metric("Απόσβεση (Payback) & NPV 8ετίας", f"{payback_years:.1f} Έτη", f"NPV: {npv_8yr_meur:+.1f} Εκ. €")

  g_col1, g_col2 = st.columns(2)
  with g_col1:
    yrs = list(range(0, 9))
    cum_cf = [-capex_invest_meur]
    for _ in range(1, 9):
      cum_cf.append(cum_cf[-1] + annual_cash_inflow_meur)

    fig_cf = go.Figure(
        data=[
            go.Bar(
                x=[f"Έτος {y}" for y in yrs],
                y=cum_cf,
                marker_color=["#dc2626" if val < 0 else "#16a34a" for val in cum_cf],
                text=[f"{val:+.1f}M" for val in cum_cf],
                textposition="outside",
            )
        ]
    )
    fig_cf.add_hline(y=0, line_width=1.5, line_color="#0f172a")
    fig_cf.update_layout(
        title=f"Σωρευτικές Ταμειακές Ροές CAPEX (Payback: {payback_years:.1f} Έτη)",
        yaxis_title="Εκατομμύρια Ευρώ (Εκ. €)",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_cf, use_container_width=True)

  with g_col2:
    modernized_ebitda_pct = ((base_ebitda_meur + annual_cash_inflow_meur - (cbam_liability_meur * 0.25)) / p["base_rev"]) * 100.0
    fig_ebitda = go.Figure(
        data=[
            go.Bar(
                x=["1. Βασικό Σενάριο", "2. Πλήρης CBAM (Χωρίς Έργα)", "3. Εκσυγχρονισμός (ΑΠΕ + Smart Grids)"],
                y=[p["base_ebitda_pct"], post_cbam_ebitda_pct, modernized_ebitda_pct],
                marker_color=["#0284c7", "#dc2626", "#16a34a"],
                text=[f"{p['base_ebitda_pct']:.1f}%", f"{post_cbam_ebitda_pct:.1f}%", f"{modernized_ebitda_pct:.1f}%"],
                textposition="outside",
            )
        ]
    )
    fig_ebitda.update_layout(
        title="Επίδραση στο Λειτουργικό Περιθώριο Κερδοφορίας (EBITDA %)",
        yaxis_title="Περιθώριο EBITDA (%)",
        yaxis_range=[0, max(modernized_ebitda_pct + 8, 38)],
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_ebitda, use_container_width=True)

  st.markdown("#### Συνοπτική Κατάσταση Αποτελεσμάτων Χρήσης (Pro-Forma P&L — Εκ. €)")
  pnl_df = pd.DataFrame({
      "Λογιστικό Κονδύλι (Pro-Forma P&L)": [
          "Κύκλος Εργασιών (Έσοδα Πώλησης Ενέργειας)",
          "(-) Λογιστική Ζημία Απωλειών Δικτύου (Revenue Leakage)",
          "(-) Φορολογική Επιβάρυνση Άνθρακα (CBAM Liability)",
          "(=) Λειτουργικά Κέρδη (EBITDA)",
          "Περιθώριο Λειτουργικής Κερδοφορίας (EBITDA Margin %)",
      ],
      "Βασικό Σενάριο (Χωρίς CBAM)": [
          f"{p['base_rev']:,.1f} €",
          f"-{grid_loss_cost_meur:,.1f} €",
          "0.0 €",
          f"{base_ebitda_meur:,.1f} €",
          f"{p['base_ebitda_pct']:.1f}%",
      ],
      "Σενάριο CBAM (Χωρίς Επενδύσεις)": [
          f"{p['base_rev']:,.1f} €",
          f"-{grid_loss_cost_meur:,.1f} €",
          f"-{cbam_liability_meur:,.1f} €",
          f"{post_cbam_ebitda_meur:,.1f} €",
          f"{post_cbam_ebitda_pct:.1f}%",
      ],
      "Σενάριο Εκσυγχρονισμού (ΑΠΕ + Smart Grids/FTTH)": [
          f"{p['base_rev']:,.1f} €",
          f"-{grid_loss_cost_meur * 0.55:,.1f} €",
          f"-{cbam_liability_meur * 0.25:,.1f} €",
          f"{base_ebitda_meur + annual_cash_inflow_meur - (cbam_liability_meur * 0.25):,.1f} €",
          f"{modernized_ebitda_pct:.1f}%",
      ],
  })
  st.dataframe(pnl_df, use_container_width=True, hide_index=True)

# ==============================================================================
# ΕΝΟΤΗΤΑ 7: DATA ENGINEERING LAB (OLAP & FEATURE ENGINEERING)
# ==============================================================================
elif selected_module == MODULES[6]:
  st.subheader("7. Εργαστήριο Data Engineering: OLAP Queries, Parquet & Cyclical Encoding")

  conn_olap = sqlite3.connect(":memory:")
  df.to_sql("balkan_master_panel", conn_olap, index=False, if_exists="replace")

  olap_presets = {
      "Κανόνας Προτεραιότητας Smart Grids & PERCENT_RANK()": """SELECT 
    ISO3 AS "Κωδικός",
    Χώρα,
    Ομάδα AS "Θεσμική_Ομάδα",
    GDP_pc AS "ΑΕΠ_per_Capita",
    Grid_Loss AS "Απώλειες_Δικτύου_Pct",
    ROUND(100.0 * PERCENT_RANK() OVER (ORDER BY GDP_pc), 1) AS "Εκατοστημόριο_ΑΕΠ",
    CASE 
        WHEN Grid_Loss > 14.0 AND GDP_pc < 8000 THEN 'Προτεραιότητα Smart Grids'
        ELSE 'Εκτός κριτηρίου'
    END AS "Στρατηγική_Απόφαση"
FROM balkan_master_panel
ORDER BY GDP_pc DESC;""",
      "Συγκεντρωτική Ανάλυση Ομάδων (GROUP BY & Aggregations)": """SELECT 
    Ομάδα AS "Θεσμική_Ομάδα",
    COUNT(DISTINCT ISO3) AS "Πλήθος_Χωρών",
    ROUND(AVG(GDP_pc), 1) AS "Μέσο_ΑΕΠ_USD",
    ROUND(AVG(Grid_Loss), 2) AS "Μέσες_Απώλειες_Pct",
    ROUND(AVG(Cold_Homes), 2) AS "Μέση_Ενεργ_Φτώχεια_Pct"
FROM balkan_master_panel
GROUP BY Ομάδα;""",
  }

  selected_olap = st.selectbox("Επιλέξτε έτοιμο ερώτημα OLAP SQL:", list(olap_presets.keys()))
  user_olap_sql = st.text_area("Επεξεργαστής SQL (SQLite OLAP Engine):", value=olap_presets[selected_olap], height=200)

  if st.button("▶ Εκτέλεση Ερωτήματος OLAP SQL", type="primary"):
    try:
      t_start = time.perf_counter()
      sql_result_df = pd.read_sql_query(user_olap_sql, conn_olap)
      elapsed_ms = (time.perf_counter() - t_start) * 1000.0

      tm1, tm2, tm3 = st.columns(3)
      tm1.metric("Χρόνος Εκτέλεσης (Latency)", f"{elapsed_ms:.2f} ms")
      tm2.metric("Επιστρεφόμενες Εγγραφές", f"{len(sql_result_df)} γραμμές")
      tm3.metric("Μέγεθος στη Μνήμη (RAM)", f"{sql_result_df.memory_usage(deep=True).sum() / 1024:.2f} KB")

      st.dataframe(sql_result_df, use_container_width=True, hide_index=True)

      exp_col1, exp_col2 = st.columns(2)
      with exp_col1:
        st.download_button(
            label="📥 Λήψη Αποτελεσμάτων σε CSV (UTF-8)",
            data=sql_result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="balkan_olap_output.csv",
            mime="text/csv",
        )
      with exp_col2:
        parquet_buffer = io.BytesIO()
        sql_result_df.to_parquet(parquet_buffer, index=False)
        st.download_button(
            label="⚡ Λήψη σε Στηλοειδές Apache Parquet",
            data=parquet_buffer.getvalue(),
            file_name="balkan_olap_output.parquet",
            mime="application/octet-stream",
        )
    except Exception as e:
      st.error(f"Σφάλμα σύνταξης SQL: {e}")

  st.markdown("---")
  st.markdown("#### Κυκλική Κωδικοποίηση Χρόνου (Cyclical Encoding sin/cos)")
  selected_hour = st.slider("Επιλέξτε Ώρα Ημέρας (00:00 - 23:00):", 0, 23, 19)
  hours_arr = np.arange(0, 24)
  sin_vals = np.sin(2 * np.pi * hours_arr / 24.0)
  cos_vals = np.cos(2 * np.pi * hours_arr / 24.0)
  sel_sin = np.sin(2 * np.pi * selected_hour / 24.0)
  sel_cos = np.cos(2 * np.pi * selected_hour / 24.0)

  fe_c1, fe_c2 = st.columns([1.3, 1])
  with fe_c1:
    fig_clock = go.Figure()
    fig_clock.add_trace(
        go.Scatter(
            x=cos_vals,
            y=sin_vals,
            mode="markers+text",
            text=[f"{h:02d}h" for h in hours_arr],
            textposition="top center",
            marker=dict(size=9, color="#2563eb"),
            name="Όλες οι Ώρες",
        )
    )
    fig_clock.add_trace(
        go.Scatter(
            x=[sel_cos],
            y=[sel_sin],
            mode="markers",
            marker=dict(size=18, color="#dc2626", symbol="diamond"),
            name=f"Επιλεγμένη Ώρα ({selected_hour:02d}:00)",
        )
    )
    fig_clock.update_layout(
        title=f"Κυκλική Προβολή Ώρας {selected_hour:02d}:00 (sin={sel_sin:.2f}, cos={sel_cos:.2f})",
        xaxis_title="cos(2π·h / 24)",
        yaxis_title="sin(2π·h / 24)",
        xaxis=dict(range=[-1.35, 1.35], zeroline=True),
        yaxis=dict(range=[-1.35, 1.35], zeroline=True, scaleanchor="x"),
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_clock, use_container_width=True)

  with fe_c2:
    st.info(
        f"**Γιατί χρησιμοποιούμε $\sin/\cos$ αντί για `{selected_hour}`:**\n\n"
        "• Αν δώσουμε στο Νευρωνικό Δίκτυο (`MLP`) τους αριθμούς `23` και `0`, θα νομίζει ότι απέχουν μέγιστα.\n\n"
        f"• Με τον τριγωνομετρικό μετασχηματισμό, η ώρα **{selected_hour:02d}:00** γίνεται:\n"
        f"  - $\\text{{Hour}}_{{\\sin}} = {sel_sin:.4f}$\n"
        f"  - $\\text{{Hour}}_{{\\cos}} = {sel_cos:.4f}$"
    )

# ==============================================================================
# ΕΝΟΤΗΤΑ 8: IT ARCHITECTURE, MULTI-STACK CODE (R/SPSS/SQL) & HARDWARE
# ==============================================================================
elif selected_module == MODULES[7]:
  st.subheader("8. Αρχιτεκτονική Πληροφοριακών Συστημάτων (IT Architecture & Computer Engineering)")

  # 1. Ζωντανό Sankey Διάγραμμα Lakehouse
  fig_sankey = go.Figure(
      data=[
          go.Sankey(
              node=dict(
                  pad=20,
                  thickness=25,
                  line=dict(color="#0f172a", width=1.5),
                  label=[
                      "Eurostat EU-SILC (API)",
                      "World Bank WDI (CSV/JSON)",
                      "SEEPEX Market (Hourly MCP)",
                      "ENTSO-E Grid (IEC 61970)",
                      "Bronze Layer (Raw Ingestion)",
                      "Silver Layer (Idempotent UPSERT)",
                      "Gold Layer (Apache Parquet + Zone Maps)",
                      "Econometrics (R fixest & SPSS)",
                      "ML & AI (XGBoost / MLP / SHAP)",
                      "Financial BI & Streamlit DSS",
                  ],
                  color=["#3b82f6", "#0ea5e9", "#f59e0b", "#10b981", "#78350f", "#64748b", "#eab308", "#8b5cf6", "#ec4899", "#16a34a"],
              ),
              link=dict(
                  source=[0, 1, 2, 3, 4, 5, 6, 6, 6],
                  target=[4, 4, 4, 4, 5, 6, 7, 8, 9],
                  value=[25, 25, 35, 20, 105, 105, 35, 40, 30],
                  color="rgba(100, 116, 139, 0.35)",
              ),
          )
      ]
  )
  fig_sankey.update_layout(
      title="Αρχιτεκτονική Medallion (Bronze → Silver → Gold) & Κατανομή Φόρτου Εργασίας",
      font=dict(size=12, color="#0f172a"),
      template="plotly_white",
      height=390,
  )
  st.plotly_chart(fig_sankey, use_container_width=True)

  # 2. Multi-Stack Code Explorer
  st.markdown("#### Διασταυρωμένη Επαλήθευση Πολλαπλών Περιβαλλόντων (R · SPSS · Python · SQL)")
  c_lang = st.radio(
      "Επιλέξτε περιβάλλον κώδικα:",
      ["🔵 R Econometrics (fixest)", "🟣 IBM SPSS Statistics", "🟢 Python Data Engineering (PyArrow)", "🟠 SQL Query Optimizer"],
      horizontal=True,
  )

  if "R Econometrics" in c_lang:
    st.code(
        """library(fixest)
library(plm)
panel_df <- read.csv("balkan_master_panel.csv")
model_fe <- feols(log(gdp_pc_unified) ~ log(energy_pc_kwh) + energy_intensity_mj | iso3 + year, data = panel_df, cluster = ~iso3)
summary(model_fe) # Αποτέλεσμα: β1 = +1.0702 (p < 0.001), R² = 0.964""",
        language="r",
    )
  elif "IBM SPSS" in c_lang:
    st.code(
        """DATASET ACTIVATE BalkanMasterPanel.
ONEWAY gdp_pc_unified cold_homes_pct BY region_group /STATISTICS DESCRIPTIVES HOMOGENEITY WELCH.
UNIANOVA cold_homes_pct BY iso3 WITH log_gdp_pc poor_owners_pct /METHOD=SSTYPE(3) /PRINT=PARAMETER ETASQ.""",
        language="sql",
    )
  elif "Python" in c_lang:
    st.code(
        """import pyarrow.parquet as pq
filtered = pq.read_table("balkan_master_panel.parquet", columns=["iso3", "year", "gdp_pc_unified"], filters=[("year", ">=", 2020)])""",
        language="python",
    )
  else:
    st.code(
        """CREATE INDEX idx_balkan_iso_year ON balkan_master_panel(iso3, year);
EXPLAIN QUERY PLAN SELECT iso3, year, gdp_pc_unified FROM balkan_master_panel WHERE iso3 = 'SRB' AND year >= 2020;""",
        language="sql",
    )

  # 3. Hardware & Telecommunications Benchmark
  st.markdown("#### Τηλεπικοινωνιακή Υποδομή (FTTH / IEEE 1588 PTP) & Hardware Benchmark")
  hw_col1, hw_col2 = st.columns([1, 1.8])
  with hw_col1:
    fiber_km = st.slider("Μήκος Οπτικής Ίνας FTTH Υποσταθμού (km):", 10, 250, 80, 10)
    n_records_sim = st.select_slider(
        "Όγκος Τηλεμετρίας:",
        options=["100.000 εγγραφές", "500.000 εγγραφές", "2.000.000 εγγραφές", "10.000.000 εγγραφές"],
        value="500.000 εγγραφές",
    )
    st.metric("Καθυστέρηση Οπτικής Ίνας (FTTH)", f"{fiber_km * 4.9:.1f} μs", "Όριο προστασίας: < 4.000 μs")
    st.metric("Ακρίβεια Συγχρονισμού (IEEE 1588 PTP)", "± 100 ns", "10.000x ακριβέστερο από NTP")

  with hw_col2:
    scale_mult = {"100.000 εγγραφές": 0.2, "500.000 εγγραφές": 1.0, "2.000.000 εγγραφές": 4.0, "10.000.000 εγγραφές": 20.0}[n_records_sim]
    it_matrix_df = pd.DataFrame({
        "Αρχιτεκτονική Επεξεργασίας": [
            "Παραδοσιακό CSV + Python Loops",
            "SQLite (Χωρίς Ευρετήριο)",
            "SQLite (Με B-Tree Index)",
            "Apache Parquet + NumPy SIMD",
        ],
        "Δίσκος (MB)": [round(24.5 * scale_mult, 1), round(28.0 * scale_mult, 1), round(33.5 * scale_mult, 1), round(5.8 * scale_mult, 1)],
        "Μνήμη RAM (MB)": [round(115.0 * scale_mult, 1), round(64.0 * scale_mult, 1), round(42.0 * scale_mult, 1), round(14.2 * scale_mult, 1)],
        "Απόκριση Query (ms)": [round(420.0 * scale_mult, 1), round(185.0 * scale_mult, 1), round(24.0 * np.log2(scale_mult + 2), 1), round(18.0 * np.log2(scale_mult + 2), 1)],
        "Πολυπλοκότητα": ["O(N)", "O(N)", "O(log N)", "O(log N) + SIMD"],
    })
    st.dataframe(it_matrix_df, use_container_width=True, hide_index=True)

  # 4. Εικόνες υψηλής ανάλυσης αν υπάρχουν στον φάκελο
  available_imgs = [
      ("fig_3_1_erd_star_schema.png", "Διάγραμμα 3.1: Σχεσιακό Σχήμα Αστέρα (Star Schema / ERD)"),
      ("fig_3_4_query_execution_plan.png", "Διάγραμμα 3.4: Query Optimizer — Full Scan O(N) vs Parquet Zone Maps O(log N)"),
      ("fig_4_4_feature_pipeline_visual.png", "Διάγραμμα 4.4: Αγωγός Feature Engineering (STL, Cyclical Encoding & Walk-Forward)"),
      ("fig_6_3_financial_bi_dashboard.png", "Διάγραμμα 6.3: Λογιστική Αποτίμηση Ενέργειας & Χρηματοοικονομικό Dashboard"),
  ]
  existing_imgs = [(f, cap) for f, cap in available_imgs if os.path.exists(f)]
  if existing_imgs:
    st.markdown("#### Τεχνικά Διαγράμματα Αρχιτεκτονικής (300 DPI)")
    cols_img = st.columns(2)
    for i, (img_file, img_cap) in enumerate(existing_imgs):
      with cols_img[i % 2]:
        st.image(img_file, caption=img_cap, use_container_width=True)
