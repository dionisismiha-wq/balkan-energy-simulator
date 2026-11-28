import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(
    page_title="Balkan Energy Lakehouse & AI Platform",
    page_icon="⚡",
    layout="wide",
)


# ==============================================================================
# 1. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ & ΔΗΜΙΟΥΡΓΙΑ ΒΑΣΗΣ SQLITE ΣΤΗ ΜΝΗΜΗ
# ==============================================================================
@st.cache_data
def load_balkan_data():
  data = [
      (
          "Σλοβενία",
          "SVN",
          "EU Balkans",
          27878,
          36183,
          30.3,
          2.7,
          59.2,
          3.2,
          6.1,
          32.5,
      ),
      (
          "Ελλάδα",
          "GRC",
          "EU Balkans",
          20354,
          29256,
          20.5,
          18.8,
          68.4,
          2.8,
          7.2,
          38.1,
      ),
      (
          "Κροατία",
          "HRV",
          "EU Balkans",
          17326,
          23611,
          8.7,
          6.5,
          88.8,
          3.1,
          9.9,
          64.2,
      ),
      (
          "Ρουμανία",
          "ROU",
          "EU Balkans",
          14588,
          19162,
          20.5,
          11.1,
          94.3,
          3.3,
          12.2,
          44.0,
      ),
      (
          "Βουλγαρία",
          "BGR",
          "EU Balkans",
          11880,
          30765,
          47.7,
          26.4,
          80.7,
          4.8,
          9.3,
          23.4,
      ),
      (
          "Μαυροβούνιο",
          "MNE",
          "Non-EU Balkans",
          9536,
          19513,
          57.1,
          13.3,
          87.2,
          3.9,
          15.9,
          52.0,
      ),
      (
          "Σερβία",
          "SRB",
          "Non-EU Balkans",
          9233,
          27036,
          64.4,
          9.9,
          84.1,
          4.5,
          13.6,
          31.8,
      ),
      (
          "Βόρεια Μακεδονία",
          "MKD",
          "Non-EU Balkans",
          7324,
          14451,
          47.7,
          29.5,
          83.9,
          4.2,
          13.5,
          28.5,
      ),
      (
          "Βοσνία-Ερζεγοβίνη",
          "BIH",
          "Non-EU Balkans",
          6982,
          21790,
          73.5,
          22.0,
          88.0,
          5.1,
          10.1,
          36.4,
      ),
      (
          "Αλβανία",
          "ALB",
          "Non-EU Balkans",
          6339,
          13276,
          12.8,
          35.4,
          96.2,
          2.7,
          15.7,
          99.4,
      ),
      (
          "Κόσοβο",
          "XKX",
          "Non-EU Balkans",
          4939,
          16275,
          70.2,
          40.2,
          95.7,
          4.6,
          17.8,
          6.5,
      ),
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

# Sidebar με γενικές πληροφορίες & φίλτρα
with st.sidebar:
  st.header("⚙️ Πίνακας Ελέγχου Συστήματος")
  st.markdown("**Αρχιτεκτονική:** Data Lakehouse (Parquet / SQL / Python)")
  st.markdown("**Κάλυψη:** 11 Βαλκανικές Χώρες (2000–2025)")
  region_filter = st.multiselect(
      "Φίλτρο Θεσμικής Ομάδας:",
      options=["EU Balkans", "Non-EU Balkans"],
      default=["EU Balkans", "Non-EU Balkans"],
  )
  st.divider()
  st.caption(
      "Μεταπτυχιακή Διπλωματική Εργασία | Οικονομική Σύγκλιση, Ενεργειακή"
      " Φτώχεια & Αγορά SEEPEX"
  )

df_filtered = (
    df[df["Ομάδα"].isin(region_filter)] if len(region_filter) > 0 else df
)

# Επικεφαλίδα
st.title(
    "⚡ Διαδραστική Πλατφόρμα Ανάλυσης Ενεργειακής Ανισότητας & Σύγκλισης στα"
    " Βαλκάνια"
)
st.markdown(
    "Ενοποιημένο περιβάλλον **Data Lakehouse, Οικονομετρίας (Two-Way Fixed"
    " Effects), Μηχανικής Μάθησης (SHAP / Isolation Forest) και Live SQL**"
)

# KPIs στην κορυφή
k1, k2, k3, k4 = st.columns(4)
k1.metric("Επιλεγμένες Χώρες", f"{len(df_filtered)} / 11")
k2.metric(
    "Μέσο Κατά Κεφ. ΑΕΠ",
    f"{df_filtered['GDP_pc'].mean():,.0f} USD",
    "Χάσμα 2,42x EU/Non-EU",
)
k3.metric(
    "Μέση Ενεργειακή Φτώχεια",
    f"{df_filtered['Cold_Homes'].mean():.1f}%",
    "Μ.Ο. Ε.Ε.-27: ~9,5%",
    delta_color="inverse",
)
k4.metric(
    "Φτωχοί Ιδιοκτήτες Κατοικίας",
    f"{df_filtered['Poor_Owners'].mean():.1f}%",
    "Κύριος δομικός παράγοντας",
    delta_color="inverse",
)

# 5 Πλήρεις Διαδραστικές Καρτέλες
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Μακροοικονομική & Ενεργειακή Σύγκλιση",
    "2. Ενεργειακή Φτώχεια & Ιδιοκατοίκηση",
    "3. Χονδρεμπορική Αγορά SEEPEX & Ανωμαλίες",
    "4. Πρόβλεψη 5 Ημερών (XGBoost / MLP / SHAP)",
    "5. Προσομοιωτής Πολιτικής & Smart Grids",
    "6. Λογιστική & Χρηματοοικονομική Ανάλυση (Financial BI)",
])

# ==============================================================================
# TAB 1: ΔΙΑΔΡΑΣΤΙΚΟΣ ΧΑΡΤΗΣ & ΣΥΓΚΡΙΣΗ ΧΩΡΩΝ
# ==============================================================================
with tab1:
  st.subheader("Διαδραστική Χαρτογράφηση Ενεργειακών & Οικονομικών Ανισοτήτων")
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
    )
    fig_map.update_geos(
        center=dict(lat=43.5, lon=21.5),
        projection_scale=5.2,
        visible=True,
        showcountries=True,
    )
    fig_map.update_layout(height=450, margin=dict(l=0, r=0, t=40, b=0))
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
        color_discrete_map={
            "EU Balkans": "#1d4ed8",
            "Non-EU Balkans": "#dc2626",
        },
        title=f"Κατάταξη Χωρών: {chosen_label}",
    )
    fig_rank.update_layout(height=450)
    st.plotly_chart(fig_rank, use_container_width=True)

# ==============================================================================
# TAB 2: ΠΡΟΣΟΜΟΙΩΤΗΣ ΠΟΛΙΤΙΚΗΣ (WHAT-IF SIMULATOR)
# ==============================================================================
with tab2:
  st.subheader("Προσομοιωτής Πολιτικής & Σύγκλισης (Policy What-If Simulator)")
  col_ctrl, col_res = st.columns([1, 2])

  with col_ctrl:
    selected_country = st.selectbox(
        "Επιλογή Χώρας Στόχου:", df["Χώρα"].tolist(), index=9
    )
    row = df[df["Χώρα"] == selected_country].iloc[0]
    st.markdown(f"**Θεσμική Ομάδα:** `{row['Ομάδα']}` (`{row['ISO3']}`)")

    energy_boost = st.slider(
        "1. Αύξηση Ενεργειακής Διαθεσιμότητας (%)", 0, 60, 20, 5
    )
    intensity_cut = st.slider(
        "2. Μείωση Ενεργειακής Έντασης (MJ/$)", 0.0, 2.0, 0.6, 0.1
    )
    retrofit_share = st.slider(
        "3. Επιδότηση Θερμομόνωσης Φτωχών Ιδιοκτητών (%)", 0, 50, 25, 5
    )
    carbon_price = st.slider(
        "4. Τιμή Άνθρακα CBAM (€/τόνο CO₂)", 0, 120, 75, 5
    )

  # Οικονομετρικοί υπολογισμοί Fixed Effects (β1=1.0702, β2=-0.8062, δ1=-18.4912)
  base_gdp = row["GDP_pc"]
  base_cold = row["Cold_Homes"]
  delta_log_gdp = (1.0702 * np.log(1 + energy_boost / 100.0)) + (
      -0.8062 * (-intensity_cut)
  )
  sim_gdp_fe = base_gdp * np.exp(delta_log_gdp)

  # Επιβάρυνση CBAM αν το μερίδιο άνθρακα είναι υψηλό
  cbam_penalty_pct = (row["Coal_Share"] / 100.0) * (carbon_price / 100.0) * 4.5
  sim_gdp_net = sim_gdp_fe * (1 - cbam_penalty_pct / 100.0)

  # Νέα Ενεργειακή Φτώχεια
  sim_cold = max(
      1.5, base_cold + (-18.4912 * delta_log_gdp) + (-0.35 * retrofit_share)
  )

  with col_res:
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Προβλεπόμενο ΑΕΠ (καθαρό CBAM)",
        f"{sim_gdp_net:,.0f} USD",
        f"{sim_gdp_net - base_gdp:+,.0f} USD",
    )
    m2.metric(
        "Επιβάρυνση Άνθρακα CBAM",
        f"-{cbam_penalty_pct:.2f}%",
        f"Μερίδιο Άνθρακα: {row['Coal_Share']}%",
        delta_color="inverse",
    )
    m3.metric(
        "Νέα Ενεργειακή Φτώχεια",
        f"{sim_cold:.1f}%",
        f"{sim_cold - base_cold:.1f} π.μ.",
        delta_color="inverse",
    )

    comp_df = pd.DataFrame({
        "Σενάριο": [
            f"{selected_country} (Τρέχον)",
            f"{selected_country} (Μετά τις Παρεμβάσεις)",
            "Μέσος Όρος EU Balkans",
        ],
        "Κατά Κεφαλήν ΑΕΠ (USD)": [base_gdp, sim_gdp_net, 18405],
        "Ενεργειακή Φτώχεια (%)": [base_cold, sim_cold, 13.1],
    })
    fig_sim = px.bar(
        comp_df,
        x="Σενάριο",
        y="Κατά Κεφαλήν ΑΕΠ (USD)",
        color="Σενάριο",
        text_auto=",.0f",
        color_discrete_sequence=["#dc2626", "#16a34a", "#2563eb"],
        title=(
            f"Τροχιά Σύγκλισης: {selected_country} (συνυπολογίζοντας CBAM"
            f" {carbon_price} €/t)"
        ),
    )
    fig_sim.update_layout(showlegend=False, height=360)
    st.plotly_chart(fig_sim, use_container_width=True)

# ==============================================================================
# TAB 3: ΕΡΓΑΣΤΗΡΙΟ AI & ΕΡΜΗΝΕΥΣΙΜΟΤΗΤΑ SHAP
# ==============================================================================
with tab3:
  st.subheader("Εργαστήριο Μηχανικής Μάθησης (Hyperparameter Tuning & SHAP)")
  c_param, c_shap = st.columns([1, 2])

  with c_param:
    n_trees = st.slider("Πλήθος Δέντρων Random Forest:", 50, 500, 200, 50)
    ridge_alpha = st.slider(
        "Παράμετρος Ομαλοποίησης Ridge (α):", 0.1, 10.0, 1.5, 0.1
    )
    country_shap = st.selectbox(
        "Χώρα για Ερμηνευσιμότητα SHAP:", df["Χώρα"].tolist(), index=10
    )

  features = [
      "Energy_kWh",
      "Coal_Share",
      "Cold_Homes",
      "Poor_Owners",
      "Energy_Intensity",
  ]
  X = df[features]
  y = df["GDP_pc"]

  rf_model = RandomForestRegressor(n_estimators=n_trees, random_state=42).fit(
      X, y
  )
  ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha)).fit(
      X, np.log(y)
  )

  idx = df[df["Χώρα"] == country_shap].index[0]
  pred_rf = rf_model.predict(X.iloc[[idx]])[0]
  pred_ridge = np.exp(ridge_model.predict(X.iloc[[idx]])[0])

  with c_param:
    st.metric("Πραγματικό ΑΕΠ", f"{y.iloc[idx]:,.0f} USD")
    st.metric(
        "Πρόβλεψη Random Forest",
        f"{pred_rf:,.0f} USD",
        f"Σφάλμα: {pred_rf - y.iloc[idx]:+,.0f}",
    )
    st.metric(
        "Πρόβλεψη Log-Ridge",
        f"{pred_ridge:,.0f} USD",
        f"Σφάλμα: {pred_ridge - y.iloc[idx]:+,.0f}",
    )

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
      diff = (
          rf_model.predict(X.iloc[[idx]])[0]
          - rf_model.predict(X_temp.iloc[[idx]])[0]
      )
      contributions.append(
          {"Μεταβλητή": greek_labels[feat], "Συνεισφορά SHAP (USD)": diff}
      )

    shap_df = pd.DataFrame(contributions).sort_values("Συνεισφορά SHAP (USD)")
    fig_shap = px.bar(
        shap_df,
        x="Συνεισφορά SHAP (USD)",
        y="Μεταβλητή",
        orientation="h",
        color="Συνεισφορά SHAP (USD)",
        color_continuous_scale=["#dc2626", "#f1f5f9", "#16a34a"],
        title=f"Αποσύνθεση Πρόβλεψης SHAP για: {country_shap}",
    )
    fig_shap.update_layout(height=400)
    st.plotly_chart(fig_shap, use_container_width=True)

# ==============================================================================
# TAB 4: ΑΓΟΡΑ SEEPEX & ISOLATION FOREST ANOMALY DETECTION
# ==============================================================================
with tab4:
  st.subheader("Διαδραστικός Ανιχνευτής Ανωμαλιών SEEPEX (Isolation Forest)")
  col_s1, col_s2 = st.columns([1, 2.5])

  with col_s1:
    contam = st.slider(
        "Ευαισθησία Isolation Forest (Contamination %):", 1.0, 10.0, 3.5, 0.5
    )
    crisis_shock = st.slider(
        "Ένταση Βραδινών Αιχμών (18:00–21:00):", 1.0, 2.5, 1.5, 0.1
    )

  np.random.seed(42)
  hours = np.arange(168)
  hod = hours % 24
  prices = (
      85
      + 25 * np.sin((hod - 7) * np.pi / 12)
      + np.random.normal(0, 8, 168)
  )
  spike_idx = [19, 20, 43, 68, 91, 92, 116, 140, 163]
  prices[spike_idx] = prices[spike_idx] * (2.2 * crisis_shock)

  iso = IsolationForest(contamination=contam / 100.0, random_state=42)
  preds = iso.fit_predict(prices.reshape(-1, 1))
  is_anomaly = preds == -1

  with col_s1:
    st.metric(
        "Εντοπισμένα Price Spikes",
        f"{is_anomaly.sum()} ώρες",
        f"Από {len(prices)} ώρες εβδομάδας",
    )
    st.metric(
        "Μέση Τιμή Ανωμαλιών",
        f"{prices[is_anomaly].mean():.2f} €/MWh",
        delta_color="inverse",
    )
    st.metric("Μέγιστη Ωριαία Αιχμή", f"{prices.max():.2f} €/MWh")

  with col_s2:
    df_seepex = pd.DataFrame({
        "Ώρα": hours,
        "Τιμή (€/MWh)": prices,
        "Κατάσταση": np.where(
            is_anomaly, "Price Spike (Ανωμαλία)", "Κανονική Λειτουργία"
        ),
    })
    fig_sp = px.scatter(
        df_seepex,
        x="Ώρα",
        y="Τιμή (€/MWh)",
        color="Κατάσταση",
        color_discrete_map={
            "Κανονική Λειτουργία": "#2563eb",
            "Price Spike (Ανωμαλία)": "#dc2626",
        },
        title=(
            f"Isolation Forest (contamination = {contam}%) — Ανίχνευση"
            " Ακραίων Τιμών SEEPEX"
        ),
    )
    fig_sp.add_trace(
        go.Scatter(
            x=df_seepex["Ώρα"],
            y=df_seepex["Τιμή (€/MWh)"],
            mode="lines",
            line=dict(color="gray", width=1),
            showlegend=False,
        )
    )
    fig_sp.update_layout(height=400)
    st.plotly_chart(fig_sp, use_container_width=True)

# ==============================================================================
# TAB 5: LIVE SQL ENGINE & ΕΞΑΓΩΓΗ ΔΕΔΟΜΕΝΩΝ
# ==============================================================================
with tab5:
  st.subheader("Εκτέλεση Ερωτημάτων SQL σε Πραγματικό Χρόνο (In-Memory SQLite)")
  conn = sqlite3.connect(":memory:")
  df.to_sql("balkan_master_panel", conn, index=False, if_exists="replace")

  preset_queries = {
      "1. Σύγκριση Μέσων Όρων EU vs Non-EU Balkans": (
          "SELECT Ομάδα, COUNT(*) AS Χώρες, ROUND(AVG(GDP_pc), 1) AS"
          " Μέσο_ΑΕΠ, ROUND(AVG(Cold_Homes), 2) AS Ενεργειακή_Φτώχεια,"
          " ROUND(AVG(Grid_Loss), 2) AS Απώλειες_Δικτύου FROM"
          " balkan_master_panel GROUP BY Ομάδα;"
      ),
      "2. Χώρες Υψηλού Κινδύνου CBAM & Φτώχειας": (
          "SELECT ISO3, Χώρα, Coal_Share AS Άνθρακας_Pct, Cold_Homes AS"
          " Αδυναμία_Θέρμανσης, Poor_Owners AS Φτωχοί_Ιδιοκτήτες FROM"
          " balkan_master_panel WHERE Cold_Homes > 18 ORDER BY Cold_Homes DESC;"
      ),
      "3. Όλος ο Πίνακας Δεδομένων (balkan_master_panel)": (
          "SELECT * FROM balkan_master_panel;"
      ),
  }

  selected_preset = st.selectbox(
      "Επιλέξτε έτοιμο ερώτημα SQL ή τροποποιήστε το παρακάτω:",
      list(preset_queries.keys()),
  )
  sql_input = st.text_area(
      "Κώδικας SQL:", value=preset_queries[selected_preset], height=100
  )

  if st.button("▶️ Εκτέλεση SQL Query"):
    try:
      res_sql = pd.read_sql_query(sql_input, conn)
      st.dataframe(res_sql, use_container_width=True)
      csv_bytes = res_sql.to_csv(index=False).encode("utf-8-sig")
      st.download_button(
          "📥 Κατέβασμα Αποτελεσμάτων (CSV)",
          data=csv_bytes,
          file_name="balkan_sql_export.csv",
          mime="text/csv",
      )
    except Exception as e:
      st.error(f"Σφάλμα σύνταξης SQL: {e}")

# ==============================================================================
# TAB 6: FINANCIAL & CARBON ACCOUNTING BI (ΛΟΓΙΣΤΙΚΗ & ΧΡΗΜΑΤΟΟΙΚΟΝΟΜΙΚΑ)
# ==============================================================================
with tab6:
  st.subheader(
      "Λογιστική Αποτίμηση Κόστους Άνθρακα (CBAM), Απωλειών Δικτύου & Αξιολόγηση"
      " CAPEX"
  )
  st.markdown(
      "Μετασχηματισμός τεχνικών ενεργειακών δεικτών σε λογιστικά μεγέθη "
      "**Κατάστασης Αποτελεσμάτων Χρήσης (P&L)**, **Περιθωρίου EBITDA** και "
      "**Καθαρής Παρούσας Αξίας (NPV / Payback Period)**."
  )

  # --- Πλαϊνές / Επάνω Παράμετροι Λογιστικής Προσομοίωσης ---
  col_p1, col_p2, col_p3, col_p4 = st.columns(4)
  with col_p1:
    selected_country_fin = st.selectbox(
        "Επιλογή Χώρας / Αγοράς:",
        [
            "Σερβία (SRB)",
            "Βοσνία-Ερζεγοβίνη (BIH)",
            "Κόσοβο (XKX)",
            "Βόρεια Μακεδονία (MKD)",
            "Αλβανία (ALB)",
        ],
        index=0,
        key="fin_country",
    )
  with col_p2:
    carbon_price_eur = st.slider(
        "Τιμή Δικαιωμάτων CO₂ CBAM (€/t):",
        min_value=40,
        max_value=130,
        value=85,
        step=5,
        key="fin_cbam",
    )
  with col_p3:
    capex_invest_meur = st.slider(
        "Επένδυση CAPEX Smart Grids / FTTH (Εκ. €):",
        min_value=40,
        max_value=250,
        value=120,
        step=10,
        key="fin_capex",
    )
  with col_p4:
    wacc_rate = st.slider(
        "Προεξοφλητικό Επιτόκιο WACC (%):",
        min_value=4.0,
        max_value=12.0,
        value=7.5,
        step=0.5,
        key="fin_wacc",
    )

  # --- Βασικά Δεδομένα Αναφοράς ανά Χώρα (Εκπομπές CO2 в Mt, Απώλειες σε GWh, Τιμή €/MWh, Βασικά Έσοδα Εκ. €) ---
  country_fin_params = {
      "Σερβία (SRB)": {
          "co2_mt": 4.82,
          "loss_gwh": 2050,
          "tariff": 95.0,
          "base_rev": 2450.0,
          "base_ebitda_pct": 24.5,
      },
      "Βοσνία-Ερζεγοβίνη (BIH)": {
          "co2_mt": 3.12,
          "loss_gwh": 1160,
          "tariff": 94.8,
          "base_rev": 1520.0,
          "base_ebitda_pct": 23.0,
      },
      "Κόσοβο (XKX)": {
          "co2_mt": 1.39,
          "loss_gwh": 930,
          "tariff": 94.6,
          "base_rev": 680.0,
          "base_ebitda_pct": 19.5,
      },
      "Βόρεια Μακεδονία (MKD)": {
          "co2_mt": 1.12,
          "loss_gwh": 760,
          "tariff": 94.7,
          "base_rev": 740.0,
          "base_ebitda_pct": 21.0,
      },
      "Αλβανία (ALB)": {
          "co2_mt": 0.05,
          "loss_gwh": 970,
          "tariff": 94.8,
          "base_rev": 890.0,
          "base_ebitda_pct": 28.0,
      },
  }

  p = country_fin_params[selected_country_fin]

  # --- Λογιστικοί Υπολογισμοί P&L, EBITDA & NPV ---
  cbam_liability_meur = p["co2_mt"] * carbon_price_eur
  grid_loss_cost_meur = (p["loss_gwh"] * p["tariff"]) / 1000.0  # σε Εκατ. €

  base_ebitda_meur = p["base_rev"] * (p["base_ebitda_pct"] / 100.0)
  post_cbam_ebitda_meur = max(base_ebitda_meur - cbam_liability_meur, 0.0)
  post_cbam_ebitda_pct = (post_cbam_ebitda_meur / p["base_rev"]) * 100.0

  # Εξοικονόμηση από μείωση απωλειών δικτύου κατά 45% μέσω Smart Grids + FTTH/PTP
  annual_cash_inflow_meur = grid_loss_cost_meur * 0.45
  payback_years = (
      capex_invest_meur / annual_cash_inflow_meur
      if annual_cash_inflow_meur > 0
      else 99.9
  )

  # Υπολογισμός 8ετούς Καθαρής Παρούσας Αξίας (NPV)
  r = wacc_rate / 100.0
  years_horizon = np.arange(1, 9)
  discounted_cf = [annual_cash_inflow_meur / ((1 + r) ** t) for t in years_horizon]
  npv_8yr_meur = sum(discounted_cf) - capex_invest_meur

  # --- Εμφάνιση KPI Cards ---
  st.markdown("---")
  k1, k2, k3, k4 = st.columns(4)
  k1.metric(
      label="Ετήσια Επιβάρυνση CBAM (COGS)",
      value=f"{cbam_liability_meur:.1f} Εκ. €",
      delta=f"-{(cbam_liability_meur/p['base_rev'])*100:.1f}% επί Εσόδων",
      delta_color="inverse",
  )
  k2.metric(
      label="Λογιστική Ζημία Απωλειών Δικτύου",
      value=f"{grid_loss_cost_meur:.1f} Εκ. €",
      delta=f"Ανάκτηση: +{annual_cash_inflow_meur:.1f} Εκ. €/έτος",
  )
  k3.metric(
      label="Περιθώριο EBITDA (Μετά CBAM)",
      value=f"{post_cbam_ebitda_pct:.1f}%",
      delta=f"{post_cbam_ebitda_pct - p['base_ebitda_pct']:.1f} ποσ. μονάδες",
      delta_color="normal",
  )
  k4.metric(
      label="Απόσβεση (Payback) & NPV 8ετίας",
      value=f"{payback_years:.1f} Έτη",
      delta=f"NPV: {npv_8yr_meur:+.1f} Εκ. €",
  )

  st.markdown("---")

  # --- Γραφήματα Χρηματοοικονομικής Ανάλυσης (2 Στήλες) ---
  import plotly.graph_objects as go

  g_col1, g_col2 = st.columns(2)

  with g_col1:
    # Γράφημα 1: Σωρευτικές Ταμειακές Ροές (Cumulative Cash Flow & Payback Point)
    yrs = list(range(0, 9))
    cum_cf = [-capex_invest_meur]
    for y in range(1, 9):
      cum_cf.append(cum_cf[-1] + annual_cash_inflow_meur)

    bar_colors = ["#dc2626" if val < 0 else "#16a34a" for val in cum_cf]
    fig_cf = go.Figure(
        data=[
            go.Bar(
                x=[f"Έτος {y}" for y in yrs],
                y=cum_cf,
                marker_color=bar_colors,
                text=[f"{val:+.1f}M" for val in cum_cf],
                textposition="outside",
            )
        ]
    )
    fig_cf.add_hline(y=0, line_width=1.5, line_color="#0f172a")
    fig_cf.update_layout(
        title=(
            f"Σωρευτικές Ταμειακές Ροές CAPEX (Payback: {payback_years:.1f}"
            " Έτη)"
        ),
        yaxis_title="Εκατομμύρια Ευρώ (Εκ. €)",
        height=390,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    st.plotly_chart(fig_cf, use_container_width=True)

  with g_col2:
    # Γράφημα 2: Σύγκριση Περιθωρίου EBITDA ανά Στρατηγικό Σενάριο
    modernized_ebitda_pct = (
        (base_ebitda_meur + annual_cash_inflow_meur - (cbam_liability_meur * 0.25))
        / p["base_rev"]
    ) * 100.0
    fig_ebitda = go.Figure(
        data=[
            go.Bar(
                x=[
                    "1. Βασικό Σενάριο",
                    "2. Πλήρης CBAM (Χωρίς Έργα)",
                    "3. Εκσυγχρονισμός (ΑΠΕ + Smart Grids)",
                ],
                y=[
                    p["base_ebitda_pct"],
                    post_cbam_ebitda_pct,
                    modernized_ebitda_pct,
                ],
                marker_color=["#0284c7", "#dc2626", "#16a34a"],
                text=[
                    f"{p['base_ebitda_pct']:.1f}%",
                    f"{post_cbam_ebitda_pct:.1f}%",
                    f"{modernized_ebitda_pct:.1f}%",
                ],
                textposition="outside",
            )
        ]
    )
    fig_ebitda.update_layout(
        title="Επίδραση στο Λειτουργικό Περιθώριο Κερδοφορίας (EBITDA %)",
        yaxis_title="Περιθώριο EBITDA (%)",
        yaxis_range=[0, max(modernized_ebitda_pct + 8, 38)],
        height=390,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    st.plotly_chart(fig_ebitda, use_container_width=True)

  # --- Συνοπτικός Λογιστικός Πίνακας P&L (Statement of Profit or Loss) ---
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
          (
              f"{base_ebitda_meur + annual_cash_inflow_meur - (cbam_liability_meur * 0.25):,.1f} €"
          ),
          f"{modernized_ebitda_pct:.1f}%",
      ],
  })
  st.dataframe(pnl_df, use_container_width=True, hide_index=True)
