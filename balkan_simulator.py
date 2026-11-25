import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.pipeline import make_pipeline

# Ρύθμιση σελίδας
st.set_page_config(
    page_title="Balkan Energy & Economic Convergence Simulator",
    page_icon="⚡",
    layout="wide"
)

# 1. Δεδομένα των 11 Βαλκανικών Χωρών (Μέσοι Όροι Διπλωματικής 2018-2023)
@st.cache_data
def load_balkan_data():
    data = [
        ("Σλοβενία", "SVN", "EU Balkans", 27878, 36183, 30.3, 2.7, 59.2, 3.2, 6.1),
        ("Ελλάδα", "GRC", "EU Balkans", 20354, 29256, 20.5, 18.8, 68.4, 2.8, 7.2),
        ("Κροατία", "HRV", "EU Balkans", 17326, 23611, 8.7, 6.5, 88.8, 3.1, 9.9),
        ("Ρουμανία", "ROU", "EU Balkans", 14588, 19162, 20.5, 11.1, 94.3, 3.3, 12.2),
        ("Βουλγαρία", "BGR", "EU Balkans", 11880, 30765, 47.7, 26.4, 80.7, 4.8, 9.3),
        ("Μαυροβούνιο", "MNE", "Non-EU Balkans", 9536, 19513, 57.1, 13.3, 87.2, 3.9, 15.9),
        ("Σερβία", "SRB", "Non-EU Balkans", 9233, 27036, 64.4, 9.9, 84.1, 4.5, 13.6),
        ("Βόρεια Μακεδονία", "MKD", "Non-EU Balkans", 7324, 14451, 47.7, 29.5, 83.9, 4.2, 13.5),
        ("Βοσνία-Ερζεγοβίνη", "BIH", "Non-EU Balkans", 6982, 21790, 73.5, 22.0, 88.0, 5.1, 10.1),
        ("Αλβανία", "ALB", "Non-EU Balkans", 6339, 13276, 12.8, 35.4, 96.2, 2.7, 15.7),
        ("Κόσοβο", "XKX", "Non-EU Balkans", 4939, 16275, 70.2, 40.2, 95.7, 4.6, 17.8)
    ]
    cols = [
        "Χώρα", "ISO3", "Ομάδα", "GDP_pc", "Energy_kWh",
        "Coal_Share", "Cold_Homes", "Poor_Owners", "Energy_Intensity", "Grid_Loss"
    ]
    return pd.DataFrame(data, columns=cols)

df = load_balkan_data()

# Εκπαίδευση Μοντέλων της Εργασίας (Log-Ridge & Random Forest)
features = ["Energy_kWh", "Coal_Share", "Cold_Homes", "Poor_Owners", "Energy_Intensity"]
X = df[features]
y = df["GDP_pc"]

pipe_log_ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.5))
pipe_log_ridge.fit(X, np.log(y))

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X, y)

# Επικεφαλίδα Εφαρμογής
st.title("⚡ Διαδραστικό Σύστημα Ανάλυσης Ενεργειακής Ανισότητας & Σύγκλισης στα Βαλκάνια")
st.markdown(
    "**Μεταπτυχιακή Διπλωματική Εργασία** | Αρχιτεκτονική Data Lakehouse, Οικονομετρία Fixed Effects & Μηχανική Μάθηση (2000–2025)"
)

tab1, tab2, tab3 = st.tabs([
    "🎛️ Προσομοιωτής Πολιτικής (What-If)",
    "🧠 Ερμηνευσιμότητα AI (SHAP & RF)",
    "📈 Αγορά SEEPEX & Isolation Forest"
])

# ==============================================================================
# TAB 1: ΠΡΟΣΟΜΟΙΩΤΗΣ ΠΟΛΙΤΙΚΗΣ (ΒΑΣΕΙ TWO-WAY FIXED EFFECTS & LOG-RIDGE)
# ==============================================================================
with tab1:
    st.subheader("Προσομοίωση Παρεμβάσεων Ενεργειακής & Στεγαστικής Πολιτικής")
    st.info(
        "Επιλέξτε μια βαλκανική χώρα και ρυθμίστε τους μοχλούς πολιτικής. Οι προβλέψεις βασίζονται "
        "στις εκτιμημένες ελαστικότητες Two-Way Fixed Effects (β₁ = 1,0702, β₂ = −0,8062, δ₁ = −18,4912) "
        "και στο μοντέλο γενίκευσης Log-Ridge (LOOCV R² = 89,7%)."
    )

    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
selected_country = st.selectbox("Επιλογή Χώρας:", df["Χώρα"].tolist(), index=9)
