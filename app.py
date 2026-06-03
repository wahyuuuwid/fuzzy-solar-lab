"""
app.py — Web App Streamlit
Fuzzy Logic PLTS: Estimasi Output Daya DC Pembangkit Listrik Tenaga Surya
Mamdani vs Sugeno From Scratch
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fuzzy PLTS – DKA",
    page_icon="☀️",
    layout="wide",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    .metric-label { font-size: 13px; color: #94a3b8; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #f97316; }
    .metric-unit  { font-size: 12px; color: #64748b; margin-top: 2px; }

    .result-box {
        background: linear-gradient(135deg, #0f3460, #1a1a2e);
        border-left: 4px solid #f97316;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
        color: white;
    }
    .result-method { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .result-value  { font-size: 32px; font-weight: 700; color: #fbbf24; }

    .section-header {
        background: linear-gradient(90deg, #f97316, #ea580c);
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        margin: 16px 0 8px 0;
        display: inline-block;
    }
    .stButton>button {
        background: linear-gradient(135deg, #f97316, #ea580c) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        padding: 10px 32px !important;
        font-size: 15px !important;
        transition: transform 0.1s !important;
    }
    .stButton>button:hover { transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FUZZY LOGIC ENGINE (from scratch — sama persis dengan notebook)
# ═══════════════════════════════════════════════════════════════════════════════

def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left  = np.where(b - a != 0, (x - a) / (b - a), 0.0)
    right = np.where(c - b != 0, (c - x) / (c - b), 0.0)
    return np.clip(np.minimum(left, right), 0, 1)

def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left  = np.where(b - a != 0, (x - a) / (b - a), 1.0)
    right = np.where(d - c != 0, (d - x) / (d - c), 1.0)
    mid   = np.ones_like(x)
    return np.clip(np.minimum(np.minimum(left, right), mid), 0, 1)

# Input MFs
irr_rendah  = lambda x: trapmf(x, 0.0, 0.0, 0.15, 0.45)
irr_sedang  = lambda x: trimf (x, 0.15, 0.50, 0.85)
irr_tinggi  = lambda x: trapmf(x, 0.60, 0.90, 1.25, 1.25)

amb_rendah  = lambda x: trapmf(x, 20, 20, 22, 25)
amb_sedang  = lambda x: trimf (x, 22, 26, 30)
amb_tinggi  = lambda x: trapmf(x, 28, 31, 36, 36)

mod_rendah  = lambda x: trapmf(x, 18, 18, 22, 28)
mod_sedang  = lambda x: trimf (x, 22, 32, 42)
mod_tinggi  = lambda x: trimf (x, 36, 48, 58)
mod_stinggi = lambda x: trapmf(x, 52, 58, 66, 66)

# Output MFs
DC_MAX = 320_000
dc_srendah  = lambda x: trapmf(x,      0,      0,  40_000,  80_000)
dc_rendah   = lambda x: trimf (x, 30_000, 80_000, 140_000)
dc_sedang   = lambda x: trimf (x,100_000,160_000, 220_000)
dc_tinggi   = lambda x: trimf (x,180_000,240_000, 290_000)
dc_stinggi  = lambda x: trapmf(x,260_000,290_000, 320_000, 320_000)

IRR_MF = [irr_rendah, irr_sedang, irr_tinggi]
AMB_MF = [amb_rendah, amb_sedang, amb_tinggi]
MOD_MF = [mod_rendah, mod_sedang, mod_tinggi, mod_stinggi]
OUT_MF = [dc_srendah, dc_rendah, dc_sedang, dc_tinggi, dc_stinggi]

RULES = [
    (0,0,0,0),(0,1,0,0),(0,2,1,0),
    (1,0,0,2),(1,0,1,2),(1,1,1,2),(1,1,2,1),
    (1,2,2,1),(1,2,3,0),
    (2,0,0,4),(2,0,1,4),(2,1,1,3),(2,1,2,3),
    (2,2,2,2),(2,2,3,1),(2,2,3,1),
    (1,0,1,2),(2,1,0,4),(1,2,2,1),(0,1,1,0),
]

SINGLETON = {0:20_000, 1:80_000, 2:160_000, 3:240_000, 4:305_000}

x_out = np.linspace(0, DC_MAX, 2000)

def fuzzify(irr_v, amb_v, mod_v):
    mu_irr = np.array([mf(irr_v) for mf in IRR_MF])
    mu_amb = np.array([mf(amb_v) for mf in AMB_MF])
    mu_mod = np.array([mf(mod_v) for mf in MOD_MF])
    return mu_irr, mu_amb, mu_mod

def mamdani_infer(irr_v, amb_v, mod_v):
    mu_irr, mu_amb, mu_mod = fuzzify(irr_v, amb_v, mod_v)
    aggregated = np.zeros(len(x_out))
    for (r_i, r_a, r_m, r_o) in RULES:
        firing    = min(mu_irr[r_i], mu_amb[r_a], mu_mod[r_m])
        clipped   = np.minimum(firing, OUT_MF[r_o](x_out))
        aggregated = np.maximum(aggregated, clipped)
    denom = np.sum(aggregated)
    if denom == 0:
        return 0.0, aggregated
    return float(np.sum(x_out * aggregated) / denom), aggregated

def sugeno_infer(irr_v, amb_v, mod_v):
    mu_irr, mu_amb, mu_mod = fuzzify(irr_v, amb_v, mod_v)
    total_w, weighted = 0.0, 0.0
    for (r_i, r_a, r_m, r_o) in RULES:
        w = min(mu_irr[r_i], mu_amb[r_a], mu_mod[r_m])
        total_w   += w
        weighted  += w * SINGLETON[r_o]
    if total_w == 0:
        return 0.0
    return float(weighted / total_w)


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 32px; border-radius: 16px; margin-bottom: 24px; text-align: center;">
    <div style="font-size: 48px; margin-bottom: 8px;">☀️</div>
    <h1 style="color: #fbbf24; font-family: Space Grotesk; font-size: 28px; margin: 0;">
        Fuzzy Logic PLTS
    </h1>
    <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 14px;">
        Estimasi Output Daya DC · Mamdani & Sugeno <em>From Scratch</em> · DKA TUBES
    </p>
    <p style="color: #64748b; margin: 4px 0 0 0; font-size: 12px;">
        Kafin Fazlur Rahman · Dzaki Khothir · Wahyu Widodo
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔢 Kalkulator Fuzzy", "📊 Fungsi Keanggotaan", "📋 Rule Base"])

# ─── TAB 1: KALKULATOR ────────────────────────────────────────────────────────
with tab1:
    col_in, col_out = st.columns([1, 1.2], gap="large")

    with col_in:
        st.markdown('<div class="section-header">⚙️ INPUT PARAMETER</div>', unsafe_allow_html=True)

        irr_val = st.slider(
            "☀️ Irradiation (kW/m²)",
            min_value=0.0, max_value=1.25, value=0.75, step=0.01,
            help="Intensitas radiasi matahari"
        )
        amb_val = st.slider(
            "🌡️ Ambient Temperature (°C)",
            min_value=20.0, max_value=36.0, value=28.0, step=0.1,
            help="Suhu lingkungan sekitar panel"
        )
        mod_val = st.slider(
            "🔥 Module Temperature (°C)",
            min_value=18.0, max_value=66.0, value=45.0, step=0.5,
            help="Suhu permukaan modul surya"
        )

        st.markdown("---")
        hitung = st.button("⚡ HITUNG SEKARANG", use_container_width=True)

    with col_out:
        st.markdown('<div class="section-header">📈 HASIL INFERENSI</div>', unsafe_allow_html=True)

        if hitung or True:  # auto-calculate on load
            mamdani_result, aggregated = mamdani_infer(irr_val, amb_val, mod_val)
            sugeno_result = sugeno_infer(irr_val, amb_val, mod_val)

            # Fuzzifikasi display
            mu_irr, mu_amb, mu_mod = fuzzify(irr_val, amb_val, mod_val)
            irr_names = ['Rendah','Sedang','Tinggi']
            amb_names = ['Rendah','Sedang','Tinggi']
            mod_names = ['Rendah','Sedang','Tinggi','Sangat Tinggi']

            # Hasil utama
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-method">🔵 Mamdani (Centroid)</div>
                    <div class="result-value">{mamdani_result/1000:.1f} <span style="font-size:16px; color:#94a3b8">kW</span></div>
                    <div style="color:#64748b; font-size:12px;">{mamdani_result:,.0f} W</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="result-box" style="border-left-color: #ef4444;">
                    <div class="result-method">🔴 Sugeno (Weighted Avg)</div>
                    <div class="result-value" style="color:#fb7185;">{sugeno_result/1000:.1f} <span style="font-size:16px; color:#94a3b8">kW</span></div>
                    <div style="color:#64748b; font-size:12px;">{sugeno_result:,.0f} W</div>
                </div>
                """, unsafe_allow_html=True)

            selisih = abs(mamdani_result - sugeno_result)
            st.markdown(f"""
            <div style="background:#1e293b; border-radius:8px; padding:10px 16px; margin:8px 0; color:#94a3b8; font-size:13px;">
                📐 Selisih Mamdani vs Sugeno: <strong style="color:#fbbf24">{selisih/1000:.2f} kW</strong>
                &nbsp;({selisih/(DC_MAX)*100:.1f}% dari DC_MAX)
            </div>
            """, unsafe_allow_html=True)

            # Fuzzifikasi detail
            st.markdown('<div class="section-header">🔺 Detail Fuzzifikasi</div>', unsafe_allow_html=True)

            df_fuzz = pd.DataFrame({
                "Variabel": [f"Irr – {n}" for n in irr_names] + [f"Amb – {n}" for n in amb_names] + [f"Mod – {n}" for n in mod_names],
                "μ (derajat keanggotaan)": list(mu_irr) + list(mu_amb) + list(mu_mod),
            })
            df_fuzz = df_fuzz[df_fuzz["μ (derajat keanggotaan)"] > 0.0001].reset_index(drop=True)
            df_fuzz["μ (derajat keanggotaan)"] = df_fuzz["μ (derajat keanggotaan)"].apply(lambda v: f"{v:.4f}")
            st.dataframe(df_fuzz, use_container_width=True, hide_index=True)

    # Defuzzifikasi chart Mamdani
    st.markdown('<div class="section-header">📉 Kurva Defuzzifikasi Mamdani</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    ax.fill_between(x_out/1000, aggregated, alpha=0.5, color='royalblue', label='Aggregated MF')
    ax.plot(x_out/1000, aggregated, color='#60a5fa', lw=1.5)
    ax.axvline(mamdani_result/1000, color='#f97316', lw=2.5, linestyle='--', label=f'Centroid = {mamdani_result/1000:.1f} kW')
    ax.axvline(sugeno_result/1000,  color='#ef4444', lw=2,   linestyle=':',  label=f'Sugeno   = {sugeno_result/1000:.1f} kW')
    ax.set_xlabel('DC Power (kW)', color='#94a3b8')
    ax.set_ylabel('μ', color='#94a3b8')
    ax.tick_params(colors='#64748b')
    for spine in ax.spines.values(): spine.set_color('#1e293b')
    ax.legend(facecolor='#1e293b', labelcolor='white', fontsize=10)
    ax.grid(alpha=0.15, color='#334155')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── TAB 2: MEMBERSHIP FUNCTIONS ─────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">📊 Visualisasi Fungsi Keanggotaan</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor='#0f172a')
    colors = ['#60a5fa','#4ade80','#f97316','#f43f5e']

    for ax in axes.flat:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        for sp in ax.spines.values(): sp.set_color('#334155')
        ax.grid(alpha=0.2, color='#334155')
        ax.set_ylim(-0.05, 1.15)

    x_irr = np.linspace(0, 1.25, 500)
    for mf, lb, c in zip(IRR_MF, ['Rendah','Sedang','Tinggi'], colors):
        axes[0,0].plot(x_irr, mf(x_irr), lw=2, label=lb, color=c)
    axes[0,0].set_title('Irradiation (kW/m²)', color='white')
    axes[0,0].legend(facecolor='#0f172a', labelcolor='white')

    x_amb = np.linspace(20, 36, 500)
    for mf, lb, c in zip(AMB_MF, ['Rendah','Sedang','Tinggi'], colors):
        axes[0,1].plot(x_amb, mf(x_amb), lw=2, label=lb, color=c)
    axes[0,1].set_title('Ambient Temperature (°C)', color='white')
    axes[0,1].legend(facecolor='#0f172a', labelcolor='white')

    x_mod = np.linspace(18, 66, 500)
    for mf, lb, c in zip(MOD_MF, ['Rendah','Sedang','Tinggi','Sangat Tinggi'], colors):
        axes[1,0].plot(x_mod, mf(x_mod), lw=2, label=lb, color=c)
    axes[1,0].set_title('Module Temperature (°C)', color='white')
    axes[1,0].legend(facecolor='#0f172a', labelcolor='white')

    x_dc = np.linspace(0, DC_MAX, 1000)
    out_labels = ['Sangat Rendah','Rendah','Sedang','Tinggi','Sangat Tinggi']
    out_colors = ['#1d4ed8','#60a5fa','#4ade80','#f97316','#ef4444']
    for mf, lb, c in zip(OUT_MF, out_labels, out_colors):
        axes[1,1].plot(x_dc/1000, mf(x_dc), lw=2, label=lb, color=c)
    axes[1,1].set_title('DC Power Output (kW)', color='white')
    axes[1,1].legend(facecolor='#0f172a', labelcolor='white', fontsize=9)

    plt.suptitle('Fungsi Keanggotaan Sistem Fuzzy PLTS', color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Singleton Sugeno
    st.markdown('<div class="section-header">📍 Singleton Output – Sugeno</div>', unsafe_allow_html=True)
    df_sg = pd.DataFrame({
        "Himpunan": out_labels,
        "Singleton Value (W)": list(SINGLETON.values()),
        "Singleton Value (kW)": [v/1000 for v in SINGLETON.values()],
    })
    st.dataframe(df_sg, use_container_width=True, hide_index=True)

# ─── TAB 3: RULE BASE ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📋 Rule Base (20 Rules)</div>', unsafe_allow_html=True)

    irr_names = ['Rendah','Sedang','Tinggi']
    amb_names = ['Rendah','Sedang','Tinggi']
    mod_names = ['Rendah','Sedang','Tinggi','Sangat Tinggi']
    out_names = ['Sangat Rendah','Rendah','Sedang','Tinggi','Sangat Tinggi']

    df_rules = pd.DataFrame([
        {
            "No": f"R{i+1:02d}",
            "IF Irradiation": irr_names[r_i],
            "AND Ambient Temp": amb_names[r_a],
            "AND Module Temp": mod_names[r_m],
            "THEN DC Power": out_names[r_o],
        }
        for i, (r_i, r_a, r_m, r_o) in enumerate(RULES)
    ])
    st.dataframe(df_rules, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background:#1e293b; border-radius:10px; padding:16px 20px; margin-top:16px; color:#94a3b8; font-size:13px; line-height:1.8;">
        <strong style="color:white;">Logika Utama Rule Base:</strong><br>
        🔵 Irradiasi tinggi = faktor dominan penentu daya output<br>
        🔴 Suhu modul sangat tinggi (>52°C) = turunkan efisiensi meski irradiasi tinggi<br>
        🟠 Kombinasi irradiasi sedang + suhu tinggi = output lebih rendah dari irradiasi tinggi + suhu rendah
    </div>
    """, unsafe_allow_html=True)

    # Perbandingan Mamdani vs Sugeno
    st.markdown('<div class="section-header">⚖️ Perbandingan Mamdani vs Sugeno</div>', unsafe_allow_html=True)
    df_comp = pd.DataFrame({
        "Aspek": ["Representasi output","Defuzzifikasi","Interpretabilitas","Kecepatan komputasi","Presisi numerik"],
        "Mamdani": ["Himpunan fuzzy (MF kontinu)","Centroid (COG)","⭐⭐⭐ Tinggi","🐢 Lebih lambat","Bergantung agregasi"],
        "Sugeno":  ["Singleton / fungsi linear","Weighted Average","⭐⭐ Sedang","🚀 Cepat","Umumnya lebih stabil"],
    })
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

# Footer
st.markdown("""
<div style="text-align:center; color:#475569; font-size:12px; margin-top:32px; padding-top:16px; border-top:1px solid #1e293b;">
    TUBES DKA 2526 · Implementasi Fuzzy Logic From Scratch · Telkom University Purwokerto<br>
    Dataset: <a href="https://www.kaggle.com/datasets/anikannal/solar-power-generation-data" target="_blank" style="color:#60a5fa;">Solar Power Generation Data (Kaggle)</a>
</div>
""", unsafe_allow_html=True)