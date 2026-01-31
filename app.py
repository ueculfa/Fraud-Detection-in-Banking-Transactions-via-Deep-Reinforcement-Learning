import streamlit as st
import pandas as pd
import torch
import numpy as np
import time
import plotly.express as px
import requests
from streamlit_lottie import st_lottie
from model import FraudDQN
from utils import risk_analizi_yap, veriyi_yukle, RISK_COLUMNS

# --PANDAS LİMİT AYARI --
pd.set_option("styler.render.max_elements", 2000000)

# --SAYFA KONFİGÜRASYONU --
st.set_page_config(page_title="AI Fraud Guard | SOC Center", page_icon="🛡️", layout="wide")

def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_thief = load_lottieurl("https://lottie.host/548f07e5-1d6a-4d37-8051-419b67329437/A90Wv8qA7L.json")

# --GELİŞMİŞ GÖRSEL TASARIM (CSS) --
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #050c12 0%, #10202d 50%, #050c12 100%); color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: rgba(15, 32, 45, 0.9) !important; border-right: 2px solid #00d2ff; }
    [data-testid="stFileUploader"] { background-color: rgba(0, 210, 255, 0.05) !important; border: 2px dashed #00d2ff !important; border-radius: 20px !important; box-shadow: 0 0 20px rgba(0, 210, 255, 0.15) !important; }
    div[data-testid="stMetric"] { background-color: rgba(255, 255, 255, 0.03); border: 1px solid rgba(0, 210, 255, 0.3); border-radius: 15px; }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: white !important; }
    [data-testid="stSidebar"] .stAlert p { color: #ffd900 !important; }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] .stCaption p, [data-testid="stSidebar"] caption { color: #ffd900 !important; }
    div[data-testid="stDownloadButton"] button p { color: black !important; }
    h1, h2, h3 { color: #00d2ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --SIDEBAR --
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🛡️ OPERASYON MERKEZİ</h2>", unsafe_allow_html=True)
    st.divider()
    st.info("**DQN Modeli:** Aktif\n\n**Girdi Katmanı:** 11 Parametre")
    st.divider()
    st.caption("Geliştirici: Utku Enes Culfa")

st.title("AI FRAUD GUARD: AKILLI ANALİZ SİSTEMİ")
st.markdown("<p style='color: white;'>DQN Algoritması ve 11 Risk Faktörü ile Şüpheli İşlem Tespiti</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Banka Ekstresini Yükleyin", type=['xlsx', 'csv'])

if uploaded_file:
    with st.spinner("Yapay Zeka Analiz Ediyor..."):
        df = veriyi_yukle(uploaded_file)
        
        # Veri Sınırlama (Performans için)
        if len(df) > 10000:
            st.warning(f"Büyük veri seti tespit edildi. İlk 10.000 satır işleniyor.")
            df = df.head(10000)
            
        features = risk_analizi_yap(df)
        
        # Modeli Yükle
        model = FraudDQN(state_dim=11, action_dim=2)
        model.load_state_dict(torch.load("fraud_dqn_model.pth"))
        model.eval()

        dqn_results, probabilities, triggered_list = [], [], []
        with torch.no_grad():
            for i in range(len(features)):
                state_t = torch.FloatTensor(features[i]).unsqueeze(0)
                q_values = model(state_t)
                action = q_values.argmax().item()
                risk_percent = float(torch.softmax(q_values, dim=1)[0][1] * 100)
                
                dqn_results.append("⚠️ ŞÜPHELİ" if action == 1 else "✅ GÜVENLİ")
                probabilities.append(risk_percent)
                active_rules = [RISK_COLUMNS[j] for j in range(11) if features[i][j] == 1]
                triggered_list.append(", ".join(active_rules) if active_rules else "Risk Yok")

        df['DQN_Kararı'] = dqn_results
        df['Risk_Skoru_%'] = probabilities
        df['Tetiklenen_Riskler'] = triggered_list

    # --DASHBOARD METRİKLER --
    m1, m2, m3, m4 = st.columns(4) 
    m1.metric("İşlem Sayısı", len(df))
    m2.metric("Tespit Edilen Risk", dqn_results.count("⚠️ ŞÜPHELİ"), delta_color="inverse")
    m3.metric("Ortalama Risk", f"%{np.mean(probabilities):.1f}")
    m4.metric("Güvenlik Skoru", f"%{100-np.mean(probabilities):.1f}")

    # --GRAFİKLER --
    st.markdown("### 📊 İstatistiksel Analiz")
    c1, c2 = st.columns(2)
    
    with c1:
        # Bar Chart - Risk Dağılımı
        risk_counts = pd.DataFrame(features, columns=RISK_COLUMNS).sum().sort_values()
        fig_rules = px.bar(risk_counts, orientation='h', title="Risk İhlal Dağılımı (Parametre Bazlı)",
                           color_discrete_sequence=["#ffd900"])
        fig_rules.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', title_font_color='white')
        st.plotly_chart(fig_rules, use_container_width=True)
        
    with c2:
        # --GÜNCELLENEN TREND GRAFİĞİ --
        fig_trend = px.line(df, y='Risk_Skoru_%', title="İşlem Bazlı Risk Trend Analizi")
        fig_trend.update_layout(title=dict(text="İşlem Bazlı Risk Trend Analizi", font=dict(color='white')))
        
        # Çizgi rengini altın/sarı yapıyoruz (Mavi karmaşasını önlemek için)
        fig_trend.update_traces(line_color='#ffd900', line_width=1.5)
        
        # Kritik eşik çizgisi ekleyelim (Kırmızı %50 çizgisi)
        fig_trend.add_hline(y=50, line_dash="dash", line_color="#ff4b4b", 
                           annotation_text="Kritik Eşik (%50)", annotation_position="top left",
                           annotation_font_color="white")
        
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='white',
            title_font_color='white',
            xaxis_title="İşlem Sırası (Index)",
            yaxis_title="Risk Yüzdesi (%)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # --DETAYLI TABLO --
    st.markdown("### 🔍 Detaylı Karar ve Raporlama")
    valid_cols = [c for c in df.columns if any(k in c.lower() for k in ['tarih', 'date', 'açıklama', 'description', 'tutar', 'amount', 'kararı', 'skoru', 'riskler'])]
    
    def color_dqn(val):
        return 'background-color: rgba(255, 75, 75, 0.3)' if val == "⚠️ ŞÜPHELİ" else 'background-color: rgba(0, 210, 255, 0.1)'

    st.dataframe(df[valid_cols].style.applymap(color_dqn, subset=['DQN_Kararı'] if 'DQN_Kararı' in df.columns else []))

    st.download_button("📥 Analiz Raporunu İndir", df.to_csv(index=False).encode('utf-8'), "fraud_raporu.csv", "text/csv")

else:
    st.warning("Lütfen analiz başlatmak için bir banka verisi yükleyiniz.")