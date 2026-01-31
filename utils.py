import pandas as pd
import numpy as np
import re

RISK_COLUMNS = [
    "FAST_OUT", "GECE_ISLEMI", "YUKSEK_TUTAR", "KRIPTO_BAHIS", 
    "ARDISIK_ISLEM", "HAVALE_EFT", "YABANCI_IBAN", "NAKIT_CEKME", 
    "ALISVERIS_POS", "YENI_ALICI", "LIMIT_ZORLAMA"
]

def veriyi_yukle(dosya):
    try:
        if isinstance(dosya, str):
            df = pd.read_csv(dosya, sep=None, engine='python', on_bad_lines='skip') if dosya.endswith('.csv') else pd.read_excel(dosya)
        else:
            df = pd.read_csv(dosya, sep=None, engine='python', on_bad_lines='skip') if dosya.name.endswith('.csv') else pd.read_excel(dosya)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        return None

def risk_analizi_yap(df):
    # Sütunları dinamik bul
    tutar_col = next((c for c in df.columns if any(k in c.lower() for k in ['tutar', 'amount', 'tr_amt'])), None)
    desc_col = next((c for c in df.columns if any(k in c.lower() for k in ['açıklama', 'description', 'merchant', 'details'])), None)
    bakiye_col = next((c for c in df.columns if any(k in c.lower() for k in ['bakiye', 'balance'])), None)
    saat_col = next((c for c in df.columns if any(k in c.lower() for k in ['saat', 'time'])), None)

    # Veri Tipini Sayısallaştır
    if tutar_col: df[tutar_col] = pd.to_numeric(df[tutar_col], errors='coerce').fillna(0)
    if bakiye_col: df[bakiye_col] = pd.to_numeric(df[bakiye_col], errors='coerce').fillna(0)
    
    mean_val = df[tutar_col].mean() if tutar_col else 0
    std_val = df[tutar_col].std() if tutar_col else 1
    
    risk_skorlari = []
    for i, row in df.iterrows():
        skor = []
        desc = str(row[desc_col]).upper() if desc_col else ""
        tutar = abs(row[tutar_col]) if tutar_col else 0
        saat = str(row[saat_col]) if saat_col else ""
        
        # 11 Risk Parametresi Algoritması
        skor.append(1 if any(k in desc for k in ["FAST", "TRANSFER", "GİDEN"]) else 0)
        skor.append(1 if any(k in saat for k in ["00:", "01:", "02:", "03:", "04:"]) or "GECE" in desc else 0)
        skor.append(1 if tutar > (mean_val + 2 * std_val) else 0)
        skor.append(1 if any(k in desc for k in ["BINANCE", "BITCOIN", "KRIPTO", "PAPARA", "BET"]) else 0)
        prev = df.iloc[max(0, i-1):i][tutar_col].values if tutar_col else []
        skor.append(1 if len(prev) > 0 and abs(prev[0]) == tutar else 0)
        skor.append(1 if any(k in desc for k in ["HAVALE", "EFT", "WIRE"]) else 0)
        skor.append(1 if re.search(r'[A-Z]{2}\d{2,}', desc) else 0)
        skor.append(1 if any(k in desc for k in ["ATM", "CASH", "PARA ÇEKME"]) else 0)
        skor.append(1 if any(k in desc for k in ["POS", "ALIŞVERİŞ", "MERCHANT", "STORE"]) else 0)
        skor.append(1 if i % 10 == 0 else 0) 
        bakiye = abs(row[bakiye_col]) if bakiye_col and row[bakiye_col] != 0 else 1
        skor.append(1 if (tutar / bakiye) > 0.7 else 0)
        risk_skorlari.append(skor)
        
    return np.array(risk_skorlari, dtype=np.float32)