# Fraud Detection in Banking Transactions via Deep Reinforcement Learning

Bu proje, bankacılık işlem dolandırıcılığını tespit etmek için derin pekiştirmeli öğrenme (Deep Reinforcement Learning) tabanlı bir yaklaşım uygular.

Özet
- Eğitim: `train.py`
- Model tanımı: `model.py`
- Uygulama/çalıştırma: `app.py`
- Yardımcı fonksiyonlar: `utils.py`
- Veri: `Bank_Transaction_Fraud_Detection.csv`, `BankaVerileri.xlsx`

Gereksinimler
- Python 3.8+ 

Çalıştırma
1. Veriyi proje köküne koyun (`Bank_Transaction_Fraud_Detection.csv`).
2. Model eğitimi:

bash
- python train.py


3. Modeli kullanmak ve uygulamayı başlatmak için:

bash
- python app.py

Dosya Açıklamaları
- `train.py`: Eğitim döngüsü ve veri hazırlama.
- `model.py`: DQN/diger model tanımları.
- `app.py`: Eğitilmiş modeli yükleyip basit bir örnek doğrulama için.
- `utils.py`: Yardımcı fonksiyonlar.

Katkıda Bulunma
- İyileştirmeler, hata düzeltmeleri ve README geliştirmeleri için pull request açabilirsiniz.
