# Fraud Detection in Banking Transactions via Deep Reinforcement Learning

Bu proje, bankacılık işlem dolandırıcılığını tespit etmek için derin pekiştirmeli öğrenme (Deep Reinforcement Learning) tabanlı bir yaklaşım uygular.

Özet
- Eğitim: `train.py`
- Model tanımı: `model.py`
- Uygulama/çalıştırma: `app.py`
- Yardımcı fonksiyonlar: `utils.py`
- Veri: `Bank_Transaction_Fraud_Detection.csv`

Gereksinimler
- Python 3.8+ (önerilir)
- Gerekli paketleri bir sanal ortamda kurun. Örnek:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # eğer requirements.txt yoksa, projedeki paketleri manuel kurun
```

Çalıştırma
1. Veriyi proje köküne koyun (`Bank_Transaction_Fraud_Detection.csv`).
2. Model eğitimi:

```bash
python train.py
```

3. Eğitilmiş modeli kullanmak / uygulamayı başlatmak:

```bash
python app.py
```

Notlar
- Eğer hassas bilgiler içeren `env.py` gibi dosyalar varsa bunları git'e koymayın (projede `env.py` örnek olarak bulunuyor).
- Eğitim sırasında GPU kullanılacaksa uygun CUDA sürücülerinin kurulu olduğundan emin olun.

Dosya Açıklamaları
- `train.py`: Eğitim döngüsü ve veri hazırlama.
- `model.py`: DQN/diger model tanımları.
- `app.py`: Eğitilmiş modeli yükleyip basit bir runner/örnek doğrulama için.
- `utils.py`: Yardımcı fonksiyonlar.

Katkıda Bulunma
- İyileştirmeler, hata düzeltmeleri ve README geliştirmeleri için pull request açabilirsiniz.

Lisans
- Varsayılan olarak açık kaynak; kullanmadan önce lisans eklemek isterseniz belirtin.
