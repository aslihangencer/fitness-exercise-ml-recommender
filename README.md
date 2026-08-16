# 🏋️ Veri Bilimi ve Makine Öğrenmesi ile Akıllı Egzersiz Analizi ve Öneri Sistemi

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

Bu proje, **1.324 fitness egzersizi** içeren zengin bir veri seti üzerinde Doğal Dil İşleme (NLP), Makine Öğrenmesi (Sınıflandırma) ve İçerik Tabanlı Öneri Sistemleri (Content-Based Filtering) kullanarak geliştirilmiş uçtan uca bir egzersiz analiz ve kişiselleştirilmiş antrenman öneri altyapısıdır.

---

## 🚀 Proje Özet Özellikleri

- **1.324 Egzersiz Verisi:** Hedef kas grupları, ekipmanlar, ikincil kaslar ve 10 dilde adım adım talimatlar.
- **NLP & TF-IDF Özellik Mühendisliği:** Metin verilerinin 500 boyutlu sayısal matrislere ve ekipman verilerinin One-Hot Encoding ile vektör uzayına dönüştürülmesi.
- **Sınıflandırma Modelleri (Supervised Learning):**
  - **Random Forest Classifier (Accuracy: %93,6| Weighted F1: 0.930)**
  - **Logistic Regression (Accuracy: %88,3 | Weighted F1: 0.880)**
  - **KNN (k=5) (Accuracy: %81,2 | Weighted F1: 0,800)**
- **İçerik Tabanlı Öneri Motoru (Unsupervised Learning):** Cosine Similarity kullanarak sakatlık, ekipman kısıtı veya alternatif arayışına yönelik anında akıllı öneri üretme.

---

## 📁 Proje Yapısı

```
fitness-exercise-ml-recommender/
├── data/
│   └── exercises.json                         # 1.324 egzersiz verisi
├── images/
│   ├── 01_category_distribution.png            # Vücut bölgesi dağılım grafiği
│   ├── 02_equipment_distribution.png           # Ekipman dağılım grafiği
│   ├── 03_bodyweight_ratio.png                 # Vücut ağırlığı vs Ekipmanlı oran
│   ├── 04_model_comparison.png                 # Model performans karşılaştırması
│   └── 05_confusion_matrix.png                 # Random Forest karmaşıklık matrisi
├── fitness_exercise_ml_recommender.ipynb       # Google Colab / Jupyter Notebook
├── run_ml_pipeline.py                           # Python ML boru hattı kodu
└── README.md                                   # Proje dokümantasyonu
```

---

## 📊 Örnek Sonuçlar ve Metrikler

| Model | Accuracy | Weighted F1-Score | Güçlü Yönü |
| --- | --- | --- | --- |
| **Random Forest** | **%93,6** | **0,930** | Metin ve kategorik özelliklerde en yüksek karmaşık ilişki yakalama başarımı |
| **Logistic Regression** | %88,3 | 0,880 | Hızlı eğitim, yüksek boyutlu seyrek verilerde tutarlı başarım |
| **KNN (k=5)** | %81,2 | 0,800 | Basit mesafe tabanlı sınıflandırma |

---

## 🛠️ Kurulum ve Çalıştırma

### Google Colab Üzerinde Çalıştırma (Önerilen)
Repo içerisinde yer alan `fitness_exercise_ml_recommender.ipynb` dosyasını Google Colab'a yükleyerek veya üstteki **Open in Colab** butonuna tıklayarak doğrudan çalıştırabilirsiniz.

### Yerel Ortamda Çalıştırma (Local)
1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/aslihangencer/fitness-exercise-ml-recommender.git
   cd fitness-exercise-ml-recommender
   ```
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn nbformat
   ```
3. Python Boru Hattını Çalıştırın:
   ```bash
   python run_ml_pipeline.py
   ```

---

## 🤝 Katkıda Bulunma
Her türlü katkıya ve fikir önerisine açığım! Pull Request açabilir veya Issue oluşturabilirsiniz.

## 📄 Lisans
Bu proje MIT Lisansı ile lisanslanmıştır.
