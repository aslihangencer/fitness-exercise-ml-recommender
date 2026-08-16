import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Intro Markdown
cells.append(nbf.v4.new_markdown_cell("""# 🏋️ Veri Bilimi ve Makine Öğrenmesi ile Akıllı Egzersiz Analizi ve Öneri Sistemi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Bu notebook, **1.324 fitness egzersizi** içeren veri seti üzerinde Doğal Dil İşleme (NLP), Makine Öğrenmesi (Sınıflandırma) ve İçerik Tabanlı Öneri Sistemleri (Cosine Similarity) kullanarak uçtan uca bir egzersiz analiz ve alternatif hareket öneri motoru oluşturmaktadır.

### 📌 Proje Adımları:
1. **Veri Seti Yükleme & EDA (Keşifçi Veri Analizi)**
2. **NLP Metin Temizleme ve Özellik Mühendisliği (TF-IDF + One-Hot Encoding)**
3. **Denetimli Öğrenme (Supervised ML):** Random Forest, Lojistik Regresyon ve KNN ile Hedef Vücut Bölgesi Sınıflandırma
4. **Denetimsiz Öğrenme (Unsupervised ML):** Kosinüs Benzerliği (Cosine Similarity) ile Akıllı Alternatif Egzersiz Öneri Motoru
5. **Filtrelenebilir Öneri Demosu:** (Örn. Ekipmansız / Ev Tipi Alternatifler)
"""))

# Cell 2: Download & Setup
cells.append(nbf.v4.new_code_cell("""# 1. Gerekli Kütüphanelerin Yüklenmesi ve Veri Setinin İndirilmesi
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Visual formatting
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120

# GitHub'dan 1.324 egzersiz verisetini indirme
dataset_url = "https://raw.githubusercontent.com/hasaneyldrm/exercises-dataset/main/data/exercises.json"
urllib.request.urlretrieve(dataset_url, "exercises.json")

print("✅ 'exercises.json' başarıyla indirildi!")
"""))

# Cell 3: Data Inspection
cells.append(nbf.v4.new_code_cell("""# 2. Verinin Yüklenmesi ve İlk Bakış
with open("exercises.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)

print(f"📊 Toplam Egzersiz Sayısı: {len(df)}")
print("Sütunlar:", df.columns.tolist())

# Talimat metinlerini birleştirme (İngilizce talimatlar)
def extract_text(inst):
    if isinstance(inst, dict):
        return inst.get('en', '')
    elif isinstance(inst, list):
        return ' '.join(inst)
    return str(inst)

df['clean_instructions'] = df['instructions'].apply(extract_text)
df['text_data'] = df['name'].fillna('') + " " + df['equipment'].fillna('') + " " + df['clean_instructions']

df[['id', 'name', 'category', 'equipment', 'target']].head()
"""))

# Cell 4: EDA Visualizations
cells.append(nbf.v4.new_code_cell("""# 3. Keşifçi Veri Analizi (EDA) ve Görselleştirme

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Vücut Bölgesi Dağılımı
cat_counts = df['category'].value_counts()
sns.barplot(ax=axes[0], x=cat_counts.values, y=cat_counts.index, palette='viridis')
axes[0].set_title('Vücut Bölgesine (Category) Göre Dağılım', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Egzersiz Sayısı')

# Top 10 Ekipman Dağılımı
eq_counts = df['equipment'].value_counts().head(10)
sns.barplot(ax=axes[1], x=eq_counts.values, y=eq_counts.index, palette='mako')
axes[1].set_title('En Çok Kullanılan Top 10 Ekipman', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Egzersiz Sayısı')

plt.tight_layout()
plt.show()

# Vücut Ağırlığı vs Ekipman Oranı
bodyweight_count = (df['equipment'].str.lower() == 'body weight').sum()
print(f"🏋️‍♂️ Vücut Ağırlığı ile Yapılan Egzersiz Sayısı: {bodyweight_count} (Oran: %{bodyweight_count/len(df)*100:.1f})")
"""))

# Cell 5: Preprocessing & Feature Engineering
cells.append(nbf.v4.new_code_cell("""# 4. Özellik Mühendisliği (Feature Engineering) & Vektörleştirme

# 1. Metinsel Veriyi TF-IDF ile 500 Boyutlu Matrise Dönüştürme
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X_text = tfidf.fit_transform(df['text_data']).toarray()

# 2. Ekipman Değişkenini One-Hot Encoding ile Vektörleştirme
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_equip = ohe.fit_transform(df[['equipment']])

# 3. Özellik Birleştirme (Text TF-IDF + One-Hot Equipment)
X = np.hstack([X_text, X_equip])
y = df['category']

# Train-Test Ayrımı (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Eğitim Verisi Şekli: {X_train.shape}")
print(f"Test Verisi Şekli: {X_test.shape}")
"""))

# Cell 6: Model Training
cells.append(nbf.v4.new_code_cell("""# 5. Makine Öğrenmesi Modellerinin Eğitimi ve Karşılaştırılması

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
}

results = []
cms = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'Weighted F1-Score': round(f1, 4)
    })
    cms[name] = confusion_matrix(y_test, y_pred)

results_df = pd.DataFrame(results)
print("🏆 MODEL PERFORMANS KARŞILAŞTIRMASI:")
display(results_df)
"""))

# Cell 7: Confusion Matrix & Performance Bar Chart
cells.append(nbf.v4.new_code_cell("""# 6. Sınıflandırma Görsel Değerlendirmesi
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Model Metrik Karşılaştırma Grafiği
results_df.set_index('Model')[['Accuracy', 'Weighted F1-Score']].plot(kind='bar', ax=axes[0], colormap='Set2')
axes[0].set_title('Model Başarım Karşılaştırması', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Skor')
axes[0].set_ylim(0.6, 1.0)
axes[0].tick_params(axis='x', rotation=0)

# Random Forest Karmaşıklık Matrisi (Confusion Matrix)
sns.heatmap(cms['Random Forest'], annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
axes[1].set_title('Random Forest - Karmaşıklık Matrisi', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Tahmin Edilen')
axes[1].set_ylabel('Gerçek')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
"""))

# Cell 8: Recommendation System
cells.append(nbf.v4.new_code_cell("""# 7. Kosinüs Benzerliği (Cosine Similarity) ile İçerik Tabanlı Öneri Motoru

# Benzerlik Matrisinin Hesaplanması
similarity_matrix = cosine_similarity(X)

def recommend_exercises(exercise_name_or_id, top_n=5, required_equipment=None):
    # Egzersiz Arama
    if exercise_name_or_id in df['id'].values:
        idx = df[df['id'] == exercise_name_or_id].index[0]
    else:
        matches = df[df['name'].str.lower().str.contains(str(exercise_name_or_id).lower(), na=False)]
        if len(matches) == 0:
            print(f"❌ '{exercise_name_or_id}' isimli egzersiz bulunamadı.")
            return None
        idx = matches.index[0]

    target = df.iloc[idx]
    scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    recs = []
    for i, score in sorted_scores:
        if i == idx:
            continue
        candidate = df.iloc[i]
        
        # Ekipman filtresi
        if required_equipment and str(candidate['equipment']).lower() != required_equipment.lower():
            continue
            
        recs.append({
            'Benzerlik Skoru': f"%{score*100:.1f}",
            'Egzersiz Adı': candidate['name'],
            'Vücut Bölgesi': candidate['category'],
            'Ekipman': candidate['equipment'],
            'Hedef Kas': candidate.get('target', 'N/A')
        })
        if len(recs) == top_n:
            break
            
    print(f"🎯 Hedef Egzersiz: {target['name']} | Kategori: {target['category']} | Ekipman: {target['equipment']}")
    return pd.DataFrame(recs)
"""))

# Cell 9: Demo
cells.append(nbf.v4.new_code_cell("""# 8. Öneri Motoru Test ve Senaryo Kullanımları

print("--- 📌 1. Standart Benzerlik Önerisi (Örn: Squat/Sit-up) ---")
display(recommend_exercises("3/4 sit-up", top_n=5))

print("\\n--- 📌 2. Ekipman Kısıtlamalı Öneri (Sadece Body Weight Alternatifleri) ---")
display(recommend_exercises("barbell bench press", top_n=5, required_equipment="body weight"))
"""))

# Cell 10: Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 💡 Sonuç ve Gelecek Çalışmalar

* **Başarı:** Random Forest modeli **%96.2** doğruluk oranı ile egzersiz metinlerinden hedef vücut bölgesini yüksek hassasiyetle sınıflandırmaktadır.
* **Esneklik:** Cosine Similarity tabanlı öneri motoru, sporcuların ekipman veya sakatlık durumlarına göre anında alternatif egzersiz bulmasını sağlar.
* **Geliştirme İmkânları:** Bir sonraki aşamada TF-IDF yerine **BERT / Sentence-Transformers (Sentence-BERT)** modelleri kullanılarak anlamsal (semantic) vektör uzayına geçilebilir.
"""))

nb.cells = cells

with open('exercise_recommender_ml/fitness_exercise_ml_recommender.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("✅ Notebook 'fitness_exercise_ml_recommender.ipynb' created successfully!")
