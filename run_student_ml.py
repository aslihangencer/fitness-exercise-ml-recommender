import json
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# 1. Veriyi İndirme ve Yükleme
url = "https://raw.githubusercontent.com/hasaneyldrm/exercises-dataset/main/data/exercises.json"
urllib.request.urlretrieve(url, "exercises.json")

with open("exercises.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Metin verilerini hazırlama (Egzersiz adı + İngilizce talimatlar)
def get_instruction_en(inst):
    if isinstance(inst, dict):
        return inst.get("en", "")
    elif isinstance(inst, list):
        return " ".join(inst)
    return str(inst)

df["clean_instructions"] = df["instructions"].apply(get_instruction_en)
df["text_data"] = df["name"].fillna("") + " " + df["clean_instructions"].fillna("")

print("Egzersiz Sayısı:", len(df))
print("\nKategori Dağılımı:\n", df["category"].value_counts())
print("\nTop 5 Ekipman:\n", df["equipment"].value_counts().head(5))

bodyweight_count = (df["equipment"].str.lower() == "body weight").sum()
print("\nBody Weight Egzersiz Sayısı:", bodyweight_count)

# 2. EDA Grafikleri
plt.figure(figsize=(10, 5))
sns.countplot(y="category", data=df, order=df["category"].value_counts().index, palette="viridis")
plt.title("Vücut Bölgelerine Göre Egzersiz Sayısı")
plt.xlabel("Egzersiz Sayısı")
plt.ylabel("Vücut Bölgesi (Category)")
plt.tight_layout()
plt.savefig("images/01_category_distribution.png")
plt.close()

plt.figure(figsize=(10, 5))
top_equipment = df["equipment"].value_counts().head(10)
sns.barplot(x=top_equipment.values, y=top_equipment.index, palette="mako")
plt.title("En Çok Kullanılan 10 Ekipman")
plt.xlabel("Egzersiz Sayısı")
plt.ylabel("Ekipman Türü")
plt.tight_layout()
plt.savefig("images/02_equipment_distribution.png")
plt.close()

# 3. TF-IDF ve Veri Bölme
tfidf = TfidfVectorizer(max_features=500, stop_words="english")
X_text = tfidf.fit_transform(df["text_data"]).toarray()
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Modelleri Eğitme
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)
    })
    
    if name == "Random Forest":
        rf_pred = y_pred

results_df = pd.DataFrame(results)
print("\n--- MODEL PERFORMANS TABLOSU ---")
print(results_df.to_string(index=False))

# Confusion Matrix for Random Forest
plt.figure(figsize=(10, 8))
categories = sorted(y.unique())
cm = confusion_matrix(y_test, rf_pred, labels=categories)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.title("Random Forest - Karmaşıklık Matrisi (Confusion Matrix)")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("images/05_confusion_matrix.png")
plt.close()

# Model Karşılaştırma Grafiği
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=results_df, palette="Set2")
plt.title("Modellerin Accuracy (Doğruluk) Karşılaştırması")
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig("images/04_model_comparison.png")
plt.close()

# 5. İçerik Tabanlı Öneri Sistemi (Cosine Similarity)
similarity_matrix = cosine_similarity(X_text)

def get_recommendations(exercise_name, top_n=3):
    matches = df[df["name"].str.lower().str.contains(exercise_name.lower(), na=False)]
    if len(matches) == 0:
        return f"'{exercise_name}' adında egzersiz bulunamadı."
    
    idx = matches.index[0]
    target_exercise = df.iloc[idx]
    
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommended_indices = [i[0] for i in sim_scores if i[0] != idx][:top_n]
    
    recs = df.iloc[recommended_indices][["name", "category", "equipment"]]
    recs["Benzerlik Skoru"] = [round(sim_scores[i][1], 4) for i in recommended_indices]
    
    print(f"\nSeçilen Egzersiz: {target_exercise['name']} ({target_exercise['category']})")
    return recs

print("\n--- ÖNERİ SİSTEMİ DEMO ---")
print(get_recommendations("bench press", top_n=3))
print(get_recommendations("squat", top_n=3))
