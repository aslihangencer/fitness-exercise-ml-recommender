import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# Create output directories
os.makedirs('exercise_recommender_ml/images', exist_ok=True)

# 1. Load Dataset
with open('exercise_recommender_ml/exercises.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)
print(f"Dataset Loaded. Total records: {len(df)}")
print("Columns:", df.columns.tolist())

# Prepare instructions text
def extract_instruction_text(inst):
    if isinstance(inst, dict):
        return inst.get('en', '')
    elif isinstance(inst, list):
        return ' '.join(inst)
    elif isinstance(inst, str):
        return inst
    return ''

df['clean_instructions'] = df['instructions'].apply(extract_instruction_text)
df['text_data'] = df['name'].fillna('') + " " + df['equipment'].fillna('') + " " + df['clean_instructions']

# 2. EDA & Visualizations

# Chart 1: Body Part / Category Distribution
plt.figure(figsize=(10, 6))
category_counts = df['category'].value_counts()
ax = sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')
plt.title('Egzersizlerin Vücut Bölgesine (Category) Göre Dağılımı', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Egzersiz Sayısı', fontsize=12)
plt.ylabel('Vücut Bölgesi (Category)', fontsize=12)

for i, count in enumerate(category_counts.values):
    ax.text(count + 3, i, str(count), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('exercise_recommender_ml/images/01_category_distribution.png', bbox_inches='tight')
plt.close()

# Chart 2: Top Equipment Distribution
plt.figure(figsize=(10, 6))
equipment_counts = df['equipment'].value_counts().head(10)
ax = sns.barplot(x=equipment_counts.values, y=equipment_counts.index, palette='mako')
plt.title('En Çok Kullanılan Top 10 Ekipman Dağılımı', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Egzersiz Sayısı', fontsize=12)
plt.ylabel('Ekipman Türü', fontsize=12)

for i, count in enumerate(equipment_counts.values):
    ax.text(count + 3, i, str(count), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('exercise_recommender_ml/images/02_equipment_distribution.png', bbox_inches='tight')
plt.close()

# Chart 3: Bodyweight vs Non-Bodyweight ratio
df['is_bodyweight'] = df['equipment'].apply(lambda x: 'Vücut Ağırlığı (Body Weight)' if str(x).lower() == 'body weight' else 'Ekipmanlı Egzersizler')
bw_counts = df['is_bodyweight'].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(bw_counts.values, labels=bw_counts.index, autopct='%1.1f%%', startangle=140, colors=['#3498db', '#e74c3c'], explode=(0.05, 0), textprops={'fontsize': 12, 'weight': 'bold'})
plt.title('Ekipman Gereksinimi: Vücut Ağırlığı vs Ekipmanlı', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('exercise_recommender_ml/images/03_bodyweight_ratio.png', bbox_inches='tight')
plt.close()

# 3. Data Preprocessing & Feature Engineering
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X_text = tfidf.fit_transform(df['text_data']).toarray()

# Encode equipment as one-hot features
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_equip = ohe.fit_transform(df[['equipment']])

# Combine Text + Equipment features for classification
X_combined = np.hstack([X_text, X_equip])
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# 4. Supervised Classification Models

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
}

results = []
confusion_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'Weighted F1-Score': f1
    })
    
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    print(f"\n--- {name} Classification Report ---")
    print(classification_report(y_test, y_pred))

res_df = pd.DataFrame(results)
print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(res_df.to_string(index=False))

# Chart 4: Model Performance Comparison Barplot
plt.figure(figsize=(10, 5))
melted_df = res_df.melt(id_vars='Model', value_vars=['Accuracy', 'Weighted F1-Score'], var_name='Metric', value_name='Score')
ax = sns.barplot(data=melted_df, x='Model', y='Score', hue='Metric', palette='Set2')
plt.title('Sınıflandırma Modelleri Başarım Karşılaştırması', fontsize=14, fontweight='bold', pad=15)
plt.ylim(0.7, 1.0)
plt.ylabel('Skor (0 - 1)', fontsize=12)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., height - 0.03),
                    ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('exercise_recommender_ml/images/04_model_comparison.png', bbox_inches='tight')
plt.close()

# Chart 5: Confusion Matrix for Random Forest
rf_cm = confusion_matrices['Random Forest']
categories = sorted(y.unique())

plt.figure(figsize=(10, 8))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title('Random Forest - Karmaşıklık Matrisi (Confusion Matrix)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
plt.ylabel('Gerçek Sınıf', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('exercise_recommender_ml/images/05_confusion_matrix.png', bbox_inches='tight')
plt.close()

# 5. Content-Based Recommendation System using Cosine Similarity
similarity_matrix = cosine_similarity(X_combined)

def get_exercise_recommendations(exercise_id_or_name, top_n=5, filter_equipment=None):
    # Search by ID or Name
    if exercise_id_or_name in df['id'].values:
        idx = df[df['id'] == exercise_id_or_name].index[0]
    else:
        matches = df[df['name'].str.lower().str.contains(str(exercise_id_or_name).lower(), na=False)]
        if len(matches) == 0:
            return f"Egzersiz bulunamadı: '{exercise_id_or_name}'"
        idx = matches.index[0]

    target_exercise = df.iloc[idx]
    scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for i, score in sorted_scores:
        if i == idx:
            continue
        candidate = df.iloc[i]
        
        # Apply equipment filter if requested
        if filter_equipment and str(candidate['equipment']).lower() != filter_equipment.lower():
            continue
            
        recommendations.append({
            'Benzerlik Skoru': round(float(score), 4),
            'Egzersiz Adı': candidate['name'],
            'Vücut Bölgesi': candidate['category'],
            'Ekipman': candidate['equipment'],
            'Hedef Kas': candidate.get('target', 'N/A')
        })
        
        if len(recommendations) == top_n:
            break
            
    return target_exercise, pd.DataFrame(recommendations)

# Demonstration of Recommendation
sample_id = "0001" # Barbell Full Squat or similar
target, recs = get_exercise_recommendations(sample_id, top_n=5)

print(f"\n--- ÖNERİ MOTORU DEMO (Hedef: {target['name']} - {target['category']}) ---")
print(recs.to_string(index=False))

# Demonstration of Bodyweight Filtered Recommendation
print(f"\n--- Sadece Vücut Ağırlığı (Body Weight) Alternatifleri ---")
_, recs_bw = get_exercise_recommendations(sample_id, top_n=5, filter_equipment="body weight")
print(recs_bw.to_string(index=False))

print("\nPipeline Execution Completed Successfully!")
