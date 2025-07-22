# meme kanseri model karsilastirma

# ==============================
# 1. Gerekli Kütüphaneler
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Matplotlib / Seaborn ayarları
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
 # ==============================
# 2. Veriyi Yükle ve Hazırla
# ==============================
data = pd.read_csv("breast-cancer.csv")

# Diagnosis sütunu kontrol
if "diagnosis" not in data.columns:
    raise ValueError("CSV dosyasında 'diagnosis' sütunu bulunamadı!")

# Etiketleri sayıya çevir
data["target"] = data["diagnosis"].map({"M": 0, "B": 1})
X = data.drop(columns=["diagnosis", "target"])
y = data["target"]

# ==============================
# 3. Eğitim/Test Ayırma
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 4. Modelleri Tanımla
# ==============================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True)  # ROC eğrisi için probability=True
}

results = {}
plt.figure(figsize=(8, 6))

# ==============================
# 5. Eğitim, Tahmin, ROC, Doğruluk
# ==============================
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "auc": roc_auc
    }

    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

# ROC grafiği çiz
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Eğrileri")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 6. Doğruluk Sonuçları
# ==============================
print("\n📊 Karşılaştırmalı Sonuçlar:\n")
for name, metrics in results.items():
    print(f"{name:<20} | Doğruluk: %{metrics['accuracy']*100:.2f} | AUC: {metrics['auc']:.4f}")
# ==============================
# 6.1 Confusion Matrix Her Model İçin
# ==============================
print("\n🧮 Confusion Matrix'ler:\n")
for name, metrics in results.items():
    y_pred = metrics["model"].predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"\n{name} Confusion Matrix:")
    print(cm)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Reds",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"]
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.show()

# ==============================
# 7. En İyi Modelin Özellik Önemi 
# ==============================
best_model_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_model_name]["model"]

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.values[:10], y=importances.index[:10])
    plt.title(f"{best_model_name} - En Önemli 10 Özellik")
    plt.xlabel("Önem Skoru")
    plt.tight_layout()
    plt.show()
    print("\n🔍 En önemli 3 özellik:")
    print(importances.head(3))
else:
    print(f"\n⚠️ {best_model_name} modeli için özellik önemi desteklenmiyor.")
