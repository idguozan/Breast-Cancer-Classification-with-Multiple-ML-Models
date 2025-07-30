#!/usr/bin/env python3
"""
Basit çalıştırma scripti - Import sorunları çözümü ile
"""

import os
import sys
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.filterwarnings('ignore')

def main():
    """Ana fonksiyon - Basitleştirilmiş analiz"""
    
    print("🎯 MEME KANSERİ SINIFLANDIRMA - BASİT ANALİZ")
    print("=" * 60)
    
    # Veri yükleme
    try:
        data = pd.read_csv("data/breast-cancer.csv")
        print(f"✅ Veri yüklendi: {data.shape}")
    except FileNotFoundError:
        print("❌ Veri dosyası bulunamadı!")
        print("Lütfen breast-cancer.csv dosyasını 'data/' klasörüne yerleştirin.")
        return
    
    # Ön işleme
    print("\n📊 Veri ön işleme...")
    data["target"] = data["diagnosis"].map({"M": 0, "B": 1})
    
    # Gereksiz sütunları kaldır
    columns_to_drop = ["diagnosis", "target"]
    if "id" in data.columns:
        columns_to_drop.append("id")
    if "Unnamed: 32" in data.columns:
        columns_to_drop.append("Unnamed: 32")
    
    X = data.drop(columns=columns_to_drop)
    y = data["target"]
    
    # Sadece numerik sütunları al
    X = X.select_dtypes(include=[np.number])
    
    print(f"✅ Özellik sayısı: {X.shape[1]}")
    print(f"✅ Örnek sayısı: {X.shape[0]}")
    
    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Modeller
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }
    
    results = {}
    
    print("\n🤖 Model eğitimi başlıyor...")
    print("-" * 40)
    
    # Her modeli eğit
    for name, model in models.items():
        print(f"⏳ {name} eğitiliyor...")
        
        # Model eğitimi
        model.fit(X_train, y_train)
        
        # Tahminler
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "auc": roc_auc,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "fpr": fpr,
            "tpr": tpr
        }
        
        print(f"✅ {name} - Doğruluk: {accuracy:.4f}, AUC: {roc_auc:.4f}")
    
    # Sonuçları göster
    print("\n📊 SONUÇLAR:")
    print("=" * 60)
    print(f"{'Model':<20} {'Doğruluk':<12} {'AUC':<8}")
    print("-" * 40)
    
    # En iyi modeli bul
    best_model_name = max(results, key=lambda k: results[k]["auc"])
    
    for name, result in sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True):
        star = "⭐" if name == best_model_name else "  "
        print(f"{star} {name:<18} {result['accuracy']:.4f}       {result['auc']:.4f}")
    
    print(f"\n🏆 En iyi model: {best_model_name}")
    
    # Görselleştirmeler
    print("\n📈 Görselleştirmeler oluşturuluyor...")
    
    # ROC eğrileri
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        plt.plot(result["fpr"], result["tpr"], 
                linewidth=2, label=f'{name} (AUC = {result["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Rastgele')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (FPR)')
    plt.ylabel('Doğru Pozitif Oranı (TPR)')
    plt.title('ROC Eğrileri Karşılaştırması')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Sonuçlar klasörünü oluştur
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/roc_curves_simple.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion matrices
    n_models = len(results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, result) in enumerate(results.items()):
        row = idx // cols
        col = idx % cols
        
        if rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        
        cm = confusion_matrix(y_test, result["y_pred"])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Malignant', 'Benign'],
                   yticklabels=['Malignant', 'Benign'])
        
        ax.set_title(f'{name}\\nAccuracy: {result["accuracy"]:.4f}')
        ax.set_xlabel('Tahmin')
        ax.set_ylabel('Gerçek')
    
    # Boş subplot'ları gizle
    for idx in range(n_models, rows * cols):
        row = idx // cols
        col = idx % cols
        if rows > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/confusion_matrices_simple.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # En iyi modelin özellik önemleri (eğer varsa)
    best_model = results[best_model_name]["model"]
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n🔍 {best_model_name} - En Önemli 10 Özellik:")
        print("-" * 50)
        
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        top_features = importances.nlargest(10)
        
        for i, (feature, importance) in enumerate(top_features.items(), 1):
            print(f"{i:2d}. {feature:<25}: {importance:.4f}")
        
        # Özellik önemleri grafiği
        plt.figure(figsize=(12, 8))
        top_features.plot(kind='barh')
        plt.title(f'{best_model_name} - En Önemli 10 Özellik')
        plt.xlabel('Önem Skoru')
        plt.tight_layout()
        plt.savefig("results/feature_importance_simple.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Özet rapor oluştur
    report_path = "results/simple_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MEME KANSERİ SINIFLANDIRMA - BASİT ANALİZ RAPORU\\n")
        f.write("=" * 60 + "\\n\\n")
        
        f.write(f"Veri Boyutu: {data.shape}\\n")
        f.write(f"Özellik Sayısı: {X.shape[1]}\\n")
        f.write(f"Eğitim Seti: {X_train.shape[0]} örnek\\n")
        f.write(f"Test Seti: {X_test.shape[0]} örnek\\n\\n")
        
        f.write("MODEL PERFORMANSLARI:\\n")
        f.write("-" * 30 + "\\n")
        for name, result in sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True):
            f.write(f"{name:<20}: Doğruluk={result['accuracy']:.4f}, AUC={result['auc']:.4f}\\n")
        
        f.write(f"\\nEN İYİ MODEL: {best_model_name}\\n")
        f.write(f"Doğruluk: {results[best_model_name]['accuracy']:.4f}\\n")
        f.write(f"AUC: {results[best_model_name]['auc']:.4f}\\n")
    
    print(f"\\n📄 Rapor kaydedildi: {report_path}")
    print("\\n✅ Analiz tamamlandı!")
    print("📁 Sonuçlar 'results/' klasöründe")

if __name__ == "__main__":
    main()
