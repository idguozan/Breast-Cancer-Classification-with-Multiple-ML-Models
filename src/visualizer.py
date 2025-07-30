"""
Visualization module
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Matplotlib and Seaborn settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

class Visualizer:
    """Data visualization class"""
    
    def __init__(self, save_plots: bool = True, output_dir: str = "results"):
        """
        Visualizer class constructor
        
        Args:
            save_plots (bool): Whether to save plots
            output_dir (str): Output directory
        """
        self.save_plots = save_plots
        self.output_dir = output_dir
        
    def plot_data_overview(self, data: pd.DataFrame, target_column: str = 'target') -> None:
        """
        Plots data overview
        
        Args:
            data (pd.DataFrame): Dataset
            target_column (str): Target variable column
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Target variable distribution
        target_counts = data[target_column].value_counts()
        axes[0, 0].pie(target_counts.values, labels=['Malignant', 'Benign'], 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Target Variable Distribution')
        
        # Numeric columns distribution
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            axes[0, 1].hist(numeric_data.std(), bins=20, edgecolor='black')
            axes[0, 1].set_title('Feature Standard Deviation Distribution')
            axes[0, 1].set_xlabel('Standard Deviation')
            axes[0, 1].set_ylabel('Frequency')
        
        # Correlation matrix summary
        corr_matrix = numeric_data.corr()
        high_corr = np.where(np.abs(corr_matrix) > 0.8)
        high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j]) 
                          for i, j in zip(*high_corr) if i != j]
        
        axes[1, 0].text(0.1, 0.5, f'High correlation feature pairs: {len(high_corr_pairs)//2}', 
                       fontsize=12, transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Correlation Analysis Summary')
        axes[1, 0].axis('off')
        
        # Missing value analysis
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            missing_data = missing_data[missing_data > 0]
            axes[1, 1].bar(range(len(missing_data)), missing_data.values)
            axes[1, 1].set_xticks(range(len(missing_data)))
            axes[1, 1].set_xticklabels(missing_data.index, rotation=45)
            axes[1, 1].set_title('Missing Values')
            axes[1, 1].set_ylabel('Missing Value Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'No missing values!', 
                           ha='center', va='center', fontsize=14, 
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Missing Value Analysis')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/data_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plots correlation matrix
        
        Args:
            data (pd.DataFrame): Dataset
            figsize (Tuple[int, int]): Figure size
        """
        # Get only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] > 20:
            # If too many features, show only highest correlations
            corr_with_target = numeric_data.corrwith(numeric_data.iloc[:, -1]).abs()
            top_features = corr_with_target.nlargest(20).index
            numeric_data = numeric_data[top_features]
        
        corr_matrix = numeric_data.corr()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, results: Dict[str, Dict], figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Draws individual ROC curve for each model
        
        Args:
            results (Dict[str, Dict]): Model results
            figsize (Tuple[int, int]): Figure size
        """
        # Determine number of models
        valid_models = []
        for name, result in results.items():
            if 'y_prob' in result and result['y_prob'] is not None:
                valid_models.append((name, result))
        
        if not valid_models:
            logger.warning("No models found that can plot ROC curves")
            return
        
        # Create separate subplot for each model
        n_models = len(valid_models)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        
        # Make axes a list if single model
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten() if n_models > 1 else axes
        
        for idx, (name, result) in enumerate(valid_models):
            # Get y_test from results or assign default value
            if 'y_test' in result:
                y_test = result['y_test']
            else:
                # If y_test not in results, skip
                logger.warning(f"y_test not found for {name}, ROC curve cannot be plotted")
                continue
            
            fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
            roc_auc = auc(fpr, tpr)
            
            ax = axes_flat[idx]
            ax.plot(fpr, tpr, linewidth=3, color='blue', 
                   label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, 
                   label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (FPR)')
            ax.set_ylabel('True Positive Rate (TPR)')
            ax.set_title(f'{name} - ROC Curve')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/individual_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves_comparison(self, results: Dict[str, Dict], figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Draws ROC curves for comparison (legacy version)
        
        Args:
            results (Dict[str, Dict]): Model results
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)
        
        for name, result in results.items():
            if 'y_prob' in result and result['y_prob'] is not None:
                # Get y_test from results or assign default value
                if 'y_test' in result:
                    y_test = result['y_test']
                else:
                    # If y_test not in results, skip
                    logger.warning(f"y_test not found for {name}, ROC curve cannot be plotted")
                    continue
                
                fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, results: Dict[str, Dict], y_test: np.ndarray) -> None:
        """
        Draws confusion matrices
        
        Args:
            results (Dict[str, Dict]): Model results
            y_test (np.ndarray): Test target variable
        """
        n_models = len(results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        model_names = list(results.keys())
        
        for idx, (name, result) in enumerate(results.items()):
            row = idx // cols
            col = idx % cols
            
            if rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]
            
            y_pred = result['y_pred']
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Malignant', 'Benign'],
                       yticklabels=['Malignant', 'Benign'])
            
            ax.set_title(f'{name}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Boş subplot'ları gizle
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.Series, top_n: int = 15,
                              figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Özellik önemlerini çizer
        
        Args:
            feature_importance (pd.Series): Özellik önemleri
            top_n (int): Gösterilecek en önemli özellik sayısı
            figsize (Tuple[int, int]): Grafik boyutu
        """
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=figsize)
        
        # Horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_features.values, color=colors)
        
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Importance Score')
        plt.title(f'En Önemli {top_n} Özellik')
        plt.gca().invert_yaxis()
        
        # Değerleri bar'ların üzerine yaz
        for i, (bar, value) in enumerate(zip(bars, top_features.values)):
            plt.text(value + 0.001, i, f'{value:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results_df: pd.DataFrame, 
                            figsize: Tuple[int, int] = (14, 8)) -> None:
        """
        Model performanslarını karşılaştırmalı olarak çizer
        
        Args:
            results_df (pd.DataFrame): Model sonuçları DataFrame'i
            figsize (Tuple[int, int]): Grafik boyutu
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for idx, metric in enumerate(available_metrics[:4]):
            if idx < len(axes):
                ax = axes[idx]
                
                bars = ax.bar(results_df['Model'], results_df[metric], 
                            color=plt.cm.Set3(np.linspace(0, 1, len(results_df))))
                
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                
                # Değerleri bar'ların üzerine yaz
                for bar, value in zip(bars, results_df[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Boş subplot'ları gizle
        for idx in range(len(available_metrics), 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, results: Dict[str, Dict], 
                                   results_df: pd.DataFrame) -> None:
        """
        Plotly ile interaktif dashboard oluşturur
        
        Args:
            results (Dict[str, Dict]): Model sonuçları
            results_df (pd.DataFrame): Model sonuçları DataFrame'i
        """
        # Subplot'lar oluştur
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(['Model Accuracy Comparison', 'AUC Scores',
                           'Precision vs Recall', 'F1-Score Distribution']),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # 1. Accuracy comparison
        fig.add_trace(
            go.Bar(x=results_df['Model'], y=results_df['Accuracy'],
                  name='Accuracy', marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. AUC scores
        if 'AUC' in results_df.columns:
            auc_data = results_df[results_df['AUC'] != 'N/A']
            if not auc_data.empty:
                fig.add_trace(
                    go.Bar(x=auc_data['Model'], y=auc_data['AUC'].astype(float),
                          name='AUC', marker_color='lightgreen'),
                    row=1, col=2
                )
        
        # 3. Precision vs Recall
        fig.add_trace(
            go.Scatter(x=results_df['Precision'], y=results_df['Recall'],
                      mode='markers+text', text=results_df['Model'],
                      textposition="top center", name='Models',
                      marker=dict(size=10, color='red')),
            row=2, col=1
        )
        
        # 4. F1-Score histogram
        fig.add_trace(
            go.Histogram(x=results_df['F1-Score'], name='F1-Score Distribution',
                        marker_color='orange'),
            row=2, col=2
        )
        
        # Layout güncellemeleri
        fig.update_layout(
            title_text="Meme Kanseri Sınıflandırma - Model Performans Dashboard",
            showlegend=False,
            height=800
        )
        
        # Eksen etiketlerini güncelle
        fig.update_xaxes(title_text="Modeller", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        
        if 'AUC' in results_df.columns:
            fig.update_xaxes(title_text="Modeller", row=1, col=2)
            fig.update_yaxes(title_text="AUC", row=1, col=2)
        
        fig.update_xaxes(title_text="Precision", row=2, col=1)
        fig.update_yaxes(title_text="Recall", row=2, col=1)
        
        fig.update_xaxes(title_text="F1-Score", row=2, col=2)
        fig.update_yaxes(title_text="Frekans", row=2, col=2)
        
        # HTML olarak kaydet
        if self.save_plots:
            fig.write_html(f'{self.output_dir}/interactive_dashboard.html')
            logger.info("İnteraktif dashboard 'interactive_dashboard.html' olarak kaydedildi")
        
        fig.show()
    
    def plot_learning_curves(self, model, X_train: np.ndarray, y_train: np.ndarray,
                           model_name: str = "Model") -> None:
        """
        Öğrenme eğrilerini çizer
        
        Args:
            model: Scikit-learn modeli
            X_train: Training features
            y_train: Training target variable
            model_name (str): Model adı
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                         alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                         alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'{model_name} - Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/learning_curves_{model_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
