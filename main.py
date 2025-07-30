"""
Breast Cancer Classification - Main Application
Advanced machine learning models for breast cancer diagnosis
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Import project modules
import importlib.util

def load_module(name, path):
    """Dynamically loads a module"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
try:
    dp_module = load_module("data_processor", "src/data_processor.py")
    DataProcessor = dp_module.DataProcessor
    
    mt_module = load_module("model_trainer", "src/model_trainer.py")
    ModelTrainer = mt_module.ModelTrainer
    
    viz_module = load_module("visualizer", "src/visualizer.py")
    Visualizer = viz_module.Visualizer
    
    print("✅ All modules loaded successfully!")
    
except Exception as e:
    print(f"❌ Module loading error: {e}")
    print("⚠️  Use 'run_simple_analysis.py' for simple analysis")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/breast_cancer_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class BreastCancerAnalysis:
    """Main breast cancer analysis class"""
    
    def __init__(self, data_path: str = "data/breast-cancer.csv", 
                 results_dir: str = "results"):
        """
        Constructor
        
        Args:
            data_path (str): Data file path
            results_dir (str): Results directory
        """
        self.data_path = data_path
        self.results_dir = results_dir
        
        # Create results directory
        Path(self.results_dir).mkdir(exist_ok=True)
        
        # Initialize classes
        self.data_processor = DataProcessor(self.data_path)
        self.model_trainer = ModelTrainer()
        self.visualizer = Visualizer(save_plots=True, output_dir=self.results_dir)
        
        logger.info("Breast Cancer Analysis initialized")
    
    def run_complete_analysis(self):
        """Runs complete analysis"""
        
        logger.info("=" * 60)
        logger.info("BREAST CANCER CLASSIFICATION ANALYSIS STARTING")
        logger.info("=" * 60)
        
        # 1. Data Loading and Preprocessing
        logger.info("\n1️⃣ DATA LOADING AND PREPROCESSING")
        logger.info("-" * 40)
        
        # Load data
        data = self.data_processor.load_data()
        
        # Get data information
        data_info = self.data_processor.get_data_info()
        logger.info(f"Data shape: {data_info['shape']}")
        logger.info(f"Number of columns: {len(data_info['columns'])}")
        
        # Preprocessing
        X, y = self.data_processor.preprocess_data()
        
        # Data visualization (with target)
        data_with_target = data.copy()
        data_with_target['target'] = data['diagnosis'].map({'M': 0, 'B': 1})
        self.visualizer.plot_data_overview(data_with_target, target_column='target')
        
        # Correlation matrix
        combined_data = X.copy()
        combined_data['target'] = y
        self.visualizer.plot_correlation_matrix(combined_data)
        
        # Split data
        X_train, X_test, y_train, y_test = self.data_processor.split_data(
            test_size=0.2, scale_features=True
        )
        
        # 2. Model Training
        logger.info("\n2️⃣ MODEL TRAINING")
        logger.info("-" * 40)
        
        # Initialize models
        self.model_trainer.initialize_models()
        
        # Train models
        results = self.model_trainer.train_models(X_train, y_train, X_test, y_test)
        
        # Add y_test to results (for visualization)
        for name in results:
            results[name]['y_test'] = y_test
        
        # 3. Model Evaluation
        logger.info("\n3️⃣ MODEL EVALUATION")
        logger.info("-" * 40)
        
        # Results summary
        results_df = self.model_trainer.get_results_summary()
        print("\n📊 MODEL PERFORMANCE SUMMARY:")
        print("=" * 80)
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Determine and save best model
        best_model_name = self.model_trainer.best_model_name
        if best_model_name:
            logger.info(f"\n🏆 Best Model: {best_model_name}")
            
            # Save best model
            self.model_trainer.save_model(
                best_model_name, 
                f"{self.results_dir}/best_model_{best_model_name.replace(' ', '_')}.joblib"
            )
        
        # 4. Visualizations
        logger.info("\n4️⃣ VISUALIZATIONS")
        logger.info("-" * 40)
        
        # ROC curves
        self.visualizer.plot_roc_curves(results)
        
        # Confusion matrices
        self.visualizer.plot_confusion_matrices(results, y_test)
        
        # Model comparison
        self.visualizer.plot_model_comparison(results_df)
        
        # Feature importance (for best model)
        if best_model_name and hasattr(self.model_trainer.best_model, 'feature_importances_'):
            try:
                # Get feature names from X
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                importance_series = pd.Series(
                    self.model_trainer.best_model.feature_importances_, 
                    index=feature_names
                ).sort_values(ascending=False)
                
                self.visualizer.plot_feature_importance(importance_series)
                
                logger.info(f"\n🔍 {best_model_name} - Top 5 Important Features:")
                for idx, (feature, importance) in enumerate(importance_series.head(5).items(), 1):
                    logger.info(f"  {idx}. {feature}: {importance:.4f}")
                    
            except Exception as e:
                logger.warning(f"Feature importance plotting failed: {e}")
        
        # Interactive dashboard
        try:
            self.visualizer.create_interactive_dashboard(results, results_df)
        except Exception as e:
            logger.warning(f"Interactive dashboard creation failed: {e}")
        
        # 5. Hyperparameter Optimization (for top 2 models)
        logger.info("\n5️⃣ HYPERPARAMETER OPTIMIZATION")
        logger.info("-" * 40)
        
        self._perform_hyperparameter_tuning(X_train, y_train, X_test, y_test, results_df)
        
        # 6. Learning Curves
        logger.info("\n6️⃣ LEARNING CURVES")
        logger.info("-" * 40)
        
        if best_model_name:
            self.visualizer.plot_learning_curves(
                self.model_trainer.best_model, 
                X_train, y_train, 
                best_model_name
            )
        
        # 7. Final Report
        self._generate_final_report(results_df, best_model_name)
        
        # 8. Generate PDF Report
        logger.info("\n📄 GENERATING PDF REPORT")
        logger.info("-" * 40)
        self._generate_pdf_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETED! ✅")
        logger.info(f"All results saved in '{self.results_dir}' directory.")
        logger.info("=" * 60)
    
    def _perform_hyperparameter_tuning(self, X_train, y_train, X_test, y_test, results_df):
        """Performs hyperparameter optimization"""
        
        # Select top 2 models
        top_models = results_df.head(2)['Model'].tolist()
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1],
                'kernel': ['rbf', 'linear']
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [2000, 3000, 5000]
            }
        }
        
        for model_name in top_models:
            if model_name in param_grids:
                try:
                    logger.info(f"🔧 Hyperparameter optimization for {model_name}...")
                    
                    optimized_model = self.model_trainer.hyperparameter_tuning(
                        model_name, param_grids[model_name], X_train, y_train
                    )
                    
                    # Test optimized model
                    y_pred_opt = optimized_model.predict(X_test)
                    accuracy_opt = np.mean(y_pred_opt == y_test)
                    
                    logger.info(f"✅ {model_name} optimized - New accuracy: {accuracy_opt:.4f}")
                    
                except Exception as e:
                    logger.warning(f"❌ {model_name} optimization failed: {e}")
    
    def _generate_final_report(self, results_df, best_model_name):
        """Generates final report"""
        
        report_path = f"{self.results_dir}/final_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BREAST CANCER CLASSIFICATION ANALYSIS - FINAL REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {self.data_path}\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(results_df.to_string(index=False, float_format='%.4f'))
            f.write("\n\n")
            
            if best_model_name:
                best_results = results_df[results_df['Model'] == best_model_name].iloc[0]
                f.write(f"BEST MODEL: {best_model_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {best_results['Accuracy']:.4f}\n")
                f.write(f"Precision: {best_results['Precision']:.4f}\n")
                f.write(f"Recall: {best_results['Recall']:.4f}\n")
                f.write(f"F1-Score: {best_results['F1-Score']:.4f}\n")
                if best_results['AUC'] != 'N/A':
                    f.write(f"AUC: {best_results['AUC']:.4f}\n")
                f.write("\n")
            
            f.write("RESULTS:\n")
            f.write("-" * 15 + "\n")
            f.write("• All models successfully trained and evaluated.\n")
            f.write("• ROC curves and confusion matrices generated.\n")
            f.write("• Feature importance analyzed.\n")
            f.write("• Hyperparameter optimization performed.\n")
            f.write("• Interactive dashboard created.\n\n")
            
            f.write("OUTPUT FILES:\n")
            f.write("-" * 20 + "\n")
            f.write("• data_overview.png - Data overview\n")
            f.write("• correlation_matrix.png - Correlation matrix\n")
            f.write("• individual_roc_curves.png - Individual ROC curves\n")
            f.write("• confusion_matrices.png - Confusion matrices\n")
            f.write("• model_comparison.png - Model comparison\n")
            f.write("• feature_importance.png - Feature importance\n")
            f.write("• interactive_dashboard.html - Interactive dashboard\n")
            if best_model_name:
                f.write(f"• best_model_{best_model_name.replace(' ', '_')}.joblib - Best model\n")
        
        logger.info(f"📄 Final report saved to '{report_path}'")

    def _generate_pdf_report(self):
        """Generates PDF report"""
        try:
            # Import PDF generator
            import subprocess
            import sys
            
            # Try to run PDF generator
            result = subprocess.run([sys.executable, "generate_pdf_report.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ PDF report generated successfully!")
                logger.info("📄 Report saved as: results/breast_cancer_analysis_report.pdf")
            else:
                logger.warning("⚠️ PDF generation failed. Check if reportlab is installed.")
                logger.info("📦 Install with: pip install reportlab Pillow")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not generate PDF report: {e}")
            logger.info("📦 Install dependencies with: pip install reportlab Pillow")

def main():
    """Main function"""
    
    # Check data file
    data_path = "data/breast-cancer.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please place breast-cancer.csv file in 'data/' folder.")
        return
    
    try:
        # Start analysis
        analysis = BreastCancerAnalysis(data_path=data_path)
        analysis.run_complete_analysis()
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
