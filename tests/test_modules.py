"""
Test module - Data processing tests
"""
import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add main directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDataProcessor(unittest.TestCase):
    """Test class for DataProcessor class"""
    
    def setUp(self):
        """Pre-test setup"""
        # Create test data
        self.test_data = pd.DataFrame({
            'diagnosis': ['M', 'B', 'M', 'B', 'M'],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'id': [1, 2, 3, 4, 5]
        })
        
        # Create test CSV file
        self.test_csv_path = 'test_data.csv'
        self.test_data.to_csv(self.test_csv_path, index=False)
    
    def tearDown(self):
        """Post-test cleanup"""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
    
    def test_data_loading(self):
        """Data loading test"""
        # Import done here because path is added in setUp
        try:
            from src.data_processor import DataProcessor
            processor = DataProcessor(self.test_csv_path)
            data = processor.load_data()
            
            self.assertEqual(data.shape, (5, 4))
            self.assertIn('diagnosis', data.columns)
        except ImportError:
            self.skipTest("DataProcessor cannot be imported")
    
    def test_preprocessing(self):
        """Preprocessing test"""
        try:
            from src.data_processor import DataProcessor
            processor = DataProcessor(self.test_csv_path)
            processor.load_data()
            X, y = processor.preprocess_data()
            
            # Is target variable correctly encoded?
            expected_y = [0, 1, 0, 1, 0]  # M=0, B=1
            np.testing.assert_array_equal(y.values, expected_y)
            
            # Is feature count correct?
            self.assertEqual(X.shape[1], 2)  # feature1, feature2
        except ImportError:
            self.skipTest("DataProcessor cannot be imported")
    
    def test_data_split(self):
        """Data split test"""
        try:
            from src.data_processor import DataProcessor
            processor = DataProcessor(self.test_csv_path)
            processor.load_data()
            processor.preprocess_data()
            
            X_train, X_test, y_train, y_test = processor.split_data(test_size=0.4)
            
            # Are dimensions correct?
            total_samples = len(processor.data)
            test_samples = int(total_samples * 0.4)
            train_samples = total_samples - test_samples
            
            self.assertEqual(len(X_train), train_samples)
            self.assertEqual(len(X_test), test_samples)
        except ImportError:
            self.skipTest("DataProcessor cannot be imported")

class TestModelTrainer(unittest.TestCase):
    """Test class for ModelTrainer class"""
    
    def setUp(self):
        """Pre-test setup"""
        self.X_train = np.random.rand(100, 5)
        self.X_test = np.random.rand(20, 5)
        self.y_train = np.random.randint(0, 2, 100)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_model_initialization(self):
        """Model initialization test"""
        try:
            from src.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            models = trainer.initialize_models()
            
            # Should have at least 4 models
            self.assertGreaterEqual(len(models), 4)
            self.assertIn('Random Forest', models)
            self.assertIn('Logistic Regression', models)
        except ImportError:
            self.skipTest("ModelTrainer cannot be imported")
    
    def test_model_training(self):
        """Model training test"""
        try:
            from src.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            trainer.initialize_models()
            
            results = trainer.train_models(
                self.X_train, self.y_train, 
                self.X_test, self.y_test
            )
            
            # Are there results?
            self.assertGreater(len(results), 0)
            
            # Are there metrics for each model?
            for name, result in results.items():
                self.assertIn('metrics', result)
                self.assertIn('accuracy', result['metrics'])
        except ImportError:
            self.skipTest("ModelTrainer cannot be imported")

if __name__ == '__main__':
    # Report test results
    print("🧪 Test Process Starting...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTrainer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary report
    print("\\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print(f"• Number of tests run: {result.testsRun}")
    print(f"• Passed: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"• Failed: {len(result.failures)}")
    print(f"• Error: {len(result.errors)}")
    print(f"• Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful() and len(result.skipped) == 0:
        print("\\n✅ All tests passed successfully!")
    elif len(result.skipped) > 0:
        print("\\n⚠️ Some tests were skipped due to import errors.")
    else:
        print("\\n❌ Some tests failed.")
