# """
# train_model.py
# ---------------
# This module trains a machine learning model to predict
# password strength using extracted numerical features.

# The trained model is saved to disk and later used by
# the web application for real-time predictions.
# """

# import os
# import joblib
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# from src.data_loader import DataLoader
# from src.preprocessing import DataPreprocessor
# from src.feature_extraction import FeatureExtractor



# class ModelTrainer:
#     """
#     ModelTrainer handles training, evaluation, and saving
#     of the password strength ML model.
#     """

#     def __init__(self, dataset_path, model_dir="models"):
#         """
#         Constructor

#         :param dataset_path: Path to cleaned or raw dataset
#         :param model_dir: Directory to save trained model
#         """
#         self.dataset_path = dataset_path
#         self.model_dir = model_dir

#         os.makedirs(self.model_dir, exist_ok=True)

#         self.model = RandomForestClassifier(
#             n_estimators=200,
#             max_depth=None,
#             random_state=42
#         )

#     def load_and_prepare_data(self):
#         """
#         Loads dataset and performs preprocessing
#         """
#         print("\n📥 Loading dataset...")
#         loader = DataLoader(self.dataset_path)
#         df = loader.load_data()

#         print("\n🧹 Preprocessing dataset...")
#         preprocessor = DataPreprocessor(df)
#         df_cleaned = preprocessor.preprocess(
#             target_column="Strength",
#             required_columns=[
#                 "Password",
#                 "Has Lowercase",
#                 "Has Uppercase",
#                 "Has Special Character",
#                 "Length",
#                 "Strength"
#             ]
#         )

#         return df_cleaned

#     def extract_features(self, df):
#         """
#         Extracts features and labels from dataset
#         """
#         print("\n🔢 Extracting features...")
#         extractor = FeatureExtractor()
#         X = extractor.extract_from_dataframe(df)
#         y = df["Strength"].values

#         return X, y

#     def split_data(self, X, y, test_size=0.2):
#         """
#         Splits dataset into training and testing sets
#         """
#         print("\n✂️ Splitting dataset...")
#         return train_test_split(
#             X,
#             y,
#             test_size=test_size,
#             random_state=42,
#             stratify=y
#         )

#     def train(self, X_train, y_train):
#         """
#         Trains the ML model
#         """
#         print("\n🤖 Training model...")
#         self.model.fit(X_train, y_train)
#         print("✅ Model training completed")

#     def evaluate(self, X_test, y_test):
#         """
#         Evaluates model performance
#         """
#         print("\n📊 Evaluating model...")
#         y_pred = self.model.predict(X_test)

#         acc = accuracy_score(y_test, y_pred)
#         report = classification_report(y_test, y_pred)
#         matrix = confusion_matrix(y_test, y_pred)

#         print(f"\n🎯 Accuracy: {acc:.4f}")
#         print("\n📄 Classification Report:\n", report)
#         print("\n🧩 Confusion Matrix:\n", matrix)

#         return acc, report, matrix

#     def save_model(self, filename="password_strength_model.pkl"):
#         """
#         Saves trained model to disk
#         """
#         model_path = os.path.join(self.model_dir, filename)
#         joblib.dump(self.model, model_path)
#         print(f"\n💾 Model saved at: {model_path}")

#     def run(self):
#         """
#         Full training pipeline
#         """
#         df = self.load_and_prepare_data()
#         X, y = self.extract_features(df)
#         X_train, X_test, y_train, y_test = self.split_data(X, y)

#         self.train(X_train, y_train)
#         self.evaluate(X_test, y_test)
#         self.save_model()


# # -------------------- MAIN EXECUTION --------------------

# if __name__ == "__main__":
#     trainer = ModelTrainer(
#         dataset_path="/home/kali/Music/hello/Password-Strength-Analyzer/dataset/raw/passwords_raw.csv"
#     )
#     trainer.run()








"""
train_model.py
---------------
Trains the ML model for SecurePass AI
Password Strength Analyzer.

Outputs:
- Trained ML model (.pkl)
- Feature order consistency
- Confidence-ready classifier
"""

import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import DataLoader

from preprocessing import DataPreprocessor
from feature_extraction import FeatureExtractor


class ModelTrainer:

    def __init__(self, dataset_path, model_dir="models"):
        self.dataset_path = dataset_path
        self.model_dir = model_dir

        os.makedirs(self.model_dir, exist_ok=True)

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced"
        )

        self.extractor = FeatureExtractor()

    def load_and_prepare_data(self):
        print("\nLoading dataset...")
        loader = DataLoader(self.dataset_path)
        df = loader.load_data()

        print("\nPreprocessing dataset...")
        preprocessor = DataPreprocessor(df)
        df_cleaned = preprocessor.preprocess(
            target_column="strength",
            required_columns=["password", "strength"]
        )

        return df_cleaned

    def extract_features(self, df):
        print("\nExtracting features...")
        X = self.extractor.extract_from_dataframe(df, password_column="password")
        y = df["strength"].values
        return X, y

    def split_data(self, X, y):
        print("\nSplitting dataset...")
        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    def train(self, X_train, y_train):
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        print("Model training completed")

    def evaluate(self, X_test, y_test):
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {acc * 100:.2f}%")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return acc

    def save_model(self):
        model_path = os.path.join(self.model_dir, "password_strength_model.pkl")

        model_bundle = {
            "model": self.model,
            "feature_names": self.extractor.get_feature_names(),
            "label_mapping": {
                0: "Weak",
                1: "Medium",
                2: "Strong",
                3: "Unbreakable"
            }
        }

        joblib.dump(model_bundle, model_path)
        print(f"\nModel saved at: {model_path}")

    def run(self):
        df = self.load_and_prepare_data()
        X, y = self.extract_features(df)

        X_train, X_test, y_train, y_test = self.split_data(X, y)

        self.train(X_train, y_train)
        self.evaluate(X_test, y_test)
        self.save_model()


#                       MAIN EXECUTION 
if __name__ == "__main__":
    trainer = ModelTrainer(
        dataset_path="/home/kali/Music/hello/Password-Strength-Analyzer/dataset/raw/passwords_raw.csv"
    )
    trainer.run()
