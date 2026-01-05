

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
