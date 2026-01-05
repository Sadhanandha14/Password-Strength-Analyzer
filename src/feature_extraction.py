# """
# feature_extraction.py
# ----------------------
# This module is responsible for converting password text
# into numerical features suitable for machine learning models.

# NOTE:
# - This file performs feature engineering, NOT rule-based validation.
# - It does not classify passwords.
# - It only extracts measurable properties from the password.
# """

# import numpy as np
# import pandas as pd
# import math


# class FeatureExtractor:
#     """
#     FeatureExtractor class converts password strings
#     into numeric feature vectors for ML models.
#     """

#     def __init__(self):
#         """
#         Constructor for FeatureExtractor
#         """
#         pass

#     def _shannon_entropy(self, password):
#         """
#         Calculates Shannon entropy of a password

#         :param password: Password string
#         :return: Entropy value
#         """
#         if not password:
#             return 0.0

#         char_set = set(password)
#         entropy = 0.0

#         for char in char_set:
#             probability = password.count(char) / len(password)
#             entropy -= probability * math.log2(probability)

#         return entropy

#     def extract_from_password(self, password):
#         """
#         Extracts numerical features from a single password

#         :param password: Password string
#         :return: Numpy array of features
#         """

#         if not isinstance(password, str):
#             raise TypeError("Password must be a string")

#         length = len(password)
#         lowercase_count = sum(char.islower() for char in password)
#         uppercase_count = sum(char.isupper() for char in password)
#         digit_count = sum(char.isdigit() for char in password)
#         special_count = sum(not char.isalnum() for char in password)

#         entropy = self._shannon_entropy(password)

#         features = np.array([
#             length,
#             lowercase_count,
#             uppercase_count,
#             digit_count,
#             special_count,
#             entropy
#         ], dtype=float)

#         return features

#     def extract_from_dataframe(self, df, password_column="Password"):
#         """
#         Extracts features for all passwords in a DataFrame

#         :param df: Pandas DataFrame
#         :param password_column: Column name containing passwords
#         :return: Feature matrix (X)
#         """

#         if not isinstance(df, pd.DataFrame):
#             raise TypeError("Input must be a Pandas DataFrame")

#         if password_column not in df.columns:
#             raise ValueError(
#                 f"Password column '{password_column}' not found in dataset"
#             )

#         feature_list = []

#         for password in df[password_column]:
#             features = self.extract_from_password(password)
#             feature_list.append(features)

#         X = np.array(feature_list)

#         print("✅ Feature extraction completed")
#         print(f"📐 Feature matrix shape: {X.shape}")

#         return X

#     def get_feature_names(self):
#         """
#         Returns list of feature names (for understanding & debugging)
#         """
#         return [
#             "length",
#             "lowercase_count",
#             "uppercase_count",
#             "digit_count",
#             "special_char_count",
#             "shannon_entropy"
#         ]


# # -------------------- TESTING SECTION --------------------
# # This block runs only when this file is executed directly

# if __name__ == "__main__":
#     from data_loader import DataLoader
#     from preprocessing import DataPreprocessor

#     # Load dataset
#     loader = DataLoader("/home/kali/Music/hello/Password-Strength-Analyzer/dataset/raw/passwords_raw.csv")
#     df = loader.load_data()

#     # Preprocess dataset
#     preprocessor = DataPreprocessor(df)
#     cleaned_df = preprocessor.preprocess(
#         target_column="Strength",
#         required_columns=[
#             "Password",
#             "Has Lowercase",
#             "Has Uppercase",
#             "Has Special Character",
#             "Length",
#             "Strength"
#         ]
#     )

#     # Extract features
#     extractor = FeatureExtractor()
#     X = extractor.extract_from_dataframe(cleaned_df)

#     print("\n🔍 Feature Names:")
#     print(extractor.get_feature_names())

#     print("\n🔢 Sample Feature Vector:")
#     print(X[0])























import numpy as np
import pandas as pd
import math
import string


class FeatureExtractor:

    def __init__(self):

        self.lowercase = set(string.ascii_lowercase)
        self.uppercase = set(string.ascii_uppercase)
        self.digits = set(string.digits)
        self.special = set(string.punctuation)

    def _calculate_entropy_bits(self, password: str) -> float:

        if not password:
            return 0.0

        charset_size = 0

        if any(c in self.lowercase for c in password):
            charset_size += 26
        if any(c in self.uppercase for c in password):
            charset_size += 26
        if any(c in self.digits for c in password):
            charset_size += 10
        if any(c in self.special for c in password):
            charset_size += len(self.special)

        if charset_size == 0:
            return 0.0

        entropy = math.log2(charset_size ** len(password))
        return round(entropy, 2)

    def extract_from_password(self, password: str) -> np.ndarray:
        if not isinstance(password, str):
            raise TypeError("Password must be a string")

        length = len(password)

        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)

        lower_count = sum(c.islower() for c in password)
        upper_count = sum(c.isupper() for c in password)
        digit_count = sum(c.isdigit() for c in password)
        special_count = sum(not c.isalnum() for c in password)

        entropy_bits = self._calculate_entropy_bits(password)

        features = np.array([
            length,
            lower_count,
            upper_count,
            digit_count,
            special_count,
            int(has_lower),
            int(has_upper),
            int(has_digit),
            int(has_special),
            entropy_bits
        ], dtype=float)

        return features

    def extract_from_dataframe(self, df: pd.DataFrame, password_column="password") -> np.ndarray:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame")

        if password_column not in df.columns:
            raise ValueError(f"Column '{password_column}' not found")

        feature_matrix = []

        for pwd in df[password_column]:
            feature_matrix.append(self.extract_from_password(pwd))

        X = np.array(feature_matrix)

        print("Feature extraction completed")
        print(f"Feature matrix shape: {X.shape}")

        return X

    def get_feature_names(self):

        return [
            "length",
            "lowercase_count",
            "uppercase_count",
            "digit_count",
            "special_char_count",
            "has_lowercase",
            "has_uppercase",
            "has_digit",
            "has_special",
            "entropy_bits"
        ]



if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocessing import DataPreprocessor

    input_path = (
        "/home/kali/Music/hello/Password-Strength-Analyzer/"
        "dataset/raw/passwords_raw.csv"
    )

    loader = DataLoader(input_path)
    df = loader.load_data()

    preprocessor = DataPreprocessor(df)
    cleaned_df = preprocessor.preprocess(
        target_column="strength",
        required_columns=["password", "strength"]
    )

    extractor = FeatureExtractor()
    X = extractor.extract_from_dataframe(cleaned_df)

    print("\nFeature Names:")
    print(extractor.get_feature_names())

    print("\nSample Feature Vector:")
    print(X[0])
