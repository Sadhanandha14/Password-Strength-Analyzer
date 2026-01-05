# """
# predict_strength.py
# --------------------
# This module loads the trained machine learning model
# and predicts the strength of a given password.

# It is used by the web application (Flask) for
# real-time password strength prediction.

# NOTE:
# - This file DOES NOT use rule-based logic
# - Prediction is done purely using the trained ML model
# """

# import os
# import joblib
# import numpy as np

# from feature_extraction import FeatureExtractor


# class PasswordStrengthPredictor:
#     """
#     PasswordStrengthPredictor handles loading the trained model
#     and predicting password strength using ML.
#     """

#     def __init__(self, model_path="../models/password_strength_model.pkl"):
#         """
#         Constructor

#         :param model_path: Path to trained ML model (.pkl file)
#         """
#         self.model_path = model_path
#         self.model = None
#         self.feature_extractor = FeatureExtractor()

#         self._load_model()

#     def _load_model(self):
#         """
#         Loads the trained ML model from disk
#         """
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(
#                 f"Trained model not found at path: {self.model_path}"
#             )

#         try:
#             self.model = joblib.load(self.model_path)
#             print("✅ Trained model loaded successfully")
#         except Exception as e:
#             raise RuntimeError(f"Error loading model: {e}")

#     def predict(self, password):
#         """
#         Predicts the strength of a given password

#         :param password: Password string entered by user
#         :return: Predicted strength label (Weak / Medium / Strong)
#         """

#         if not isinstance(password, str):
#             raise TypeError("Password must be a string")

#         if password.strip() == "":
#             raise ValueError("Password cannot be empty")

#         # Extract numerical features from password
#         features = self.feature_extractor.extract_from_password(password)

#         # Reshape for model prediction (1 sample)
#         features = np.array(features).reshape(1, -1)

#         # Predict using ML model
#         prediction = self.model.predict(features)[0]

#         # Convert numeric label back to text
#         strength_label = self._decode_prediction(prediction)

#         return strength_label

#     def _decode_prediction(self, prediction):
#         """
#         Converts numeric prediction to readable label

#         Example:
#         0 -> Weak
#         1 -> Medium
#         2 -> Strong
#         """
#         label_mapping = {
#             0: "Weak",
#             1: "Medium",
#             2: "Strong"
#         }

#         return label_mapping.get(prediction, "Unknown")


# # -------------------- TESTING SECTION --------------------
# # This block runs only when this file is executed directly

# if __name__ == "__main__":
#     predictor = PasswordStrengthPredictor(
#         model_path="../models/password_strength_model.pkl"
#     )

#     test_passwords = [
#         "password",
#         "Pass@123",
#         "A9#xQ!7mZ@12"
#     ]

#     print("\n🔍 Password Strength Predictions:\n")

#     for pwd in test_passwords:
#         strength = predictor.predict(pwd)
#         print(f"Password: {pwd} -> Strength: {strength}")








#   BEST CODE

# """
# predict_strength.py
# --------------------
# This module loads the trained machine learning model
# and predicts the strength of a given password.

# It is used by the web application (Flask) for
# real-time password strength prediction.

# NEW FEATURE:
# - Explains why the password is Weak / Medium / Strong
#   based on its characteristics (length, uppercase, lowercase, digits, special chars, entropy)

# //NOTE:
# - Prediction is still purely ML-based
# """

# import os
# import joblib
# import numpy as np

# from src.feature_extraction import FeatureExtractor



# class PasswordStrengthPredictor:
#     """
#     PasswordStrengthPredictor handles loading the trained model
#     and predicting password strength using ML.
#     """

#     def __init__(self, model_path="../models/password_strength_model.pkl"):
#         self.model_path = model_path
#         self.model = None
#         self.feature_extractor = FeatureExtractor()
#         self._load_model()

#     def _load_model(self):
#         """Loads the trained ML model from disk"""
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(
#                 f"Trained model not found at path: {self.model_path}"
#             )
#         try:
#             self.model = joblib.load(self.model_path)
#             print("✅ Trained model loaded successfully")
#         except Exception as e:
#             raise RuntimeError(f"Error loading model: {e}")

#     def predict(self, password):
#         """
#         Predicts the strength of a password and provides explanation

#         :param password: Password string
#         :return: tuple (strength_label, explanation_list)
#         """
#         if not isinstance(password, str):
#             raise TypeError("Password must be a string")

#         if password.strip() == "":
#             raise ValueError("Password cannot be empty")

#         # Extract numerical features from password
#         features = self.feature_extractor.extract_from_password(password)
#         features = np.array(features).reshape(1, -1)

#         # Predict using ML model
#         prediction = self.model.predict(features)[0]
#         strength_label = self._decode_prediction(prediction)

#         # --- New Feature: Generate explanation ---
#         length, lowercase, uppercase, digits, special, entropy = features[0]
#         explanation = []

#         # Length explanation
#         if length < 8:
#             explanation.append("Password length is short (<8 characters)")
#         else:
#             explanation.append("Password length is good")

#         # Character variety
#         explanation.append("Contains lowercase letters" if lowercase > 0 else "No lowercase letters")
#         explanation.append("Contains uppercase letters" if uppercase > 0 else "No uppercase letters")
#         explanation.append("Contains numbers" if digits > 0 else "No numbers")
#         explanation.append("Contains special characters" if special > 0 else "No special characters")

#         # Entropy explanation
#         if entropy < 3:
#             explanation.append("Low entropy (predictable pattern)")
#         else:
#             explanation.append("Good entropy (hard to guess)")

#         return strength_label, explanation

#     def _decode_prediction(self, prediction):
#         """Converts numeric prediction to readable label"""
#         label_mapping = {
#             0: "Weak",
#             1: "Medium",
#             2: "Strong"
#         }
#         return label_mapping.get(prediction, "Unknown")


# # -------------------- TESTING SECTION --------------------
# if __name__ == "__main__":
#     predictor = PasswordStrengthPredictor(
#         model_path="../models/password_strength_model.pkl"
#     )

#     test_passwords = [
#         "password",
#         "Pass@123",
#         "A9#xQ!7mZ@12"
#     ]

#     print("\n🔍 Password Strength Predictions with Explanation:\n")

#     for pwd in test_passwords:
#         strength, explanation = predictor.predict(pwd)
#         print(f"Password: {pwd} -> Strength: {strength}")
#         print("Why:")
#         for line in explanation:
#             print(f" - {line}")
#         print()











"""
predict_strength.py
--------------------
Loads the trained ML model and predicts password strength
for SecurePass AI UI.

Features:
- Strength prediction (Weak / Medium / Strong / Unbreakable)
- ML confidence percentage
- Entropy (bits)
- Time-to-crack estimation
- Explanation WHY the password has that strength
"""






import os
import joblib
import numpy as np

from src.feature_extraction import FeatureExtractor


class PasswordStrengthPredictor:
    """
    Handles real-time password strength prediction
    """

    def __init__(self, model_path="models/password_strength_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.label_mapping = None
        self.feature_extractor = FeatureExtractor()
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.label_mapping = bundle["label_mapping"]

        print("ML model loaded successfully")

    def _estimate_time_to_crack(self, entropy_bits: float) -> str:
        """
        Rough estimation based on entropy bits
        """
        if entropy_bits < 5:
            return "Instantly"
        elif entropy_bits < 10:
            return "Minutes"
        elif entropy_bits < 30:
            return "Hours"
        elif entropy_bits < 50:
            return "weeks"
        elif entropy_bits < 80:
            return "Months"
        else:
            return "Years"

    def predict(self, password: str) -> dict:
        """
        Predict password strength and explanation
        """
        if not isinstance(password, str):
            raise TypeError("Password must be a string")

        if not password.strip():
            raise ValueError("Password cannot be empty")

        # Feature extraction
        features = self.feature_extractor.extract_from_password(password)
        X = np.array(features).reshape(1, -1)

        # ML prediction
        pred_class = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        confidence = round(float(np.max(proba)) * 100, 2)

        strength_label = self.label_mapping[pred_class]

        # Feature unpacking
        (
            length,
            lower_count,
            upper_count,
            digit_count,
            special_count,
            has_lower,
            has_upper,
            has_digit,
            has_special,
            entropy_bits
        ) = features

        # Explanation logic
        explanation = []

        if length < 8:
            explanation.append("Password is too short")
        elif length < 12:
            explanation.append("Password length is acceptable")
        else:
            explanation.append("Password length is strong")

        if has_upper:
            explanation.append("Used uppercase letters")
        else:
            explanation.append("Missing uppercase letters")

        if has_lower:
            explanation.append("Used Lowercase letters")
        else:
            explanation.append("Missing Lowercase letters")

        if has_digit:
            explanation.append("Used Numbers")
        else:
            explanation.append("Missing Numbers")

        if has_special:
            explanation.append("Used special characters")
        else:
            explanation.append("Missing special characters")

        if entropy_bits < 40:
            explanation.append("Low entropy (easy to guess)")
        elif entropy_bits < 70:
            explanation.append("Moderate entropy")
        else:
            explanation.append("High entropy (very hard to guess)")

        # Score (0–100)
        score = min(100, int(entropy_bits))

        return {
            "strength": strength_label,
            "confidence": confidence,
            "entropy_bits": round(entropy_bits, 2),
            "time_to_crack": self._estimate_time_to_crack(entropy_bits),
            "score": score,
            "explanation": explanation
        }


# -------------------- TESTING SECTION --------------------
if __name__ == "__main__":
    predictor = PasswordStrengthPredictor(
        model_path="models/password_strength_model.pkl"
    )

    test_passwords = [
        "password",
        "Pass@123",
        "A9#xQ!7mZ@12"
    ]

    print("\nSecurePass AI Predictions:\n")

    for pwd in test_passwords:
        result = predictor.predict(pwd)
        print(f"Password: {pwd}")
        for k, v in result.items():
            print(f"{k}: {v}")
        print("-" * 40)
