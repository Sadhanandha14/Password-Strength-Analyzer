
import os
import joblib
import numpy as np

from src.feature_extraction import FeatureExtractor

class PasswordStrengthPredictor:
  
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

        
        features = self.feature_extractor.extract_from_password(password)
        X = np.array(features).reshape(1, -1)
      
        pred_class = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        confidence = round(float(np.max(proba)) * 100, 2)

        strength_label = self.label_mapping[pred_class]

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


        score = min(100, int(entropy_bits))

        return {
            "strength": strength_label,
            "confidence": confidence,
            "entropy_bits": round(entropy_bits, 2),
            "time_to_crack": self._estimate_time_to_crack(entropy_bits),
            "score": score,
            "explanation": explanation
        }



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
