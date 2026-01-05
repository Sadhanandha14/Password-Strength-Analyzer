# """
# app.py
# -------
# Flask backend for Password Strength Analyzer using Machine Learning.

# This application:
# - Accepts password input from the website
# - Uses a trained ML model to predict password strength
# - Returns prediction result to the frontend

# NOTE:
# This is a machine learning–based system, NOT rule-based.
# """
# from src.feature_extraction import FeatureExtractor

# from flask import Flask, render_template, request
# from src.predict_strength import PasswordStrengthPredictor


# # -------------------------------------------------
# # Flask App Initialization
# # -------------------------------------------------

# app = Flask(__name__)

# # Load ML model once when app starts
# predictor = PasswordStrengthPredictor(
#     model_path="models/password_strength_model.pkl"
# )


# # -------------------------------------------------
# # Routes
# # -------------------------------------------------

# @app.route("/", methods=["GET", "POST"])
# def index():
#     """
#     Home page route.
#     Handles both GET and POST requests.
#     """
#     result = None
#     error = None
#     password_input = ""

#     if request.method == "POST":
#         try:
#             password_input = request.form.get("password")

#             if not password_input:
#                 raise ValueError("Password field cannot be empty")

#             # Predict password strength using ML model
#             result = predictor.predict(password_input)

#         except Exception as e:
#             error = str(e)

#     return render_template(
#         "index.html",
#         result=result,
#         error=error,
#         password=password_input
#     )


# # -------------------------------------------------
# # Application Entry Point
# # -------------------------------------------------

# if __name__ == "__main__":
#     app.run(
#         host="0.0.0.0",
#         port=5000,
#         debug=True
#     )






##   BEST CODE



# from src.feature_extraction import FeatureExtractor

# from flask import Flask, render_template, request
# from src.predict_strength import PasswordStrengthPredictor

# # -------------------------------------------------
# # Flask App Initialization
# # -------------------------------------------------

# app = Flask(__name__)

# # Load ML model once when app starts
# predictor = PasswordStrengthPredictor(
#     model_path="models/password_strength_model.pkl"
# )


# # -------------------------------------------------
# # Helper Function: Explain Strength
# # -------------------------------------------------

# def explain_strength(password, strength):
#     """
#     Returns a reason why the password is classified as Weak/Medium/Strong.
#     """
#     reasons = []

#     if len(password) < 6:
#         reasons.append("Password is too short")
#     if password.islower() or password.isupper():
#         reasons.append("Password lacks mixed case letters")
#     if password.isalpha():
#         reasons.append("Password has no numbers or special characters")
#     if password.isdigit():
#         reasons.append("Password has no letters or special characters")
#     if not any(c.isdigit() for c in password):
#         reasons.append("Password has no numbers")
#     if not any(not c.isalnum() for c in password):
#         reasons.append("Password has no special characters")

#     if strength == "Weak":
#         reasons.append("This password can be easily guessed or brute-forced")
#     elif strength == "Medium":
#         reasons.append("This password is moderately secure but could be stronger")
#     elif strength == "Strong":
#         reasons.append("This password is strong and hard to guess")

#     return "; ".join(reasons)


# # -------------------------------------------------
# # Routes
# # -------------------------------------------------

# @app.route("/", methods=["GET", "POST"])
# def index():
#     """
#     Home page route.
#     Handles both GET and POST requests.
#     """
#     result = None
#     reason = None  # New variable for explanation
#     error = None
#     password_input = ""

#     if request.method == "POST":
#         try:
#             password_input = request.form.get("password")

#             if not password_input:
#                 raise ValueError("Password field cannot be empty")

#             # Predict password strength using ML model
#             result = predictor.predict(password_input)

#             # Generate explanation for the predicted strength
#             reason = explain_strength(password_input, result)

#         except Exception as e:
#             error = str(e)

#     return render_template(
#         "index.html",
#         result=result,
#         reason=reason,  # Pass reason to template
#         error=error,
#         password=password_input
#     )


# # -------------------------------------------------
# # Application Entry Point
# # -------------------------------------------------

# if __name__ == "__main__":
#     app.run(
#         host="0.0.0.0",
#         port=5000,
#         debug=True
#     )










"""
app.py
-------
Flask web application for SecurePass AI
Password Strength Analyzer with ML Suggestions.
"""





from flask import Flask, render_template, request, jsonify
from src.predict_strength import PasswordStrengthPredictor

# -------------------------------------------------
# Flask App Initialization
# -------------------------------------------------

app = Flask(__name__)

# Load ML model ONCE at startup
predictor = PasswordStrengthPredictor(
    model_path="models/password_strength_model.pkl"
)

# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Home page (UI only)"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Real-time password prediction (AJAX)"""

    try:
        data = request.get_json() or {}
        password = data.get("password", "").strip()

        # ✅ IMPORTANT: allow empty password for live typing
        if not password:
            return jsonify({
                "strength": "",
                "score": 0,
                "time_to_crack": "—",
                "entropy_bits": 0,
                "confidence": 0,
                "explanation": []
            })

        # ML prediction
        result = predictor.predict(password)

        return jsonify({
            "strength": result["strength"],
            "score": int(result["score"]),
            "time_to_crack": result["time_to_crack"],
            "entropy_bits": float(result["entropy_bits"]),
            "confidence": int(result["confidence"]),
            "explanation": result["explanation"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------
# Application Entry Point
# -------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
