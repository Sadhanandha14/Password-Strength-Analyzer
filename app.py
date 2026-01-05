





from flask import Flask, render_template, request, jsonify
from src.predict_strength import PasswordStrengthPredictor



app = Flask(__name__)

# Load ML model ONCE at startup
predictor = PasswordStrengthPredictor(
    model_path="models/password_strength_model.pkl"
)



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

        # IMPORTANT: allow empty password for live typing
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


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
