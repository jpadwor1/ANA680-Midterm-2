from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

MODEL_PATH = "svm_rbf_top_features.pkl"

# Load bundle once at startup
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

selected_features = bundle["selected_features"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
model = bundle["model"]


def _coerce_float(value: str):
    """Convert form input to float; raise ValueError if invalid."""
    if value is None:
        raise ValueError("Missing value")
    v = value.strip()
    if v == "":
        raise ValueError("Empty value")
    return float(v)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", features=selected_features)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        x = []
        for feat in selected_features:
            x.append(_coerce_float(request.form.get(feat)))

        X = np.array([x], dtype=float)

        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)

        pred = int(model.predict(X_scaled)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_scaled)[0][1])

        label = "Malignant" if pred == 1 else "Benign"

        return render_template(
            "index.html",
            features=selected_features,
            prediction=pred,
            label=label,
            proba=proba
        )

    except Exception as e:
        return render_template(
            "index.html",
            features=selected_features,
            error=str(e)
        ), 400


if __name__ == "__main__":
    app.run(port=5000, debug=True)
