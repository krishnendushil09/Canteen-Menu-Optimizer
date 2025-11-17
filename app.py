from flask import Flask, render_template, request
import os
import pickle
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
app.secret_key = "canteen-secret"

# ============================
# LOAD MODELS & PREPROCESSORS
# ============================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Load trained model
with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

# Load target encoder
with open(os.path.join(MODEL_DIR, "label_encoder_target.pkl"), "rb") as f:
    le_target = pickle.load(f)

# Load all feature encoders
with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

# Load numeric imputer
with open(os.path.join(MODEL_DIR, "numeric_imputer.pkl"), "rb") as f:
    numeric_imputer = pickle.load(f)

# Load meta file (contains final feature list)
with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
    meta = json.load(f)

FEATURES = meta["features"]  # exact 67 training features


# ============================
# PREPROCESS FUNCTION
# ============================

def preprocess(df):
    """
    df = DataFrame containing the columns used during training
    returns processed numpy array
    """

    # 1. Ensure all expected features exist
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan  # create missing column

    X = df[FEATURES].copy()

    # 2. Identify numeric columns
    numeric_cols = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)

    # Apply numeric imputation
    if numeric_cols:
        X[numeric_cols] = numeric_imputer.transform(X[numeric_cols])

    # 3. Encode categorical features
    for col, encoder in label_encoders.items():
        if col in X.columns:
            vals = X[col].astype(str).str.strip().fillna("<<MISSING>>")

            # handle unseen labels
            known = set(encoder.classes_)
            cleaned = []
            for v in vals:
                if v in known:
                    cleaned.append(v)
                else:
                    cleaned.append("<<MISSING>>")

            X[col] = encoder.transform(cleaned)

    return X.values.astype(float)


# ============================
# ROUTES
# ============================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        # ============================
        #  BULK CSV PREDICTION
        # ============================
        if "csv_file" in request.files and request.files["csv_file"].filename:
            csv = request.files["csv_file"]
            df = pd.read_csv(csv)

            # Preprocess
            X = preprocess(df)
            preds = model.predict(X)
            labels = le_target.inverse_transform(preds)

            df["Predicted Diet"] = labels

            # Show only relevant columns (inputs + output)
            show_cols = list(df.columns)
            table_html = df[show_cols].head(50).to_html(index=False)

            return render_template(
                "results.html",
                table_html=table_html,
                image_data=None,
                single_prediction=None
            )

        # ============================
        #  SINGLE PREDICTION
        # ============================
        # For a single form, we create a row with all 67 features
        form_dict = request.form.to_dict()

        # Create empty row with ALL features
        row = pd.DataFrame([{col: np.nan for col in FEATURES}])

        # Fill known fields (ONLY if user provided)
        for key, val in form_dict.items():
            if key in row.columns:
                row[key] = val

        X = preprocess(row)
        pred = model.predict(X)[0]
        label = le_target.inverse_transform([pred])[0]

        return render_template("results.html", single_prediction=label)

    return render_template("index.html")


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    app.run(debug=True)
