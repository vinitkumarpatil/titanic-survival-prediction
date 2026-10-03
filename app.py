import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, template_folder="templates")

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Load model or fallback train
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    
    print("Training model on startup...")
    from sklearn.ensemble import RandomForestClassifier
    data_path = os.path.join(BASE_DIR, "data", "titanic.csv")
    df = pd.read_csv(data_path)
    cols_to_drop = [c for c in ['name', 'ticket', 'cabin', 'passengerid', 'deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'] if c in df.columns]
    clean_df = df.drop(columns=cols_to_drop)
    clean_df['age'] = clean_df['age'].fillna(clean_df['age'].median())
    clean_df['embarked'] = clean_df['embarked'].fillna(clean_df['embarked'].mode()[0])
    clean_df['sex'] = clean_df['sex'].map({'male': 0, 'female': 1}).astype(int)
    clean_df['embarked'] = clean_df['embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    X = clean_df.drop('survived', axis=1)
    y = clean_df['survived']

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, MODEL_PATH)
    return rf

model = load_or_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        
        pclass = int(data.get('pclass', 3))
        raw_sex = str(data.get('sex', 'male')).lower().strip()
        sex = 1 if raw_sex in ['female', '1', 'f'] else 0
        age = float(data.get('age', 28.0))
        sibsp = int(data.get('sibsp', 0))
        parch = int(data.get('parch', 0))
        fare = float(data.get('fare', 15.0))
        raw_emb = str(data.get('embarked', 'S')).upper().strip()
        emb_map = {'S': 0, 'C': 1, 'Q': 2}
        embarked = emb_map.get(raw_emb, 0)

        # Feature dataframe matching training columns
        features = pd.DataFrame([{
            'pclass': pclass,
            'sex': sex,
            'age': age,
            'sibsp': sibsp,
            'parch': parch,
            'fare': fare,
            'embarked': embarked
        }], dtype=float)

        pred = int(model.predict(features)[0])
        probs = model.predict_proba(features)[0]
        survival_prob = float(probs[1])

        # Generate intelligent contextual feedback factors
        factors = []
        if sex == 1:
            factors.append("Female passenger: strongly benefited from the 'women and children first' maritime protocol (~74.2% historical survival).")
        else:
            factors.append("Male passenger: lower evacuation priority reduced overall survival odds (~18.9% historical survival).")

        if pclass == 1:
            factors.append("First-class ticket: upper deck cabins provided direct, immediate access to lifeboat stations (~63% survival).")
        elif pclass == 2:
            factors.append("Second-class ticket: moderate deck proximity with ~47.3% survival rate.")
        else:
            factors.append("Third-class steerage: lower deck quarters faced locked gates and delayed evacuation warning (~24.2% survival).")

        if age < 14:
            factors.append("Child passenger: prioritized into lifeboats during the initial loading phase.")
        elif age > 60:
            factors.append("Senior passenger: physical mobility in cold water presented survival challenges.")

        if fare > 75:
            factors.append(f"High ticket fare (£{fare:.2f}): correlated with high socioeconomic standing and privileged boat access.")

        if sibsp + parch > 3:
            factors.append("Large family unit: keeping family together introduced evacuation coordination delays.")

        return jsonify({
            "status": "success",
            "prediction": "Survived" if pred == 1 else "Did Not Survive",
            "survival_probability": round(survival_prob, 4),
            "confidence": f"{max(survival_prob, 1 - survival_prob) * 100:.1f}%",
            "factors": factors[:3]
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    print("Starting Titanic Survival Predictor Web Application...")
    print("Serving on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
