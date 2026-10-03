import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    
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

model = get_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        
        pclass = int(data.get('pclass', 3))
        raw_sex = str(data.get('sex', 'male')).lower().strip()
        sex = 1 if raw_sex in ['female', '1', 'f'] else 0
        age = float(data.get('age', 25.0))
        sibsp = int(data.get('sibsp', 0))
        parch = int(data.get('parch', 0))
        
        # If fare not provided, default by class
        default_fares = {1: 85.0, 2: 22.0, 3: 8.05}
        fare = float(data.get('fare', default_fares.get(pclass, 15.0)))
        
        raw_emb = str(data.get('embarked', 'S')).upper().strip()
        emb_map = {'S': 0, 'C': 1, 'Q': 2}
        embarked = emb_map.get(raw_emb, 0)

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

        # Short, simple, friendly explanations
        factors = []
        if sex == 1:
            factors.append("Women were prioritized for lifeboats under the 'women and children first' order.")
        else:
            factors.append("Men were directed to step back during lifeboat boarding.")

        if pclass == 1:
            factors.append("1st class passengers had cabins closest to the top boat deck.")
        elif pclass == 2:
            factors.append("2nd class passengers had moderate access to emergency boats.")
        else:
            factors.append("3rd class cabins were located far below on lower decks.")

        if age <= 12:
            factors.append("Children were given early access to lifeboats.")

        return jsonify({
            "status": "success",
            "prediction": "Survived" if pred == 1 else "Did Not Survive",
            "survival_probability": round(survival_prob, 4),
            "factors": factors[:2]
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    print("Serving simple Titanic Predictor on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
