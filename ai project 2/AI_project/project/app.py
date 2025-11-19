from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from openai import OpenAI

# Load ML Models
tfidf = joblib.load("model/tfidf.pkl")
le = joblib.load("model/label_encoder.pkl")
nn_model = load_model("model/nn_model.h5")

# OpenAI Client
client = OpenAI(api_key="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")

app = Flask(__name__)


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- DISEASE PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["symptoms"]

    X = tfidf.transform([text]).toarray()
    pred = nn_model.predict(X)
    final = np.argmax(pred)

    disease = le.inverse_transform([final])[0]

    return jsonify({"prediction": disease})


# ---------------- CHATBOT API ----------------
@app.route("/chat_api", methods=["POST"])
def chat_api():
    data = request.get_json()
    user_msg = data.get("message", "")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful medical chatbot."},
                {"role": "user", "content": user_msg},
            ]
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == "__main__":
    app.run(debug=True)
