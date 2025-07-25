from flask import Flask, request, jsonify, render_template_string
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fake News Detector</title>
    <style>
        body { font-family: Arial; background: #f4f4f9; padding: 50px; }
        .container { max-width: 600px; margin: auto; padding: 30px; background: white; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        textarea { width: 100%; height: 150px; margin-bottom: 20px; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        #result { margin-top: 20px; font-size: 18px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📰 Fake News Detector</h1>
        <form method="POST" action="/predict">
            <textarea name="text" placeholder="Paste news content here..."></textarea>
            <button type="submit">Check News</button>
        </form>
        {% if prediction %}
        <div id="result">
            Prediction: {{ prediction }} <br>
            Confidence: {{ confidence }}%
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][pred]
    label = "Real" if pred == 1 else "Fake"
    return render_template_string(HTML_PAGE, prediction=label, confidence=round(float(prob)*100, 2))

if __name__ == "__main__":
    app.run(debug=True)
