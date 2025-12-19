from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# ---------------- Dataset ----------------
# Load CSV
data = pd.read_csv("spam.csv")

# Features and Labels
X = data["v2"]
y = data["v1"]

# Vectorize text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_vec, y)

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        msg = request.form["message"]
        msg_vec = vectorizer.transform([msg])
        pred = model.predict(msg_vec)
        if pred[0] == "spam":
            prediction = "SPAM 🚫"
        else:
            prediction = "SAFE EMAIL ✅"
    return render_template("index.html", prediction=prediction)

# ---------------- Run Server ----------------
if __name__ == "__main__":
    app.run(debug=True)
