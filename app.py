from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from preprocess import encode_texts
impoer os

app = Flask(__name__)
app.secret_key = "secret123"

# Load model
model = load_model("models/fake_news_model.h5")

# Load tokenizer
with open("tokenizer/tokenizer.json") as f:
    tokenizer = tokenizer_from_json(f.read())

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":
        text = request.form.get("news", "").strip()

        if text:
            seq = encode_texts([text], tokenizer)
            pred = model.predict(seq)[0][0]

            percentage = round(float(pred) * 100, 2)   # 👈 ADD THIS

            if pred > 0.8:
                result = "Fake News ❌"
                confidence = percentage
            elif pred < 0.2:
                result = "Real News ✅"
                confidence = 100 - percentage
            else:
                result = "Uncertain 🤔"
                confidence = percentage

            session["result"] = result
            session["text"] = text
            session["confidence"] = confidence   # 👈 SAVE

        else:
            session["result"] = None
            session["text"] = ""
            session["confidence"] = None

        return redirect(url_for("home"))

    result = session.pop("result", None)
    text = session.pop("text", "")
    confidence = session.pop("confidence", None)   # 👈 GET

    return render_template(
        "index.html",
        result=result,
        text=text,
        confidence=confidence
    )



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)