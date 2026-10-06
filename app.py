from flask import Flask, render_template, request
import joblib
import numpy as np

# Load trained model
obj = joblib.load("winequality_linear_regression.joblib")

model = obj["model"]
columns = obj["columns"]

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    values = []

    for column in columns:
        value = request.form.get(column, type=float)
        values.append(value)

    input_data = np.array([values])

    prediction = model.predict(input_data)[0]

    prediction = round(float(prediction), 2)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)