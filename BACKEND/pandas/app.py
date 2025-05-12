from flask import Flask, render_template, jsonify
import joblib
from pathlib import Path
from model import HCFPGrowthModel

app = Flask(__name__)

# File paths
MODEL_PATH = "hc_fp_growth_model.joblib"
DATA_PATH = "DATASET/Groceries_dataset.csv"

# Load or train model
if Path(MODEL_PATH).exists():
    model = joblib.load(MODEL_PATH)
else:
    model = HCFPGrowthModel(min_item_frequency=5, min_support=5)
    model.fit(DATA_PATH)
    joblib.dump(model, MODEL_PATH)

# Routes
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/homepage.html")
def homepage():
    # Get recommendations and trends data
    recs = model.recommend(top_n=10)
    active_day, peak_hour, avg_basket = model.get_trends()

    return render_template(
        "homepage.html",
        recommendations=recs,
        active_day=active_day,
        peak_hour=peak_hour,
        avg_basket=avg_basket
    )
@app.route("/reports")
def reports():
    trends = model.get_trends()
    return jsonify(trends)

if __name__ == "__main__":
    app.run(debug=True)