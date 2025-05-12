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
def home():
    return render_template("login.html")

@app.route("/homepage.html")
def dashboard():
    return render_template("homepage.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/frequent-items")
def frequent_items():
    return render_template("frequent_items.html")

@app.route("/recommendations")
def recommendations():
    recs = model.recommend(top_n=10)
    return jsonify([
        {"items": items, "count": count}
        for items, count in recs
    ])

@app.route("/reports")
def reports():
    trends = model.get_trends()
    return jsonify(trends)

if __name__ == "__main__":
    app.run(debug=True)