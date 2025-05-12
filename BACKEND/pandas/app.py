from flask import Flask, render_template
import joblib
import pandas as pd
from pathlib import Path
from model import HCFPGrowthModel

app = Flask(__name__)

MODEL_PATH = "hc_fp_growth_model.joblib"
DATA_PATH = "DATASET/Groceries_dataset.csv"

# Load or train model
if Path(MODEL_PATH).exists():
    model = joblib.load(MODEL_PATH)
else:
    df = pd.read_csv(DATA_PATH)
    model = HCFPGrowthModel(min_item_frequency=5, min_support=5)
    model.fit(df)
    joblib.dump(model, MODEL_PATH)

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/homepage.html")
def homepage():
    recs = model.recommend(top_n=10)
    active_day, peak_hour, avg_basket = model.get_trends()
    return render_template(
        "homepage.html",
        recommendations=recs,
        trends={
            "active_day": active_day,
            "peak_hour": peak_hour,
            "basket_size": avg_basket
        } if active_day and peak_hour and avg_basket else None
    )
@app.route('/upload.html')
def upload():
    return render_template('upload.html') 
@app.route("/frequent.html")
def frequent_items():
    return render_template("frequent.html")

@app.route('/recommendations.html')
def recommendations():
    return render_template('recommendations.html') 

if __name__ == "__main__":
    app.run(debug=True)
