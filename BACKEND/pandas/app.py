from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Load or train model
MODEL_PATH = "hc_fp_growth_model.joblib"
DATA_PATH = "C:/Users/tejav/OneDrive/Documents/GitHub/market-analysis/BACKEND/pandas/DATASET/Groceries_dataset.csv"

# Load model or train if missing
if Path(MODEL_PATH).exists():
    model = joblib.load(MODEL_PATH)
else:
    #from model import HCFPGrowthModel  # Assuming you have a class for HCFP-Growth
    df = pd.read_csv(DATA_PATH)
    #model = HCFPGrowthModel(min_item_frequency=5, min_support=5)
    #model.fit(df)
    #joblib.dump(model, MODEL_PATH)

@app.route("/")
def home():
    return render_template("login.html")

@app.route("/recommendations")
def recommendations():
    recs = model.recommend(top_n=10)
    formatted = [
        {
            "items": " + ".join(items),
            "count": count
        }
        for items, count in recs
    ]
    return jsonify(formatted)

@app.route("/trends")
def trends():
    day, hour, basket_size = model.get_trends()
    return jsonify({
        "most_active_day": day,
        "peak_hour": f"{hour}:00",
        "average_basket_size": f"{basket_size:.2f} items"
    })

if __name__ == "__main__":
    app.run(debug=True)
