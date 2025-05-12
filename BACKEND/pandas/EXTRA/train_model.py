import pandas as pd
import joblib
from pathlib import Path
import logging
from model import HCFPGrowthModel  # Make sure to import your model class

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_save_model(data_path, model_path="hc_fp_growth_model.joblib"):
    """
    Train and save the HCFPGrowth model
    Args:
        data_path (str): Path to the dataset CSV file
        model_path (str): Path to save the trained model
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate paths
        data_path = Path(data_path)
        model_path = Path(model_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")

        logger.info("Loading dataset...")
        df = pd.read_csv(data_path)

        # Validate required columns
        required_columns = [{'Member_number', 'itemDescription'}, 
                          {'transaction_id', 'item'}]
        if not any(col_set.issubset(df.columns) for col_set in required_columns):
            raise ValueError("Dataset must contain either (Member_number, itemDescription) or (transaction_id, item) columns")

        logger.info("Training model...")
        model = HCFPGrowthModel(min_item_frequency=5, min_support=5)
        model.fit(df)

        logger.info(f"Saving model to {model_path}...")
        joblib.dump(model, model_path)
        
        logger.info("Model trained and saved successfully!")
        return True

    except Exception as e:
        logger.error(f"Error in train_and_save_model: {str(e)}")
        return False

def load_and_analyze(model_path="hc_fp_growth_model.joblib"):
    """
    Load saved model and generate insights
    Returns:
        tuple: (recommendations, trends) or (None, None) if error occurs
    """
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        logger.info("Loading model...")
        model = joblib.load(model_path)

        # Generate recommendations
        logger.info("Generating recommendations...")
        recommendations = []
        top_recs = model.recommend()
        for items, count in top_recs:
            rec_str = f"{' + '.join(items)} -> {count}"
            recommendations.append(rec_str)
            print(rec_str)

        # Get purchase trends
        logger.info("Analyzing purchase trends...")
        trends = model.get_trends()
        
        # Handle different trend formats
        if isinstance(trends, dict):
            active_day = trends.get('active_day', 'N/A')
            peak_hour = trends.get('peak_hour', 'N/A')
            avg_basket = trends.get('avg_basket_size', 'N/A')
        else:
            active_day, peak_hour, avg_basket = trends if len(trends) == 3 else ('N/A', 'N/A', 'N/A')

        trends_summary = {
            'active_day': active_day,
            'peak_hour': peak_hour,
            'avg_basket_size': avg_basket
        }

        print("\nPurchase Trends:")
        print(f"Most active day: {active_day}")
        print(f"Peak hour: {peak_hour}")
        print(f"Avg basket size: {avg_basket:.2f}")

        return recommendations, trends_summary

    except Exception as e:
        logger.error(f"Error in load_and_analyze: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "C:/Users/tejav/OneDrive/Documents/GitHub/market-analysis/BACKEND/pandas/DATASET/Groceries_dataset.csv"
    MODEL_PATH = "hc_fp_growth_model.joblib"

    # Step 1: Train and save model (if needed)
    if not Path(MODEL_PATH).exists():
        logger.info("Model not found, training new model...")
        if not train_and_save_model(DATA_PATH, MODEL_PATH):
            exit(1)

    # Step 2: Load model and generate insights
    recommendations, trends = load_and_analyze(MODEL_PATH)
    
    if recommendations is None:
        logger.error("Failed to generate insights")
        exit(1)

    logger.info("Insights generated successfully!")