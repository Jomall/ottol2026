import pandas as pd
import sys
import os

# Add current directory to path to import app
sys.path.insert(0, os.path.dirname(__file__))

from app import generate_sequence, load_existing_df, TENSORFLOW_AVAILABLE

print("TensorFlow available:", TENSORFLOW_AVAILABLE)

# Load sample data
df = load_existing_df('sample.csv')

if df is not None and not df.empty:
    try:
        sequence, model_nla, model_bb, model_letter, model_sup, model_sup_letter = generate_sequence(df)
        print("Prediction successful:")
        print("NLA:", sequence['numbers'], "BB:", sequence['bb'], "Letter:", sequence['letter'])
        if sequence.get('sup6'):
            print("SUP6:", sequence['sup6']['numbers'], "Letter:", sequence['sup6']['letter'])
        print("No errors in ML code.")
    except Exception as e:
        print("Error in prediction:", e)
else:
    print("Sample data not loaded properly.")
