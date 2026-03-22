import pandas as pd
from pathlib import Path
from src.training import load_model
from src.preparation import preprocess_data

def predict_with_model(model_path, input_df):
    model = load_model(model_path)
    processed_df = preprocess_data(input_df)
    cols_to_drop = ['target', 'INSR_BEGIN', 'INSR_END'] if 'target' in processed_df.columns else ['INSR_BEGIN', 'INSR_END']
    processed_df = processed_df.drop(columns=[c for c in cols_to_drop if c in processed_df.columns])
    predictions = model.predict(processed_df)
    return predictions

def process_inference_file(input_path, output_path, model_path):
    df = pd.read_csv(input_path)
    predictions = predict_with_model(model_path, df)
    df['predict'] = predictions
    df.to_csv(output_path, index=False)
    return output_path