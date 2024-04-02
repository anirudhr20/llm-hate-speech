'''
To run this script: python evaluation.py path_to_excel_file.xlsx
Excel File name format: modelName_datasetName.xlsx
'''

import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import os

def setup_logging():
    logging.basicConfig(filename='../outputs/evaluation_results.log', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def log_results(model_name, dataset_name, metrics):
    logging.info(f'Model: {model_name}, Dataset: {dataset_name}, Metrics: {metrics}')

def evaluate_predictions(file_path):
    # Extract model and dataset names from the file name
    base_name = os.path.basename(file_path)  
    model_name, dataset_name = os.path.splitext(base_name)[0].split('_')  # Split by '_' and remove file extension

    df = pd.read_excel(file_path)

    valid_labels = {'Toxic': 1, 'Non-Toxic': 0}

    # Encode both ground truth and predicted labels
    df['ground_truth_encoded'] = df['ground truth'].map(valid_labels)

    # Identify and penalize hallucinations by inverting the ground truth values
    df['predicted_label_encoded'] = df['predicted label'].apply(lambda x: valid_labels.get(x, 'Hallucination'))
    df['final_prediction'] = df.apply(
    lambda row: 1 - row['ground_truth_encoded'] 
                if row['predicted_label_encoded'] == 'Hallucination' 
                else row['predicted_label_encoded'], 
    axis=1
    )

    # Calculate metrics
    precision = precision_score(df['ground_truth_encoded'], df['final_prediction'], zero_division=0)
    accuracy = accuracy_score(df['ground_truth_encoded'], df['final_prediction'])
    recall = recall_score(df['ground_truth_encoded'], df['final_prediction'], zero_division=0)
    f1 = f1_score(df['ground_truth_encoded'], df['final_prediction'], zero_division=0)


    metrics = f"Precision: {precision}, Accuracy: {accuracy}, Recall: {recall}, F1 Score: {f1}"
    
    print(metrics)

    log_results(model_name, dataset_name, metrics)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='The path to the Excel file containing the predictions and ground truth labels.')

    args = parser.parse_args()

    evaluate_predictions(args.file_path)

