import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_predictions(file_path):
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
    precision = precision_score(df['ground_truth_encoded'], df['final_prediction'])
    accuracy = accuracy_score(df['ground_truth_encoded'], df['final_prediction'])
    recall = recall_score(df['ground_truth_encoded'], df['final_prediction'])
    f1 = f1_score(df['ground_truth_encoded'], df['final_prediction'])


    print(f"Precision: {precision}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='The path to the Excel file containing the predictions and ground truth labels.')

    args = parser.parse_args()

    evaluate_predictions(args.file_path)

# Sample : python evaluation.py path_to_excel_file.xlsx