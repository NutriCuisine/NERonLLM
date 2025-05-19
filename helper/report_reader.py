import pandas as pd
import json
import re
from datetime import datetime

def parse_classification_report(report_text):
    """
    Parse a classification report string into a dictionary.
    
    Args:
        report_text (str): The classification report string
        
    Returns:
        dict: Dictionary containing the parsed metrics
    """
    class_metrics = {}
    
    # Extract metrics for each class
    lines = report_text.strip().split('\n')
    for line in lines[2:-4]:  # Skip header and average lines
        if line.strip():
            parts = line.split()
            if len(parts) >= 5:
                class_name = parts[0]
                class_metrics[class_name] = {
                    'precision': float(parts[1]),
                    'recall': float(parts[2]),
                    'f1_score': float(parts[3]),
                    'support': int(parts[4])
                }
    
    # Extract average metrics
    avg_lines = lines[-4:]
    for line in avg_lines:
        if 'avg' in line:
            parts = line.split()
            if len(parts) >= 5:
                avg_type = parts[0] + '_' + parts[1]  # e.g., 'micro_avg'
                try:
                    class_metrics[avg_type] = {
                        'precision': float(parts[2]),
                        'recall': float(parts[3]),
                        'f1_score': float(parts[4]),
                        'support': int(parts[5])
                    }
                except (ValueError, IndexError):
                    continue
    
    return class_metrics

# Read the JSON file
with open('training_results.json', 'r') as f:
    data = json.load(f)

# Extract model reports
model_reports = data['model_reports']

# Create a list to store flattened data
flattened_data = []

# Process each model report
for model in model_reports:
    model_name = model['model_name']
    best_val_f1 = model['best_val_f1']
    ner_flag = model['ner_flag']
    
    # Process epoch results
    for epoch in model['epoch_results']:
        # Parse classification report
        report_text = epoch['val_classification_report']
        class_metrics = parse_classification_report(report_text)
        
        # Combine all metrics
        epoch_data = {
            'model_name': model_name,
            'ner_flag': ner_flag,
            'best_val_f1': best_val_f1,
            'epoch': epoch['epoch'],
            'train_loss': epoch['train_loss'],
            'val_loss': epoch['val_loss'],
            'val_f1': epoch['val_f1'],
            'val_precision': epoch['val_precision'],
            'val_recall': epoch['val_recall'],
            'val_accuracy': epoch['val_accuracy'],
            'val_h_loss': epoch['val_h_loss'],
            **class_metrics
        }
        flattened_data.append(epoch_data)

# Create DataFrame from flattened data
df = pd.DataFrame(flattened_data)

# Get best epoch for each model
best_epochs = df.loc[df.groupby('model_name')['val_f1'].idxmax()]

# Create NER Impact Analysis Report
report_lines = []
report_lines.append("NER Impact Analysis Report")
report_lines.append("=" * 50)
report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 1. Direct NER Impact Comparison
report_lines.append("NER Impact Comparison")
report_lines.append("-" * 30)
report_lines.append("\nModels are sorted by their F1 scores with NER enabled:")

# Sort models by F1 score with NER
ner_models = best_epochs[best_epochs['ner_flag'] == True].sort_values('val_f1', ascending=False)
for _, row in ner_models.iterrows():
    report_lines.append(f"\n{row['model_name']}:")
    report_lines.append(f"  F1 Score: {row['val_f1']:.4f}")
    report_lines.append(f"  Accuracy: {row['val_accuracy']:.4f}")
    report_lines.append(f"  Micro Avg F1: {row['micro_avg']['f1_score']:.4f}")
    report_lines.append(f"  Macro Avg F1: {row['macro_avg']['f1_score']:.4f}")

# 2. Performance Comparison
report_lines.append("\n\nOverall Model Rankings")
report_lines.append("-" * 30)
report_lines.append("\nAll models ranked by F1 Score:")
top_models = best_epochs.sort_values('val_f1', ascending=False)
for _, row in top_models.iterrows():
    report_lines.append(f"\n{row['model_name']} (NER: {'Yes' if row['ner_flag'] else 'No'})")
    report_lines.append(f"  F1 Score: {row['val_f1']:.4f}")
    report_lines.append(f"  Accuracy: {row['val_accuracy']:.4f}")

# 3. Class-wise Performance
report_lines.append("\n\nClass-wise Performance Analysis")
report_lines.append("-" * 30)
report_lines.append("\nBest performing model for each class:")
for class_name in ['healthy', 'vegan', 'low-carb', 'gluten-free', 'nut-free', 'high-protein', 'low-sugar']:
    if class_name in best_epochs.columns:
        # Get the best model for this class based on F1 score
        best_model_idx = best_epochs[class_name].apply(lambda x: x['f1_score']).idxmax()
        best_model = best_epochs.loc[best_model_idx]
        report_lines.append(f"\n{class_name}:")
        report_lines.append(f"  Best Model: {best_model['model_name']} (NER: {'Yes' if best_model['ner_flag'] else 'No'})")
        report_lines.append(f"  F1 Score: {best_model[class_name]['f1_score']:.4f}")

# 4. NER Impact Summary
report_lines.append("\n\nNER Impact Summary")
report_lines.append("-" * 30)
ner_avg = best_epochs[best_epochs['ner_flag'] == True]['val_f1'].mean()
non_ner_avg = best_epochs[best_epochs['ner_flag'] == False]['val_f1'].mean() if len(best_epochs[best_epochs['ner_flag'] == False]) > 0 else 0
report_lines.append(f"\nAverage F1 Score with NER: {ner_avg:.4f}")
if non_ner_avg > 0:
    report_lines.append(f"Average F1 Score without NER: {non_ner_avg:.4f}")
    report_lines.append(f"NER Impact: {((ner_avg - non_ner_avg) / non_ner_avg * 100):.2f}% improvement")

# Write report to file
report_path = 'ner_impact_analysis.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"\nNER Impact Analysis Report has been saved to: {report_path}")

# Export detailed data to CSV
df.to_csv('training_results_analysis.csv', index=False)
print("Detailed results exported to 'training_results_analysis.csv'")