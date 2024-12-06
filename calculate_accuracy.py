import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read the TSV files
llm_df = pd.read_csv('llm_result.tsv', sep='\t')
human_df = pd.read_csv('human_result.tsv', sep='\t')

# Merge the dataframes on the Image column
merged_df = pd.merge(llm_df, human_df, on='Image', how='inner')

# Define the labels
labels = ['No Suctioning', 'Oral Suctioning', 'Tracheal Suctioning']

# Calculate accuracy
accuracy = accuracy_score(merged_df['human_evaluation'], merged_df['llm_evaluation'])

# Generate confusion matrix
conf_matrix = confusion_matrix(merged_df['human_evaluation'], merged_df['llm_evaluation'], 
                             labels=labels)

# Generate detailed classification report
class_report = classification_report(merged_df['human_evaluation'], merged_df['llm_evaluation'],
                                   labels=labels, target_names=labels)

# Print results
print(f"Accuracy: {accuracy:.2%} ({accuracy:.3f})")
print("\nConfusion Matrix:")
print("Labels: No Suctioning, Oral Suctioning, Tracheal Suctioning")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Save detailed results to a file
with open('accuracy_results.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.2%} ({accuracy:.3f})\n\n")
    f.write("Confusion Matrix:\n")
    f.write("Labels: No Suctioning, Oral Suctioning, Tracheal Suctioning\n")
    f.write(str(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(class_report)
    
# Create a DataFrame with disagreements
disagreements = merged_df[merged_df['human_evaluation'] != merged_df['llm_evaluation']]
disagreements = disagreements[['Image', 'human_evaluation', 'llm_evaluation', 'Reason']]
disagreements.to_csv('disagreements.tsv', sep='\t', index=False)

print(f"\nNumber of disagreements: {len(disagreements)}")
print("Disagreements have been saved to 'disagreements.tsv'")