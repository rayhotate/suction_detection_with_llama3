import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate accuracy metrics for selected videos')
    parser.add_argument('--videos', nargs='+', help='List of video names to analyze. If not specified, analyzes all videos.')
    return parser.parse_args()

def filter_by_videos(df, video_names):
    if not video_names:
        return df
    
    # Create a filter condition for each video name
    filters = [df['Image'].str.startswith(name) for name in video_names]
    # Combine all filters with OR operation
    combined_filter = pd.concat(filters, axis=1).any(axis=1)
    return df[combined_filter]

def main():
    args = parse_arguments()
    
    # Read the TSV files
    llm_df = pd.read_csv('llm_result.tsv', sep='\t')
    human_df = pd.read_csv('human_result.tsv', sep='\t')
    
    # Filter dataframes if specific videos are requested
    if args.videos:
        llm_df = filter_by_videos(llm_df, args.videos)
        human_df = filter_by_videos(human_df, args.videos)
        print(f"\nAnalyzing videos: {', '.join(args.videos)}")
    
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
    
    # Create output filename suffix if specific videos are selected
    suffix = '_selected' if args.videos else ''
    
    # Print results
    print(f"Total images analyzed: {len(merged_df)}")
    print(f"Accuracy: {accuracy:.2%} ({accuracy:.3f})")
    print("\nConfusion Matrix:")
    print("Labels: No Suctioning, Oral Suctioning, Tracheal Suctioning")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    # Save detailed results to a file
    with open(f'accuracy_results{suffix}.txt', 'w') as f:
        f.write(f"Videos analyzed: {', '.join(args.videos) if args.videos else 'All'}\n")
        f.write(f"Total images: {len(merged_df)}\n")
        f.write(f"Accuracy: {accuracy:.2%} ({accuracy:.3f})\n\n")
        f.write("Confusion Matrix:\n")
        f.write("Labels: No Suctioning, Oral Suctioning, Tracheal Suctioning\n")
        f.write(str(conf_matrix))
        f.write("\n\nClassification Report:\n")
        f.write(class_report)
    
    # Create a DataFrame with disagreements
    disagreements = merged_df[merged_df['human_evaluation'] != merged_df['llm_evaluation']]
    disagreements = disagreements[['Image', 'human_evaluation', 'llm_evaluation', 'Reason']]
    disagreements.to_csv(f'disagreements{suffix}.tsv', sep='\t', index=False)
    
    print(f"\nNumber of disagreements: {len(disagreements)}")
    print(f"Results have been saved to 'accuracy_results{suffix}.txt'")
    print(f"Disagreements have been saved to 'disagreements{suffix}.tsv'")

if __name__ == "__main__":
    main()