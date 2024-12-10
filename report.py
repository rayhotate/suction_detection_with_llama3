import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import markdown

def read_code_file(filepath, start_line=None, end_line=None):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            # If no line numbers specified, return entire file
            if start_line is None and end_line is None:
                return ''.join(lines).strip()
            
            # Adjust line numbers to 0-based indexing
            start = (start_line - 1) if start_line else 0
            end = end_line if end_line else len(lines)
            
            # Return specified lines
            return ''.join(lines[start:end]).strip()
            
    except FileNotFoundError:
        return f"# Error: File '{filepath}' not found"
    except Exception as e:
        return f"# Error reading file: {str(e)}"

def generate_confusion_matrix(merged_df, f):
    # Create confusion matrix
    conf_matrix = confusion_matrix(merged_df['human_evaluation'], merged_df['llm_evaluation'])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Suctioning', 'Oral Suctioning', 'Tracheal Suctioning'],
                yticklabels=['No Suctioning', 'Oral Suctioning', 'Tracheal Suctioning'])
    plt.title('Confusion Matrix')
    plt.ylabel('Human Evaluation')
    plt.xlabel('LLM Evaluation')
    
    # Save plot to file
    plt.savefig('assets/confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    # Add to markdown
    f.write("\n### Confusion Matrix\n")
    f.write("![Confusion Matrix](assets/confusion_matrix.png)\n\n")

def calculate_video_metrics(merged_df, video_name):
    # Filter dataframe for specific video
    print(merged_df)
    video_df = merged_df[merged_df['Image'].str.startswith(video_name)]
    if len(video_df) == 0:
        return None
        
    # Calculate metrics with zero_division parameter
    accuracy = accuracy_score(video_df['human_evaluation'], video_df['llm_evaluation'])
    conf_matrix = confusion_matrix(
        video_df['human_evaluation'], 
        video_df['llm_evaluation'],
        labels=['No Suctioning', 'Oral Suctioning', 'Tracheal Suctioning']
    )
    class_report = classification_report(
        video_df['human_evaluation'], 
        video_df['llm_evaluation'],
        labels=['No Suctioning', 'Oral Suctioning', 'Tracheal Suctioning'],
        output_dict=True,
        zero_division=0
    )
    
    return {
        'total_frames': len(video_df),
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

def generate_report():
    # Read results
    llm_df = pd.read_csv('llm_result.tsv', sep='\t')
    human_df = pd.read_csv('human_result.tsv', sep='\t')
    disagreements_df = pd.read_csv('disagreements.tsv', sep='\t')
    
    # Merge dataframes
    merged_df = pd.merge(llm_df, human_df, on='Image', how='inner')
    
    def add_image_examples(f, merged_df, disagreements_df):
        f.write("## Image Analysis Examples\n\n")
        
        # Correct Classifications
        for category in ['No Suctioning', 'Oral Suctioning', 'Tracheal Suctioning']:
            correct_predictions = merged_df[
                (merged_df['llm_evaluation'] == category) & 
                (merged_df['human_evaluation'] == category)
            ]
            
            f.write(f"### Correct {category} Detection\n")
            if not correct_predictions.empty:
                samples = correct_predictions.sample(min(2, len(correct_predictions)))
                for _, row in samples.iterrows():
                    f.write(f"\n**Image**: `{row['Image']}`\n")
                    f.write(f"- **Evaluation**: Both human and LLM correctly identified {category.lower()}\n")
                    f.write(f"- **LLM Reasoning**: {row['Reason'][:200]}...\n")
                    f.write("- **Key Features**: ")
                    if category == 'No Suctioning':
                        f.write("Absence of suctioning equipment or procedure\n\n")
                    elif category == 'Oral Suctioning':
                        f.write("Dental setting, oral cavity access, wide-bore suction tools\n\n")
                    else:  # Tracheal Suctioning
                        f.write("Tracheostomy access, sterile catheter, supine positioning\n\n")
            else:
                f.write(f"\nNo examples of correct {category.lower()} detection found in the dataset.\n\n")
        
        # Disagreements section remains largely the same but with updated wording
        f.write("### Notable Disagreements\n")
        if not disagreements_df.empty:
            samples = disagreements_df.sample(min(3, len(disagreements_df)))
            for _, row in samples.iterrows():
                f.write(f"\n**Image**: `{row['Image']}`\n")
                f.write(f"- **Human Evaluation**: {row['human_evaluation']}\n")
                f.write(f"- **LLM Evaluation**: {row['llm_evaluation']}\n")
                f.write(f"- **LLM Reasoning**: {row['Reason'][:200]}...\n")
                f.write("- **Analysis of Disagreement**: ")
                f.write(f"Misclassification between {row['human_evaluation']} and {row['llm_evaluation']}\n\n")
        else:
            f.write("\nNo disagreements found in the dataset.\n\n")
    
    # Calculate metrics
    accuracy = accuracy_score(merged_df['human_evaluation'], merged_df['llm_evaluation'])
    conf_matrix = confusion_matrix(merged_df['human_evaluation'], merged_df['llm_evaluation'])
    class_report = classification_report(merged_df['human_evaluation'], merged_df['llm_evaluation'], output_dict=True)
    
    # Generate report
    with open('README.md', 'w') as f:
        # Title and Executive Summary
        f.write("# Medical Suctioning Detection Analysis Report\n\n")
        f.write("## Executive Summary\n")
        f.write(f"""This analysis evaluates the performance of LLaMA 3.2 Vision model in detecting three types of medical suctioning procedures across {len(merged_df)} medical images. The model achieved {accuracy:.2%} accuracy for the three label classifications (No Suctioning, Oral Suctioning, Tracheal Suctioning), with varying performance across different suctioning types.

Key findings from the analysis:

1. **Overall Performance**
   - Total Images Analyzed: {len(merged_df)} (from four videos)
   - Overall Accuracy: {accuracy:.2%}
   - Number of Disagreements: {len(disagreements_df)}

2. **Category-Specific Performance**
   - No Suctioning: High precision (0.88) but moderate recall (0.65)
   - Oral Suctioning: Good precision (0.78) but low recall (0.33)
   - Tracheal Suctioning: Low precision (0.20) but high recall (0.94)

3. **Video-Specific Results**
   - Best Performance: "Suctioning (National Tracheostomy Safety Project)" - 80.49% accuracy
   - Moderate Performance: "Performing Oropharyngeal Suctioning" - 65.12% accuracy
   - Lower Performance: "#9 How to perform oral suctioning" - 49.44% accuracy

4. **Key Challenges**
   - Difficulty distinguishing between oral and tracheal suctioning procedures
   - Inconsistent performance across different video sources

5. **Notable Strengths**
   - Strong ability to identify absence of suctioning (0.88 precision)
   - Good performance in clear clinical settings
   - Reliable detection of standard medical equipment

This analysis highlights both the potential and current limitations of using LLaMA 3.2 Vision for medical procedure detection, suggesting specific areas for improvement in future iterations.\n\n""")
        
        # Data Sources
        f.write("## Data Sources\n")
        f.write("### Video Sources\n")
        video_sources = [
            "[Oral Suctioning](https://www.youtube.com/shorts/l-Rygg3N04Y)",
            "[Suctioning (National Tracheostomy Safety Project)](https://www.youtube.com/watch?v=lGpfuHdrUgk)",
            "[Performing Oropharyngeal Suctioning](https://www.youtube.com/watch?v=SwoLb3z25fc)",
            "[#9 How to perform oral suctioning](https://www.youtube.com/watch?v=DIBMp_yh0gY)"
        ]
        for source in video_sources:
            f.write(f"- {source}\n")
        
        # Add frame extraction visualization and details
        f.write("### Frame Extraction Process\n")
        f.write("The frame extraction process is implemented using OpenCV (cv2) with the following specifications:\n\n")
        f.write("- **Sampling Rate**: Every 2 seconds extracted for consistent analysis\n")
        f.write("- **Implementation**:\n")
        f.write("  - Uses OpenCV's VideoCapture for efficient video processing\n")
        f.write("  - Frames are saved as high-quality JPG images\n")
        f.write("  - Maintains original aspect ratio and resolution\n")
        f.write("- **Processing Flow**:\n")
        f.write("  1. Reads video files from source directory\n")
        f.write("  2. Creates unique output directories for each video\n")
        f.write("  3. Extracts frames at specified intervals\n")
        f.write("  4. Applies consistent naming convention: `{video_name}_frame_{frame_number}.jpg`\n\n")
        f.write("- **Data Cleaning**:\n")
        f.write("  - Excluded non-medical content (e.g., `#9 How to perform oral suctioning_frame_drawing.jpg`)\n")
        f.write("  - Ensured patients in the dataset are only real medical procedure images\n\n")
        f.write("  - ![Example excluded frame](assets/9_How_to_perform_oral_suctioning_frame_drawing.jpg)\n")
        f.write("- **Statistics**:\n")
        f.write("  - Total frames analyzed: 190\n")
        f.write("  - Format: High-quality JPG images\n")
        f.write("  - Original video sources: 4\n\n")
        f.write("For detailed implementation, see:\n")
        f.write("```python:split2frames.py\n")
        f.write(read_code_file('split2frames.py', 5, 5))
        f.write("\n```\n")
        
        # 3. Technical Implementation
        f.write("## Technical Implementation\n\n")
        f.write("### Core Components\n")
        f.write("1. **LLaMA 3.2 Vision Model Integration**\n")
        f.write("```python:llama32_detect.py\n")
        f.write(read_code_file('llama32_detect.py', 137, 255))
        f.write("\n```\n")
        
        # Evaluation Process
        f.write("\n## Evaluation Process\n")
        f.write("### Human Evaluation Interface\n\n")

        # Add the interface screenshot
        f.write("![Human Evaluation Interface](assets/evaluation.png)\n\n")
        f.write("The human evaluation interface provides a simple way to assess images with the following features:\n")
        f.write("- Displays current image with filename\n")
        f.write("- Shows LLM's evaluation and reasoning\n")
        f.write("- Keyboard controls:\n")
        f.write("  - 'n' for No Suctioning\n")
        f.write("  - 'o' for Oral Suctioning\n")
        f.write("  - 't' for Tracheal Suctioning\n")
        f.write("- Progress tracking and automatic result saving\n\n")

        f.write("Implementation details:\n")
        f.write("```python:human_evaluation.py\n")
        f.write(read_code_file('human_evaluation.py', 9, 98))
        f.write("\n```\n")
        
        # Results Analysis
        f.write("\n## Results Analysis\n")
        f.write("### Performance Metrics\n")
        f.write(f"- Total Images: {len(merged_df)}\n")
        f.write(f"- Overall Accuracy: {accuracy:.2%}\n")
        f.write(f"- Number of Disagreements: {len(disagreements_df)}\n\n")
        
        # Add classification report and confusion matrix
        write_classification_report(f, class_report)
        generate_confusion_matrix(merged_df, f)
        
        # Add image examples
        add_image_examples(f, merged_df, disagreements_df)
        
        # Add per-video results
        write_video_results(f, merged_df, video_sources)
        
        # Recommendations
        f.write("\n## Future Work\n")
        f.write("1. **Model Improvements**\n")
        f.write("   - Enhance distinction between oral and tracheal suctioning\n")
        f.write("   - Improve detection of suctioning equipment and setup\n")
        f.write("   - Add confidence scoring for predictions\n\n")
        
        f.write("2. **Data Collection**\n")
        f.write("   - Balance dataset across all three suctioning types\n")
        f.write("   - Include more examples of tracheal suctioning\n")
        f.write("   - Add temporal context between frames\n\n")
        
        # Project Structure
        f.write("\n## Project Files\n")
        f.write("### Core Components\n")
        f.write("- **llama32_detect.py**: Vision model implementation\n")
        f.write("- **human_evaluation.py**: Manual annotation interface\n")
        f.write("- **calculate_accuracy.py**: Performance analysis\n")
        f.write("- **report.py**: Analysis report generation\n\n")
        
        f.write("### Output Files\n")
        f.write("- **llm_result.tsv**: Model predictions and reasoning\n")
        f.write("- **human_result.tsv**: Human annotations\n")
        f.write("- **disagreements.tsv**: Cases where model and human differ\n")
        f.write("- **accuracy_results.txt**: Detailed performance metrics\n")
        
        # Add LLM Detection Pipeline section
        f.write("## LLM Detection Pipeline\n\n")
        
        f.write("### Model Configuration\n")
        f.write("```python\n")
        f.write("model_id = 'meta-llama/Llama-3.2-11B-Vision-Instruct'\n")
        f.write("model = MllamaForConditionalGeneration.from_pretrained(\n")
        f.write("    model_id,\n")
        f.write("    torch_dtype=torch.bfloat16\n")
        f.write(")\n```\n\n")
        
        f.write("### Prompt Engineering\n")
        f.write("The model uses a carefully crafted prompt with three key components:\n\n")

        f.write("1. **Role Definition**\n")
        f.write("```\nYou are a medical image analysis expert. Your task is to carefully analyze the image and determine if it shows a patient undergoing suctioning using a tube. Classify the scenario into one of the following categories: No Suctioning, Oral Suctioning (dental), or Tracheal Suctioning (throat/covid).\n```\n\n")

        f.write("2. **Definitions and Criteria**\n")
        f.write("```\n1. Oral Suctioning:\n")
        f.write("   - Performed exclusively by licensed dentists or dental assistants\n")
        f.write("   - Suction device must be actively placed inside patient's oral cavity\n")
        f.write("   - Specifically for removal of oral fluids during dental procedures\n")
        f.write("   - Patient must be seated upright in a dental chair\n")
        f.write("   - Equipment: Wide-bore dental suction tools (>8mm diameter)\n")
        f.write("   - Caregiver position: Within 45 degrees of patient's front, at oral level\n\n")

        f.write("2. Tracheal Suctioning:\n")
        f.write("   - Performed only by licensed healthcare professionals\n")
        f.write("   - Sterile catheter must be actively inserted through tracheostomy opening\n")
        f.write("   - Exclusively for clearing respiratory secretions from airways\n")
        f.write("   - Patient must be supine or at maximum 30 degree incline\n")
        f.write("   - Equipment: Sterile flexible catheter (10-14 French/3.3-4.7mm diameter)\n")
        f.write("   - Caregiver position: Standing at head of bed, within 30cm of patient's head\n```\n\n")

        f.write("3. **Analysis Framework**\n")
        f.write("The model evaluates each image using four key aspects:\n\n")
        f.write("- **Patient and Caregiver Assessment**\n")
        f.write("  - Patient presence and positioning\n")
        f.write("  - Healthcare provider identification\n")
        f.write("  - Provider positioning relative to patient\n\n")

        f.write("- **Equipment Verification**\n")
        f.write("  - Suction device type and size\n")
        f.write("  - Active insertion verification\n")
        f.write("  - Proper equipment usage\n\n")

        f.write("- **Procedure Context**\n")
        f.write("  - Clinical setting assessment\n")
        f.write("  - Patient positioning\n")
        f.write("  - Supporting medical equipment\n\n")

        f.write("- **Active Suctioning Indicators**\n")
        f.write("  - Ongoing procedure verification\n")
        f.write("  - Proper technique assessment\n")
        f.write("  - Supporting device presence\n\n")

        f.write("### Processing Pipeline\n")
        f.write("```mermaid\n")
        f.write("graph TD\n")
        f.write("    A[Input Image] --> B[Image Processing]\n")
        f.write("    B --> C[LLaMA Vision Model]\n")
        f.write("    C --> D[Structured Analysis]\n")
        f.write("    D --> E[Classification]\n")
        f.write("    E --> F[Detailed Reasoning]\n")
        f.write("```\n\n")
        
        f.write("### Output Format\n")
        f.write("The model generates a structured output with three components:\n")
        f.write("1. Detailed analysis of the medical scene\n")
        f.write("2. Classification into one of three categories:\n")
        f.write("   - No Suctioning\n")
        f.write("   - Oral Suctioning\n")
        f.write("   - Tracheal Suctioning\n")
        f.write("3. Supporting reasoning with key observations\n\n")
        
        f.write("Example output:\n")
        f.write("```\n")
        f.write("**Analysis of the Image**\n")
        f.write("The image shows a medical professional in PPE standing at the head of a hospital bed...\n\n")
        f.write("**Key Observations**\n")
        f.write("- Patient positioning: Supine at 30° incline\n")
        f.write("- Equipment: Sterile catheter (4mm diameter)\n")
        f.write("- Procedure: Active insertion through tracheostomy\n")
        f.write("- Setting: ICU with monitoring equipment\n\n")
        f.write("**Classification**\n")
        f.write("Tracheal Suctioning\n")
        f.write("```\n\n")

def write_classification_report(f, class_report):
    f.write("### Classification Report\n")
    f.write("| Class | Precision | Recall | F1-Score | Support |\n")
    f.write("|-------|-----------|---------|-----------|----------|\n")
    for class_name in ['No Suctioning', 'Oral Suctioning', 'Tracheal Suctioning']:
        metrics = class_report.get(class_name, {})
        # Handle cases where metrics might be missing
        if not metrics or metrics.get('support', 0) == 0:
            f.write(f"| {class_name} | N/A | N/A | N/A | 0 |\n")
        else:
            f.write(f"| {class_name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1-score']:.3f} | {metrics['support']} |\n")

def write_video_results(f, merged_df, video_sources):
    f.write("\n## Per-Video Analysis\n\n")
    #print(video_sources)
    for video_url in video_sources:
        # Extract video name from URL
        video_name = video_url.split(']')[0].replace('[', '')
        #print(video_name)
        metrics = calculate_video_metrics(merged_df, video_name)
        
        if metrics is None:
            continue
            
        f.write(f"### {video_name}\n")
        
        # Find a representative frame showing suctioning
        video_df = merged_df[merged_df['Image'].str.startswith(video_name)]
        suctioning_frames = video_df[
            (video_df['human_evaluation'] == 'Oral Suctioning') | 
            (video_df['human_evaluation'] == 'Tracheal Suctioning')
        ]
        
        if not suctioning_frames.empty:
            example_frame = suctioning_frames.iloc[0]['Image']
            f.write(f"![Representative Frame](frames/{example_frame})\n\n")
            f.write(f"*Representative frame showing {suctioning_frames.iloc[0]['human_evaluation'].lower()}*\n\n")
        
        f.write(f"- **Total Frames**: {metrics['total_frames']}\n")
        f.write(f"- **Accuracy**: {metrics['accuracy']:.2%}\n\n")
        
        # Add classification report
        f.write("#### Classification Report\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("|-------|-----------|---------|-----------|----------|\n")
        for class_name in ['No Suctioning', 'Oral Suctioning', 'Tracheal Suctioning']:
            metrics_data = metrics['classification_report'].get(class_name, {})
            if metrics_data.get('support', 0) == 0:
                f.write(f"| {class_name} | No samples | No samples | No samples | 0 |\n")
            else:
                f.write(f"| {class_name} | {metrics_data['precision']:.3f} | {metrics_data['recall']:.3f} | {metrics_data['f1-score']:.3f} | {metrics_data['support']} |\n")
        f.write("\n")
        
        # Add confusion matrix visualization
        f.write("#### Confusion Matrix\n")
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Suctioning', 'Oral Suctioning', 'Tracheal Suctioning'],
                    yticklabels=['No Suctioning', 'Oral Suctioning', 'Tracheal Suctioning'])
        plt.title(f'Confusion Matrix - {video_name}')
        plt.ylabel('Human Evaluation')
        plt.xlabel('LLM Evaluation')
        
        # Save plot to file
        plt.savefig(f'assets/confusion_matrix_{video_name.replace(" ", "_")}.png', bbox_inches='tight')
        plt.close()
        
        f.write(f"![Confusion Matrix](assets/confusion_matrix_{video_name.replace(' ', '_')}.png)\n\n")

if __name__ == "__main__":
    generate_report()
    # Read the Markdown file
    with open('README.md', 'r') as file:
        text = file.read()

    """# Convert Markdown to HTML
    html = markdown.markdown(text)

    # Save the HTML output
    with open('output.html', 'w') as file:
        file.write(html)"""
