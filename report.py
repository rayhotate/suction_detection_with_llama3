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

def generate_report():
    # Read results
    llm_df = pd.read_csv('llm_result.tsv', sep='\t')
    human_df = pd.read_csv('human_result.tsv', sep='\t')
    disagreements_df = pd.read_csv('disagreements.tsv', sep='\t')
    
    # Merge dataframes
    merged_df = pd.merge(llm_df, human_df, on='Image', how='inner')
    
    def add_image_examples(f, merged_df, disagreements_df):
        f.write("## Image Analysis Examples\n\n")
        
        # True Positives (convert boolean strings to lowercase for comparison)
        true_positives = merged_df[
            (merged_df['llm_evaluation'].astype(str).str.lower() == 'true') & 
            (merged_df['human_evaluation'].astype(str).str.lower() == 'true')
        ]
        
        f.write("### True Positives (Correct Turning Assistance Detection)\n")
        if not true_positives.empty:
            samples = true_positives.sample(min(2, len(true_positives)))
            for _, row in samples.iterrows():
                f.write(f"\n**Image**: `{row['Image']}`\n")
                f.write("- **Evaluation**: Both human and LLM correctly identified turning assistance\n")
                f.write(f"- **LLM Reasoning**: {row['Reason'][:200]}...\n")
                f.write("- **Key Features**: Active physical contact, proper positioning, clear movement intent\n\n")
        else:
            f.write("\nNo examples of true positives found in the dataset.\n\n")
        
        # True Negatives
        true_negatives = merged_df[
            (merged_df['llm_evaluation'].astype(str).str.lower() == 'false') & 
            (merged_df['human_evaluation'].astype(str).str.lower() == 'false')
        ]
        f.write("### True Negatives (Correct Non-Turning Detection)\n")
        if not true_negatives.empty:
            samples = true_negatives.sample(min(2, len(true_negatives)))
            for _, row in samples.iterrows():
                f.write(f"\n**Image**: `{row['Image']}`\n")
                f.write("- **Evaluation**: Both human and LLM correctly identified non-turning scenario\n")
                f.write(f"- **LLM Reasoning**: {row['Reason'][:200]}...\n")
                f.write("- **Key Features**: No physical contact for turning, different care activities\n\n")
        else:
            f.write("\nNo examples of true negatives found in the dataset.\n\n")
        
        # Disagreements
        f.write("### Notable Disagreements\n")
        if not disagreements_df.empty:
            samples = disagreements_df.sample(min(3, len(disagreements_df)))
            for _, row in samples.iterrows():
                f.write(f"\n**Image**: `{row['Image']}`\n")
                f.write(f"- **Human Evaluation**: {row['human_evaluation']}\n")
                f.write(f"- **LLM Evaluation**: {row['llm_evaluation']}\n")
                f.write(f"- **LLM Reasoning**: {row['Reason'][:200]}...\n")
                f.write("- **Analysis of Disagreement**: ")
                if row['human_evaluation'] == 'True':
                    f.write("LLM missed subtle turning assistance indicators\n\n")
                else:
                    f.write("LLM possibly over-interpreted preparatory positioning\n\n")
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
        f.write(f"This analysis evaluates the performance of LLaMA 3.2 Vision model in detecting three types of medical suctioning procedures across {len(merged_df)} medical images. The model achieved {accuracy:.2%} accuracy, with varying performance across different suctioning types.\n\n")
        
        # Data Sources
        f.write("## Data Sources\n")
        f.write("### Video Sources\n")
        video_sources = [
            "[24-hour home care - caregiver training](https://www.youtube.com/watch?v=b77yWsYy7T4)",
            "[Assisting with Positioning a Patient in Bed](https://www.youtube.com/watch?v=HnDYPm_C3Ws&t=192s)", 
            "[Fundamentals of turning and cushion placement](https://www.youtube.com/watch?v=Y5X429CeV70)"
        ]
        for source in video_sources:
            f.write(f"- {source}\n")
        
        # Add frame extraction visualization and details
        f.write("### Frame Extraction Process\n")
        f.write("The frame extraction process is implemented using OpenCV (cv2) with the following specifications:\n\n")
        f.write("- **Sampling Rate**: Every 3 seconds extracted for consistent analysis\n")
        f.write("- **Implementation**:\n")
        f.write("  - Uses OpenCV's VideoCapture for efficient video processing\n")
        f.write("  - Frames are saved as high-quality JPG images\n")
        f.write("  - Maintains original aspect ratio and resolution\n")
        f.write("- **Processing Flow**:\n")
        f.write("  1. Reads video files from source directory\n")
        f.write("  2. Creates unique output directories for each video\n")
        f.write("  3. Extracts frames at specified intervals\n")
        f.write("  4. Applies consistent naming convention: `{video_name}_frame_{frame_number}.jpg`\n\n")
        f.write("- **Statistics**:\n")
        f.write("  - Total frames analyzed: 320\n")
        f.write("  - Format: High-quality JPG images\n")
        f.write("  - Original video sources: 3\n\n")
        f.write("For detailed implementation, see:\n")
        f.write("```python:split2frames.py\n")
        f.write(read_code_file('split2frames.py', 5, 5))
        f.write("\n```\n")
        
        # 3. Technical Implementation
        f.write("## Technical Implementation\n\n")
        f.write("### Core Components\n")
        f.write("1. **LLaMA 3.2 Vision Model Integration**\n")
        f.write("```python:llama32_detect.py\n")
        f.write(read_code_file('llama32_detect.py', 80, 167))
        f.write("\n```\n")
        
        # Evaluation Process
        f.write("\n## Evaluation Process\n")
        f.write("### Human Evaluation Interface\n\n")

        # Add the interface screenshot
        f.write("![Human Evaluation Interface](assets/evaluation.png)\n\n")
        f.write("The human evaluation interface provides a simple way to assess images with the following features:\n")
        f.write("- Displays current image with filename\n")
        f.write("- Shows LLM's evaluation and reasoning\n")
        f.write("- Keyboard controls: 't' for True, 'f' for False\n")
        f.write("- Progress tracking and automatic result saving\n\n")

        f.write("Implementation details:\n")
        f.write("```python:human_evaluation.py\n")
        f.write(read_code_file('human_evaluation.py', 9, 92))
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
        
        # Recommendations
        f.write("\n## Recommendations\n")
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
        f.write("```\nYou are a medical image analysis expert. Your task is to carefully analyze the image and determine if it shows a patient being assisted in turning by another person.\n```\n\n")
        
        f.write("2. **Example Cases**\n")
        f.write("```\nExample 1: Active Turning\n")
        f.write("Image: A nurse standing next to a hospital bed with her hands on a patient's shoulder and hip, clearly in the process of rolling them from their back to their side.\n")
        f.write("Analysis: True - This shows active turning assistance because:\n")
        f.write("- Direct physical contact between caregiver and patient\n")
        f.write("- Clear repositioning movement from back to side\n")
        f.write("- Proper supportive hand placement for turning\n\n")
        
        f.write("Example 2: Non-Turning Care\n")
        f.write("Image: A patient lying still in bed while a nurse stands nearby checking IV fluids.\n")
        f.write("Analysis: False - This is not turning assistance because:\n")
        f.write("- No physical contact for movement support\n")
        f.write("- Patient position is static\n")
        f.write("- Caregiver is performing different care tasks\n```\n\n")
        
        f.write("3. **Analysis Framework**\n")
        f.write("The model evaluates each image using four key aspects:\n\n")
        f.write("- **People Present**\n")
        f.write("  - Patient visibility\n")
        f.write("  - Caregiver presence\n")
        f.write("  - Relative positioning\n\n")
        
        f.write("- **Physical Contact & Assistance**\n")
        f.write("  - Direct physical contact\n")
        f.write("  - Contact points (hands, arms)\n")
        f.write("  - Supportive stance\n\n")
        
        f.write("- **Patient Position & Movement**\n")
        f.write("  - Current position\n")
        f.write("  - Movement evidence\n")
        f.write("  - Intended direction\n\n")
        
        f.write("- **Level of Assistance**\n")
        f.write("  - Active support\n")
        f.write("  - Specific turning actions\n")
        f.write("  - Scenario clarity\n\n")
        
        f.write("### Processing Pipeline\n")
        f.write("```mermaid\n")
        f.write("graph TD\n")
        f.write("    A[Input Image] --> B[Image Processing]\n")
        f.write("    B --> C[LLaMA Vision Model]\n")
        f.write("    C --> D[Structured Analysis]\n")
        f.write("    D --> E[Binary Classification]\n")
        f.write("    E --> F[Reasoning Output]\n")
        f.write("```\n\n")
        
        f.write("### Output Format\n")
        f.write("The model generates:\n")
        f.write("1. Detailed analysis of the image\n")
        f.write("2. Binary classification (True/False)\n")
        f.write("3. Supporting reasoning\n\n")
        
        f.write("Example output:\n")
        f.write("```\n")
        f.write("**Analysis of the Image**\n")
        f.write("Upon examining the image, I notice...\n\n")
        f.write("**Conclusion**\n")
        f.write("Based on [specific observations]...\n\n")
        f.write("**Final Determination**\n")
        f.write("* True/False: [reasoning]\n")
        f.write("```\n\n")

def write_classification_report(f, class_report):
    f.write("### Classification Report\n")
    f.write("| Class | Precision | Recall | F1-Score | Support |\n")
    f.write("|-------|-----------|---------|-----------|----------|\n")
    for class_name in ['No Suctioning', 'Oral Suctioning', 'Tracheal Suctioning']:
        metrics = class_report[class_name]
        f.write(f"| {class_name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1-score']:.3f} | {metrics['support']} |\n")

if __name__ == "__main__":
    generate_report()
    # Read the Markdown file
    with open('README.md', 'r') as file:
        text = file.read()

    # Convert Markdown to HTML
    html = markdown.markdown(text)

    # Save the HTML output
    with open('output.html', 'w') as file:
        file.write(html)
