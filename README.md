# Medical Suctioning Detection Analysis Report

## Executive Summary
This analysis evaluates the performance of LLaMA 3.2 Vision model in detecting three types of medical suctioning procedures across 218 medical images. The model achieved 52.29% accuracy, with varying performance across different suctioning types.

## Data Sources
### Video Sources
- [Oral Suctioning](https://www.youtube.com/shorts/l-Rygg3N04Y)
- [Suctioning (National Tracheostomy Safety Project)](https://www.youtube.com/watch?v=lGpfuHdrUgk)
- [Performing Oropharyngeal Suctioning](https://www.youtube.com/watch?v=SwoLb3z25fc)
- [Suctioning the endotracheal tube - medical animation](https://www.youtube.com/watch?v=pN6-EYoeh3g)
- [#9 How to perform oral suctioning](https://www.youtube.com/watch?v=DIBMp_yh0gY)
### Frame Extraction Process
The frame extraction process is implemented using OpenCV (cv2) with the following specifications:

- **Sampling Rate**: Every 3 seconds extracted for consistent analysis
- **Implementation**:
  - Uses OpenCV's VideoCapture for efficient video processing
  - Frames are saved as high-quality JPG images
  - Maintains original aspect ratio and resolution
- **Processing Flow**:
  1. Reads video files from source directory
  2. Creates unique output directories for each video
  3. Extracts frames at specified intervals
  4. Applies consistent naming convention: `{video_name}_frame_{frame_number}.jpg`

- **Statistics**:
  - Total frames analyzed: 320
  - Format: High-quality JPG images
  - Original video sources: 3

For detailed implementation, see:
```python:split2frames.py
def extract_frames_from_videos(video_dir, output_dir, frequency=3):
```
## Technical Implementation

### Core Components
1. **LLaMA 3.2 Vision Model Integration**
```python:llama32_detect.py
- Wider suction tool typical of dental procedures
- Dental office/chair setting

Now analyze the given image considering:
Consider the following key aspects to determine the type of suctioning:
1. Patient and Caregiver Assessment
- Identify if a patient is present in the frame
- Look for healthcare providers (dentist, nurse, assistant)
- Check if their positioning aligns with:
  * Dental setup: Provider within 45° of patient's front
  * Medical setup: Provider at head of bed within 30cm

2. Equipment Verification  
- Identify presence and type of suction device:
  * Dental: Wide-bore tool (>8mm diameter)
  * Medical: Thin flexible catheter (3.3-4.7mm)
- Verify active insertion and ongoing suctioning:
  * Dental: Must be inside oral cavity and actively suctioning
  * Medical: Must be through tracheostomy, 10cm+ depth and actively suctioning
  * No Suctioning: Device visible but not inserted, or inserted but not actively suctioning

3. Procedure Context
- Evaluate patient positioning:
  * Dental: Upright in dental chair
  * Medical: Supine or max 30° incline
- Assess clinical setting:
  * Dental office vs medical facility
- Look for procedure-specific equipment:
  * Dental chair, lights, tools
  * Hospital bed, monitors, sterile field

4. Active Suctioning Indicators
- Check for clear evidence of ongoing suctioning process:
  * Device must be actively inserted and performing suction
  * Merely holding or positioning device is not sufficient
  * No Suctioning if device is visible but not actively used
- Verify proper technique:
  * Provider in correct procedural stance
  * Proper equipment selection and active use
- Look for supporting medical devices:
  * Tracheostomy tube
  * Ventilator equipment
  * Dental procedure setup

Note: Classify as No Suctioning if:
- Device is visible but not inserted in patient
- Device is inserted but no active suctioning occurring
- Provider is only holding/preparing device
- Any pause or break in active suctioning process

Based on your analysis, provide your response in the following format:

OBSERVATION: [Detailed description of what you observe in the image]
CLASSIFICATION: [One of: No Suctioning, Oral Suctioning, or Tracheal Suctioning]
EVIDENCE: [List the key evidence that led to your conclusion]
"""

def img2text(input_path, output_file = None, exportedfile_indexing = False, show_img = False, max_new_tokens = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    #device_map="auto",
    )
    
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    #tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct', trust_remote_code=True)
    #model.eval()
    dir = [input_path]
    if os.path.isdir(input_path):
        dir = os.listdir(input_path)
    
    data = []
    result = {}
    for i, image_path in enumerate(sorted(dir)):
        # Read the image
        if os.path.isdir(input_path):
            image = Image.open(Path(input_path).joinpath(image_path))
        else:
            image = Image.open(image_path)
        
        
        # Describe the image
        input_text = processor.apply_chat_template(msgs("Describe the image in detail."), add_generation_prompt=True)
```

## Evaluation Process
### Human Evaluation Interface

![Human Evaluation Interface](assets/evaluation.png)

The human evaluation interface provides a simple way to assess images with the following features:
- Displays current image with filename
- Shows LLM's evaluation and reasoning
- Keyboard controls: 't' for True, 'f' for False
- Progress tracking and automatic result saving

Implementation details:
```python:human_evaluation.py
class ImageEvaluator:
    def __init__(self):
        # Get list of images from frames directory
        self.image_files = sorted([f for f in os.listdir("frames") if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.current_index = 0
        self.results = {}
        
        # Load LLM evaluations
        self.llm_df = pd.read_csv('llm_result.tsv', sep='\t')
        self.llm_df.set_index('Image', inplace=True)
        
        # Create figure
        self.fig = plt.figure(figsize=(10, 10))
        self.ax_img = plt.axes([0.1, 0.2, 0.8, 0.7])
        
        # Connect keyboard event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Start evaluation
        self.evaluate_images()
        
    def evaluate_images(self):
        plt.ion()  # Turn on interactive mode
        
        while self.current_index < len(self.image_files):
            self.display_current_image()
            plt.pause(0.001)  # Small pause to allow GUI to update
            
            # Wait for keyboard input
            while self.current_index == len(self.results):
                plt.pause(0.1)
                
                # Check if we've processed all images
                if self.current_index >= len(self.image_files):
                    plt.close('all')
                    self.save_results()
                    return  # Exit the method after saving
        
        # Save results if we exit the main loop
        plt.close('all')
        self.save_results()

    def display_current_image(self):
        current_image = self.image_files[self.current_index]
        
        # Get LLM evaluation and reason if available
        llm_eval = "Unknown"
        reason = "No reason provided"
        if current_image in self.llm_df.index:
            llm_eval = self.llm_df.loc[current_image, 'llm_evaluation']
            reason = self.llm_df.loc[current_image, 'Reason']
        
        # Clear previous image
        self.ax_img.clear()
        
        # Load and display current image
        image_path = os.path.join("frames", current_image)
        img = Image.open(image_path)
        self.ax_img.imshow(img)
        self.ax_img.axis('off')
        self.ax_img.set_title(f"Image {self.current_index + 1}/{len(self.image_files)}\n"
                       f"Filename: {current_image}\n"
                       f"LLM Evaluation: {llm_eval}\n"
                       f"LLM Reason: {reason[:300]}...\n"
                       f"Press 'n' for No Suctioning, 'o' for Oral Suctioning, or 't' for Tracheal Suctioning")  # Show first 300 chars of reason
        
        plt.draw()
        
    def on_key_press(self, event):
        if event.key in ['n', 'o', 't'] and self.current_index < len(self.image_files):
            current_image = self.image_files[self.current_index]
            if event.key == 'n':
                self.results[current_image] = 'No Suctioning'
            elif event.key == 'o':
                self.results[current_image] = 'Oral Suctioning'
            elif event.key == 't':
                self.results[current_image] = 'Tracheal Suctioning'
            self.current_index += 1
            if self.current_index < len(self.image_files):
                self.display_current_image()
            plt.draw()
        
    def save_results(self):
        # Convert results to DataFrame and save as TSV
```

## Results Analysis
### Performance Metrics
- Total Images: 218
- Overall Accuracy: 52.29%
- Number of Disagreements: 104

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|-----------|----------|
| No Suctioning | 0.857 | 0.639 | 0.732 | 122.0 |
| Oral Suctioning | 0.769 | 0.256 | 0.385 | 78.0 |
| Tracheal Suctioning | 0.158 | 0.889 | 0.269 | 18.0 |

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

## Image Analysis Examples

### True Positives (Correct Turning Assistance Detection)

No examples of true positives found in the dataset.

### True Negatives (Correct Non-Turning Detection)

No examples of true negatives found in the dataset.

### Notable Disagreements

**Image**: `#9 How to perform oral suctioning_frame_92.jpg`
- **Human Evaluation**: Oral Suctioning
- **LLM Evaluation**: Tracheal Suctioning
- **LLM Reasoning**: {}...
- **Analysis of Disagreement**: LLM possibly over-interpreted preparatory positioning


**Image**: `Oral suctioning_frame_9.jpg`
- **Human Evaluation**: Oral Suctioning
- **LLM Evaluation**: Tracheal Suctioning
- **LLM Reasoning**: {'observation': "**\nThe image depicts a woman in a hospital setting, wearing a mask and gloves, standing next to a mannequin in a hospital bed. The woman is holding a suction device with the tube vis...
- **Analysis of Disagreement**: LLM possibly over-interpreted preparatory positioning


**Image**: `#9 How to perform oral suctioning_frame_63.jpg`
- **Human Evaluation**: No Suctioning
- **LLM Evaluation**: Tracheal Suctioning
- **LLM Reasoning**: {}...
- **Analysis of Disagreement**: LLM possibly over-interpreted preparatory positioning


## Per-Video Analysis

### Suctioning (National Tracheostomy Safety Project)
- **Total Frames**: 41
- **Accuracy**: 80.49%

#### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|-----------|----------|
| No Suctioning | 0.944 | 0.708 | 0.810 | 24.0 |
| Oral Suctioning | No samples | No samples | No samples | 0 |
| Tracheal Suctioning | 0.762 | 0.941 | 0.842 | 17.0 |

#### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix_Suctioning_(National_Tracheostomy_Safety_Project).png)

### Performing Oropharyngeal Suctioning
- **Total Frames**: 43
- **Accuracy**: 65.12%

#### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|-----------|----------|
| No Suctioning | 0.933 | 0.848 | 0.889 | 33.0 |
| Oral Suctioning | 0.000 | 0.000 | 0.000 | 10.0 |
| Tracheal Suctioning | No samples | No samples | No samples | 0 |

#### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix_Performing_Oropharyngeal_Suctioning.png)

### Suctioning the endotracheal tube - medical animation
- **Total Frames**: 20
- **Accuracy**: 10.00%

#### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|-----------|----------|
| No Suctioning | 0.333 | 0.333 | 0.333 | 3.0 |
| Oral Suctioning | 0.500 | 0.062 | 0.111 | 16.0 |
| Tracheal Suctioning | 0.000 | 0.000 | 0.000 | 1.0 |

#### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix_Suctioning_the_endotracheal_tube_-_medical_animation.png)

### #9 How to perform oral suctioning
- **Total Frames**: 97
- **Accuracy**: 46.39%

#### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|-----------|----------|
| No Suctioning | 0.794 | 0.540 | 0.643 | 50.0 |
| Oral Suctioning | 0.857 | 0.383 | 0.529 | 47.0 |
| Tracheal Suctioning | No samples | No samples | No samples | 0 |

#### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix_#9_How_to_perform_oral_suctioning.png)


## Recommendations
1. **Model Improvements**
   - Enhance distinction between oral and tracheal suctioning
   - Improve detection of suctioning equipment and setup
   - Add confidence scoring for predictions

2. **Data Collection**
   - Balance dataset across all three suctioning types
   - Include more examples of tracheal suctioning
   - Add temporal context between frames


## Project Files
### Core Components
- **llama32_detect.py**: Vision model implementation
- **human_evaluation.py**: Manual annotation interface
- **calculate_accuracy.py**: Performance analysis
- **report.py**: Analysis report generation

### Output Files
- **llm_result.tsv**: Model predictions and reasoning
- **human_result.tsv**: Human annotations
- **disagreements.tsv**: Cases where model and human differ
- **accuracy_results.txt**: Detailed performance metrics
## LLM Detection Pipeline

### Model Configuration
```python
model_id = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)
```

### Prompt Engineering
The model uses a carefully crafted prompt with three key components:

1. **Role Definition**
```
You are a medical image analysis expert. Your task is to carefully analyze the image and determine if it shows a patient being assisted in turning by another person.
```

2. **Example Cases**
```
Example 1: Active Turning
Image: A nurse standing next to a hospital bed with her hands on a patient's shoulder and hip, clearly in the process of rolling them from their back to their side.
Analysis: True - This shows active turning assistance because:
- Direct physical contact between caregiver and patient
- Clear repositioning movement from back to side
- Proper supportive hand placement for turning

Example 2: Non-Turning Care
Image: A patient lying still in bed while a nurse stands nearby checking IV fluids.
Analysis: False - This is not turning assistance because:
- No physical contact for movement support
- Patient position is static
- Caregiver is performing different care tasks
```

3. **Analysis Framework**
The model evaluates each image using four key aspects:

- **People Present**
  - Patient visibility
  - Caregiver presence
  - Relative positioning

- **Physical Contact & Assistance**
  - Direct physical contact
  - Contact points (hands, arms)
  - Supportive stance

- **Patient Position & Movement**
  - Current position
  - Movement evidence
  - Intended direction

- **Level of Assistance**
  - Active support
  - Specific turning actions
  - Scenario clarity

### Processing Pipeline
```mermaid
graph TD
    A[Input Image] --> B[Image Processing]
    B --> C[LLaMA Vision Model]
    C --> D[Structured Analysis]
    D --> E[Binary Classification]
    E --> F[Reasoning Output]
```

### Output Format
The model generates:
1. Detailed analysis of the image
2. Binary classification (True/False)
3. Supporting reasoning

Example output:
```
**Analysis of the Image**
Upon examining the image, I notice...

**Conclusion**
Based on [specific observations]...

**Final Determination**
* True/False: [reasoning]
```

