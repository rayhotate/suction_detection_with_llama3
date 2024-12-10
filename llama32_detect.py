# Dir Related
import os
from pathlib import Path
import pandas as pd

import requests
import torch
from PIL import Image
from IPython.display import display, HTML
from transformers import MllamaForConditionalGeneration, AutoProcessor

#####

def msgs(prompt, with_image=True):
    if with_image:
        return [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
    else:
        return [
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]

prompt_orig = """You are a medical image analysis expert. Your task is to carefully analyze the image and determine if it shows a patient undergoing suctioning using a tube. Classify the scenario into one of the following categories: No Suctioning, Oral Suctioning (dental), or Tracheal Suctioning (throat/covid).


Definitions:
1. Oral Suctionin:
   - Performed exclusively by licensed dentists or dental assistants
   - Suction device must be actively placed inside patient's oral cavity
   - Specifically for removal of oral fluids during dental procedures
   - Patient must be seated upright in a dental chair
   - Equipment: Wide-bore dental suction tools (>8mm diameter)
   - Caregiver position: Within 45 degrees of patient's front, at oral level

2. Tracheal Suctioning: (does not include nasal/nose suctioning)
   - Performed only by licensed healthcare professionals
   - Sterile catheter must be actively inserted through tracheostomy opening in neck into trachea, minimum 10cm depth
   - Exclusively for clearing respiratory secretions from airways
   - Patient must be supine or at maximum 30 degree incline
   - Equipment: Sterile flexible catheter (10-14 French/3.3-4.7mm diameter)
   - Caregiver position: Standing at head of bed, within 30cm of patient's head

3. No Suctioning:
   - Complete absence of any suction equipment in frame
   - Suction equipment visible but no physical contact with patient
   - Suction device >30cm away from patient's airways
   - Caregiver not in correct position for either procedure
   - Equipment type does not match procedure specifications
   - No active insertion of suction device into any body cavity

Here are some examples:

Example 1:
Image: A nurse holding a suction catheter with the tube visibly penetrating the patient's neck through a tracheostomy opening.
Analysis: Tracheal Suctioning - This shows active tracheal suctioning because:
- Direct insertion of catheter through tracheostomy site in neck
- Sterile catheter penetrating at least 10cm into trachea
- Proper positioning of nurse at head of bed
- Thin, flexible catheter appropriate for tracheal procedures

Example 2:
Image: A patient lying in bed with a nurse holding a suction catheter near the patient but not touching.
Analysis: No Suctioning - This is not suctioning because:
- The suction catheter is not in contact with the patient
- No visible tube insertion or contact
- Caregiver is not actively performing suctioning
- Only preparation or assessment visible

Example 3:
Image: A dentist using a suction device in a patient's mouth.
Analysis: Oral Suctioning - This shows oral suctioning because:
- The suction device is in contact with the patient's mouth
- The caregiver is positioned as a dentist would be during oral procedures
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
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        
        res = model.generate(**inputs, max_new_tokens=max_new_tokens)
        res = processor.decode(res[0]).split("<|end_header_id|>")[-1].replace('\n', ' ')
        
        # Show the steps based on the image
        prompt = "The picture is about the following:\n" +res +'\n' + prompt_orig
        
        input_text = processor.apply_chat_template(msgs(prompt), add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        
        res = model.generate(**inputs, max_new_tokens=max_new_tokens)
        res = processor.decode(res[0])
        
        
        print('\n', i, image_path)
        #print(generated_text,'\n')
        print('Full Response\n', res)
        reason = res.split("<|end_header_id|>")[-1]
        print("Reason:", reason.replace('\n', ' '))
        
        # Conclude
        input_text = processor.apply_chat_template(
            msgs(reason + "\nTask: Provide your final classification in the following format ONLY:\nCLASSIFICATION: [No Suctioning/Oral Suctioning/Tracheal Suctioning]"),
            add_generation_prompt=True
        )
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        
        res = model.generate(**inputs, max_new_tokens=max_new_tokens)
        res = processor.decode(res[0])
        response_text = res.split("<|end_header_id|>")[-1]
        
        # Extract classification using more robust parsing
        classification = "Unknown"
        if "CLASSIFICATION:" in response_text:
            classification_text = response_text.split("CLASSIFICATION:")[-1].strip()
            # Remove any trailing text after the classification
            classification_text = classification_text.split("\n")[0].strip()
            # Remove any square brackets
            classification_text = classification_text.strip("[]")
            
            # Normalize the text and check for matches
            classification_text = classification_text.lower()
            if "no suctioning" in classification_text:
                classification = "No Suctioning"
            elif "oral suctioning" in classification_text:
                classification = "Oral Suctioning"
            elif "tracheal suctioning" in classification_text:
                classification = "Tracheal Suctioning"
        
        print("Classification:", classification)
        
        # Parse the detailed reason response
        parsed_reason = {}
        if "OBSERVATION:" in reason:
            sections = reason.split("EVIDENCE:")
            if len(sections) > 1:
                parsed_reason = {
                    "observation": sections[0].split("OBSERVATION:")[-1].split("CLASSIFICATION:")[0].strip(),
                    "evidence": sections[1].strip()
                }
        
        data.append([image_path, classification, parsed_reason])
        result[image_path] = (classification, parsed_reason)
        if show_img:
            display(HTML(f'<img src="{Path(input_path).joinpath(image_path) if os.path.isdir(input_path) else image_path }" style="width:30%;">'))
    data.sort()
    
    # if output_file is specified, it generates tsv file
    if output_file is not None:
        data_frame = pd.DataFrame(data, columns=['Image', 'llm_evaluation', 'Reason'])
        data_frame.to_csv(output_file, sep = '\t', index = exportedfile_indexing, encoding = 'utf-8')
    return result 

if __name__ == "__main__" :
    print(img2text("frames", output_file = "result.tsv", show_img = False))
