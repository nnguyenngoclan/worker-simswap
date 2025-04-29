# from huggingface_hub import login

# # Paste your Hugging Face token here
# login(token="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")

# import torch
# from diffusers import StableDiffusionPipeline

# pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
#                                                 torch_dtype=torch.bfloat16, 
#                                                 )
# pipe = pipe.to("mps")

# # Define prompt and negative prompt
# prompt = (
#     "A happy family portrait in a cozy living room. Father, mother, teenage son, young daughter, warm smiles, soft lighting,"
#     "Christmas tree in background. Ultra-realistic, high-definition, cinematic lighting."
# )

# negative_prompt = (
#     "extra people, multiple heads, distorted features, missing limbs, extra limbs, "
#     "low resolution, unrealistic lighting, overly exaggerated expressions, cartoonish style, "
#     "text artifacts, deformed hands, unnatural body proportions, unwanted background elements."
# )

# # Generate the image with the negative prompt
# image = pipe(
#     prompt,
#     negative_prompt=negative_prompt,  # Add negative prompt here    
# ).images[0]

# image.save("capybara.png")


import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import pandas as pd

from insightface_func.face_detect_crop_multi import Face_detect_crop
import cv2
import cv2
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from deepface import DeepFace
import json


def predidct_age_gender_race_emotion(images):
    traits = []

    for _, image in enumerate(images):        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # trait = fairface_detect(image_rgb)
        trait = deepface_detect(image_rgb)
        traits.append((image, trait))        

    # print("Got traits saved results at ", only_traints)

    return traits

def detect_age(image):
    # List of different face detection backends
    backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]
    ages = []
    print("Got type of image: ", type(image))

    for backend in backends:
        result = DeepFace.analyze(img_path=image, actions=['age'], detector_backend=backend, enforce_detection=False)
        ages.append(result[0]['age'])

    # Compute the average age
    final_age = np.mean(ages)
    return round(final_age)

def deepface_detect(image):
    trait = {}
    result = DeepFace.analyze(img_path=image, detector_backend="retinaface")
    trait["age"] = result[0]["age"]
    trait["gender"] = result[0]['dominant_gender']
    trait["race"] = result[0]['dominant_race']
    trait["emotion"] = result[0]['dominant_emotion']
    
    return trait

def fairface_detect(image):
    device = torch.device("cpu")

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_7_20190809.pt', map_location=device))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_4_20190809.pt', map_location=device))
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # img pth of face images

    # image = dlib.load_rgb_image(img_name)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = trans(image)
    image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
    image = image.to(device)

    # # fair
    outputs = model_fair_7(image)
    outputs = outputs.cpu().detach().numpy()
    outputs = np.squeeze(outputs)

    race_outputs = outputs[:7]
    gender_outputs = outputs[7:9]
    age_outputs = outputs[9:18]

    race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
    gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
    age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

    race_pred = np.argmax(race_score)
    gender_pred = np.argmax(gender_score)
    age_pred = np.argmax(age_score)

    # fair 4 class
    # outputs = model_fair_4(image)
    # outputs = outputs.cpu().detach().numpy()
    # outputs = np.squeeze(outputs)

    # race_outputs = outputs[:4]
    # race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
    # race_pred = np.argmax(race_score)

    trait = {}

    match race_pred:
        case 0:
            trait["race"] = "White"
        case 1:
            trait["race"] = "Black"
        case 2:
            trait["race"] = "Latino_Hispanic"
        case 3:
            trait["race"] = "East Asian"
        case 4:
            trait["race"] = "Southeast Asian"
        case 5:
            trait["race"] = "Indian"
        case 6:
            trait["race"] = "Middle Eastern"

    match age_pred:
        case 0:
            trait["age"] = "0-2"
        case 1:
            trait["age"] = "3-9"
        case 2:
            trait["age"] = "10-19"
        case 3:
            trait["age"] = "20-29"
        case 4:
            trait["age"] = "30-39"
        case 5:
            trait["age"] = "40-49"
        case 6:
            trait["age"] = "50-59"
        case 7:
            trait["age"] = "60-69"
        case 8:
            trait["age"] = "70+"

    match gender_pred:
        case 0:
            trait["gender"] = "Male"
        case 1:
            trait["gender"] = "Female"
    
    return trait

crop_size = 224
#Crop the face from the  and extract the traints: age, gender and ethnicity
def crop_face(image_path, prefix_name):
    # Load the face detector
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_crop = Face_detect_crop(name="antelope", root='./insightface_func/models')
    # Use 'antelope' model
    face_crop.prepare(ctx_id=0 if torch.cuda.is_available() else -1)  # Set device

    # image_path = "image.png"

    img = cv2.imread(image_path)
    img_list, M_list = face_crop.get(img, crop_size)

    # Save the cropped faces
    # for i, cropped_face in enumerate(img_list):
    #     # cropped_face_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    #     index = f"0{i}"  if i < 10 else i
    #     cv2.imwrite(f"swap_multiple/{prefix_name}_{index}.png", cropped_face)
        
    
    # print(f"Extracted {len(img_list)} faces and saved them as images.")

    return predidct_age_gender_race_emotion(images=img_list)
    

def compute_similarity(traits1, traits2):
    """
    Compute similarity between two trait dictionaries.
    - Uses Euclidean distance for age.
    - Uses a high penalty for gender mismatch.
    - Uses 0/1 scoring for race similarity.
    """
    age_diff = abs(traits1['age'] - traits2['age'])
    gender_score = float('inf') if traits1['gender'] != traits2['gender'] else 0  # Ensure gender must match    
    race_score = 0 if traits1['race'] == traits2['race'] else 1
    emotion_score = 0 if traits1['emotion'] == traits2['emotion'] else 0.5
    
    return age_diff + gender_score + race_score + emotion_score

def match_faces(source_faces, target_faces):
    """Match each face in source_faces to the closest face in target_faces based on traits."""
    matches = []
    used_targets = set()
    
    for src_idx, (src_img, src_traits) in enumerate(source_faces):
        best_match = None
        best_score = float('inf')
        best_tgt_idx = None
        
        for tgt_idx, (tgt_img, tgt_traits) in enumerate(target_faces):
            if tgt_idx in used_targets:
                continue  # Ensure one-to-one matching
            
            score = compute_similarity(src_traits, tgt_traits)
            
            if score < best_score:
                best_score = score
                best_match = (tgt_idx, tgt_img, tgt_traits)
                best_tgt_idx = tgt_idx
        
        if best_match:
            matches.append((src_idx, best_match[0], src_img, best_match[1]))  # Pairing source index, target index, source image, target image
            used_targets.add(best_tgt_idx)
    
    return matches

#Generate the prompt for an image generator using ChatGPT
def create_prompt(traits):
    from openai import OpenAI
    import json

    client = OpenAI()
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
            "role": "system",
            "content": [
                {
                "type": "input_text",
                "text": f"Give the data as follows:\n {traits}\nCan you generate a prompt for me to create a family portrait photo from the description of each member above? Please consider the role of each member based on age. Because the prompt is used as the prompt input on the klingai.com platform, please refine it to match the platform's specifics"
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "output_text",
                "text": "{\n  \"result\": \"Create a family portrait photo featuring three members:\\n\\n1. **Father**: A 28-year-old white man, embodying a joyful, happy expression. He should be positioned in the center, exuding warmth and charm with a welcoming smile.\\n\\n2. **Younger Brother**: A 24-year-old white man with a neutral expression. Place him to the left of the father, standing confidently, showcasing a calm and relaxed demeanor.\\n\\n3. **Sister**: A 24-year-old white woman, sharing the same age as the younger brother, appearing happy. She should be on the father's right, with a bright, cheerful smile that adds balance and harmony to the family dynamic.\\n\\nArrange the composition to highlight their close bond, capturing the essence of family connection. Ensure the setting reflects a cozy and inviting atmosphere, such as a comfortable living room or a lovely outdoor garden.\"\n}"
                }
            ]
            }
        ],
        text={
            "format": {
            "type": "json_schema",
            "name": "response_format",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                "result": {
                    "type": "string",
                    "description": "The prompt placed here"
                }
                },
                "required": [
                "result"
                ],
                "additionalProperties": False
            }
            }
        },
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=True
    )
    response = response.model_dump()
    # Extract the 'result' property
    try:
        response_text = response['output'][0]["content"][0]["text"]  # Extract the raw JSON string
        response_json = json.loads(response_text)  # Parse the JSON string
        result = response_json.get("result", "No result found")  # Get the result key
        print("Got Extracted Result:\n", result)
    except (KeyError, json.JSONDecodeError) as e:
        print("Error extracting result:", e)


#Swap face in the source image with the faces in the destination image
def swap_faces(source_image, des_image_path):
    #source image
    source_image = "source_4.jpeg"
    des_image_path = "demo_file/multi_people.jpg"

    #Crop the face from source
    source_faces = crop_face(source_image, prefix_name="SRC")
    des_faces = crop_face(des_image_path, prefix_name="DST")

    matches = match_faces(source_faces, des_faces)

    import os
    import uuid

    # Display results
    index = 1
    unique_id = uuid.uuid4().hex[:8]  # Use first 8 characters of UUID
    swap_folder = f"swap/swap_{unique_id}"
    os.makedirs(swap_folder, exist_ok=True)
    print("Got swap_folder: ", swap_folder)

    for src_idx, tgt_idx, src, tgt in matches:
        print(f"Source Index: {src_idx}, Target Index: {tgt_idx}")    
        cv2.imwrite(f"{swap_folder}/SRC_0{index}.png", src)
        cv2.imwrite(f"{swap_folder}/DST_0{index}.png", tgt)
        index += 1

    #Swap multiple faces
    from test_wholeimage_swap_multispecific import execute
    from options.test_options import TestOptions

    opt = TestOptions().parse()
    opt.crop_size = crop_size
    opt.use_mask = True
    opt.name = "people"
    opt.gpu_ids = []
    opt.pic_b_path = source_image
    opt.output_path = f"./output/{swap_folder}"
    opt.multisepcific_dir = f"./{swap_folder}"

    os.makedirs(f"./output/{swap_folder}", exist_ok=True)

    execute(opt)

def get_images(path):
    from pathlib import Path

    folder_path = Path(path)

    # Get all image files in the folder
    image_files = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg"))
    
    # Read images using PIL    
    images =  [ cv2.imread(img_path) for img_path in image_files]        
        
    return images

# input_images = get_images("swap/input")
# input_images_traits = predidct_age_gender_race_emotion(input_images)
# input_traits = [item[1] for item in input_images_traits]
# print("Got input_traits: ", input_traits)

# create_prompt(json.dumps(input_traits))

# face_crop = Face_detect_crop(name="antelope", root='./insightface_func/models')
# face_crop.prepare(ctx_id=0 if torch.cuda.is_available() else -1)  # Set device

# image_path = "swap_multiple/SRC_01.png"
# img = cv2.imread(image_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_list, _ = face_crop.get(img, 224)
# print("Got check detect on SRC: ", img_list)

def train_face():
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from peft import get_peft_model, LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    import torch
    import os

    from huggingface_hub import login

    # Paste your Hugging Face token here
    login(token="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")

    # 1. Load Stable Diffusion Base
    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # 2. LoRA config
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["attn1", "attn2"],
        lora_dropout=0.05,
        bias="none",
        task_type="text2image"
    )

    # 3. Apply LoRA to text encoder
    pipe.text_encoder = get_peft_model(pipe.text_encoder, lora_config)

    # 4. Load face dataset (e.g., boy)
    dataset = load_dataset("imagefolder", data_dir="./face_training/ronaldo")
    dataloader = DataLoader(dataset["train"], batch_size=1, shuffle=True)

    # 5. Training loop
    optimizer = torch.optim.Adam(pipe.text_encoder.parameters(), lr=1e-5)
    pipe.train()

    for epoch in range(5):  # 5 epochs for fast training
        for batch in dataloader:
            inputs = pipe.feature_extractor(batch["image"][0].convert("RGB")).unsqueeze(0)
            prompt = ["a photo of <ronaldo>"]  # Custom token to inject
            tokens = pipe.tokenizer(prompt, return_tensors="pt").input_ids

            outputs = pipe.text_encoder(input_ids=tokens)
            loss = outputs.last_hidden_state.norm()  # dummy loss for structure
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print("✅ Training Complete!")    

# train_face()

