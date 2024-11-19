from PIL import Image
import requests
import numpy as np
import pandas
from Base_Agent import Base_Agent
from utils import compute_cos_similarity
from sentence_transformers import SentenceTransformer
import time
import numpy as np
from LLM_Interface import transformers_interface
from Environment import CookingEnv
# import LaViLa_Interface
# from utils import load_video_frames_cv2, get_video_metadata
import json
import torch
from typing import List, Dict

from transformers import Blip2Processor, Blip2Model, Blip2ForConditionalGeneration

# LLAMA_PATH = "/media/atkeonlab-3/Mass Storage/models/Llama-3-8b-chat-hf"
# MODEL_PATH = "/media/atkeonlab-3/Mass Storage/models"

# agent = Base_Agent(load_llm=True, load_narrator=False)


# # Load model directly
# from transformers import AutoModel
# model = AutoModel.from_pretrained("Efficient-Large-Model/Llama-3-VILA1.5-8B")

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

img1 = Image.open("/media/atkeonlab-3/Mass Storage/EPIC-KITCHENS_Processed/cropped_images/P01/P01_01/P01_01_0000000839_open_fridge.jpg")
img2 = Image.open("/media/atkeonlab-3/Mass Storage/EPIC-KITCHENS_Processed/cropped_images/P01/P01_01/P01_01_0000000983_holding_celery.jpg")
images = [img1, img2]

question = "Give a description for each of the images provided"
inputs = processor(images, question, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))



# from transformers import Blip2Processor, Blip2Model, Blip2ForConditionalGeneration
# torch.cuda.set_device(cuda_device)
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",do_rescale=False)
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
# model.eval()
# model.to(self.cuda_device)

# inputs = processor(images=frame, text=text_prompt, return_tensors="pt").to(cuda_device, torch.float16)
# generated_ids = model.generate(**inputs)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)


# from transformers import pipeline

# pipe = pipeline("text-generation", model="Efficient-Large-Model/Llama-3-VILA1.5-8B")

# model.generate("hi")

# Open the image


# # Display the image (optional)
# img.show()


# img_array = np.array(img)

