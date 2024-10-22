import numpy as np
from LLM_Interface import transformers_interface
from Environment import CookingEnv
# import LaViLa_Interface
from utils import load_video_frames_cv2, get_video_metadata
import json
import torch
from typing import List, Dict
LLAMA_PATH = "gemma-2-2b-it"
MODEL_PATH = "/home/atkesonlab2/models/"
import time
from sentence_transformers import SentenceTransformer
from utils import compute_cos_similarity
import json
from Base_Agent import Base_Agent
LLAMA_PATH = "/home/atkesonlab2/models/Llama-3.2-1B-Instruct"
MODEL_PATH = "/home/atkesonlab2/models"

agent = Base_Agent(load_llm = True, load_narrator = False)
