import pandas
from Base_Agent import Base_Agent
from utils import compute_cos_similarity
from sentence_transformers import SentenceTransformer
import time
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
LLAMA_PATH = "/home/atkesonlab2/models/Llama-3.2-1B-Instruct"
MODEL_PATH = "/home/atkesonlab2/models"

agent = Base_Agent(load_llm=True, load_narrator=False)

all_annotations = pandas.read_csv("EPIC_100_train.csv")
grouped_annotations = [
    x for _, x in all_annotations.groupby(all_annotations['video_id'])]

action_sequences = []

for df in grouped_annotations:
    action_sequences.append(df['narration'].to_numpy())


chunk_size = 10
stride = 3

start_idx = 0
end_idx = start_idx

test_df = grouped_annotations[0]
while (start_idx < len(test_df)):
    end_idx = min(start_idx + chunk_size, len(test_df) - 1)

    actions = test_df['narration'][start_idx:end_idx].to_numpy()

    prompt = '''Summarize the following actions into a set of high level tasks. The summary must be one sentence long and summarize the entire lost of actions into a statement conveying its intended goal.
    Follow the following sample format exactly. Do not add any unnecessary sentences:
        If the action list is [open drawer, pick up fork, close drawer]
        The output summary will be: The drawer was opened to pick up a fork.
    Here are the actions:
    '''
    for action in actions:
        prompt += "\n" + action

    print(prompt)

    res = agent.llm.generate(prompt=prompt)
    print(res)
    break

    # update start and end
    start_idx = start_idx + stride
