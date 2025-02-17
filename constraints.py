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
import csv
from tqdm import tqdm

# messages format
# messages = [
#   {"role": "system", "content": "You are a bot that responds to weather queries."},
#   {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
# ]

LLAMA_PATH = "/media/atkeonlab-3/Mass Storage/models/Llama-3-8b-chat-hf"
MODEL_PATH = "/media/atkeonlab-3/Mass Storage/models"

agent = Base_Agent(load_llm=True, load_narrator=False)

all_annotations = pandas.read_csv("EPIC_100_train.csv")
grouped_annotations = [
    x for _, x in all_annotations.groupby(all_annotations["video_id"])
]

action_sequences = []

for df in grouped_annotations:
    action_sequences.append(df["narration"].to_numpy())


chunk_size = 5
stride = 5
# chunk_size = stride - overlap of one

start_idx = 0
end_idx = start_idx


"""
        Follow the following sample format exactly. Do not add any unnecessary sentences.
        In this sample, your input will be labeled "INPUT:" and the sample output will be labeled "OUTPUT:" . Your response should be formatted the same way as the output.
            INPUT: [open drawer, pick up fork, close drawer, pick up pasta, eat pasta]
            OUTPUT: Pasta was eaten with a fork from the drawer.


                    For example, if the input is [open drawer, pick up fork, close drawer, pick up pasta, eat pasta], You must output { "goal": Pasta was eaten with a fork from the drawer. }
            """

data = []

for current_df in grouped_annotations:
    while tqdm(start_idx < len(current_df)):
        end_idx = min(start_idx + chunk_size, len(current_df) - 1)
        print(start_idx, end_idx)

        actions = current_df["narration"][start_idx:end_idx].to_numpy()

        prompt = """What is a high level summary these actions? Your answer must be formatted as a JSON object. Only use information present in the actions. Keep it concise.
Here is the format of the JSON object you must follow: { "summary": summary of actions provided enclosed in quotation marks }
For example, if the input is [open drawer, pick up fork, close drawer, pick up pasta, eat pasta], You must output { "summary": Pasta was eaten with a fork from the drawer. }
Here is the list of actions to summarize into a JSON object:
["""

        for action in actions:
            prompt += action + ", "
        prompt = (
            prompt[:-2]
            + """]
Generated summary dictionary from the above actions:"""
        )

        messages = [
            {
                "role": "system",
                "content": "You are a bot that outputs concise JSON-formatted responses.",
            },
            {"role": "user", "content": prompt},
        ]

        # print(prompt)
        # print ("------------------------------------------------")

        # res = agent.llm.generate(prompt=prompt, num_tokens=20)
        res = agent.llm.generate(agent.llm.generate_chat(messages), num_tokens=500)
        print(res[0])
        start_idx_res = max(res.find("{"), 0)
        end_idx_res = res.find("}")
        if end_idx_res == -1:
            end_idx_res = len(res)
        else:
            end_idx_res = min(end_idx_res + 1, len(res))

        res = res[start_idx_res:end_idx_res]
        # print(res)

        try:
            res_dict = json.loads(res)
            res = res_dict["summary"]
        except:
            for _ in range(5):
                # res = agent.llm.generate(prompt=prompt, num_tokens=20)
                res = agent.llm.generate(
                    agent.llm.generate_chat(messages), num_tokens=20
                )
                res = res[max(res.find("{"), 0) : min(res.find("}") + 1, len(res))]
                try:
                    res_dict = json.loads(res)
                    res = res_dict["summary"]
                    break
                except:
                    print("not json")

        idx_colon = res.find(":")
        if idx_colon != -1:
            res = res[idx_colon:-1]

        res = res.replace("'", "")
        res = res.replace('"', "")
        res = res.replace("{", "")
        res = res.replace("}", "")

        csv_data = {
            "narration_id": current_df["narration_id"][start_idx],
            "participant_id": current_df["participant_id"][start_idx],
            "video_id": current_df["video_id"][start_idx],
            "narration_timestamp": current_df["narration_timestamp"][start_idx],
            "start_timestamp": current_df["start_timestamp"][start_idx],
            "stop_timestamp": current_df["stop_timestamp"][end_idx],
            "start_frame": current_df["start_frame"][start_idx],
            "stop_frame": current_df["stop_frame"][end_idx],
            "summary": res,
        }

        data.append(csv_data)

        # break

        # update start and end
        start_idx = start_idx + stride
    break


with open("summaries55_generate_chat_test.csv", "w", newline="") as csvfile:
    fieldnames = [
        "narration_id",
        "participant_id",
        "video_id",
        "narration_timestamp",
        "start_timestamp",
        "stop_timestamp",
        "start_frame",
        "stop_frame",
        "summary",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(data)
