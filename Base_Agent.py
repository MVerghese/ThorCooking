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


class Base_Agent:
	def __init__(self,language_model = LLAMA_PATH, load_llm = True, load_narrator = True):
		if load_llm:
			self.llm = transformers_interface(model_name = MODEL_PATH + language_model, cuda_devices = [0])
		if load_narrator:
			self.narrator = LaViLa_Interface.LaViLa_Interface(load_nar = True, load_dual = False)
			self.frames_per_clip = 4
			print("loaded narrator")
		else:
			self.narrator = None
			self.frames_per_clip = 4
		# system_prompt_path = "Llama_chat_prompt.txt"
		# with open(system_prompt_path, 'r') as file:
		# 	self.system_prompt = file.read()

	def narrate(self,frames,num_sequences = 1):
		if isinstance(frames[0],np.ndarray):
			torch_images = []
			for image in images:
				torch_image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()
				torch_images.append(torch_image)
				torch_images = torch.cat(torch_images,dim=0)
		else:
			torch_images = frames
		
		torch_images = torch_images.permute(0,2,3,1)
		narrations = self.narrator.eval(torch_images,num_return_sequences=num_sequences)
		for i, narration in enumerate(narrations):
			narration = narration.replace("#C", "")
			narration = narration.replace("#c", "")
			narration = narration.replace("#O", "")
			narration = narration.replace("X ","")
			narration = narration.replace("Y ", "")
			narration = narration.replace("C ", "A person ")
			narration = narration.replace("c ", "A person ")
			narration = narration.lstrip().rstrip()
			narrations[i] = narration
		return narrations

	def narrate_video_unsegmented(self,video_path,narrate_frequency,narrations_per_clip=1):
		video_metadata = get_video_metadata(video_path)
		frame_rate = int(eval(video_metadata['avg_frame_rate']))
		duration = float(video_metadata['duration'])
		num_frames = int(duration*frame_rate)
		out = []
		frames_per_segment = narrate_frequency*frame_rate
		for i in range(0,num_frames,frames_per_segment):
			frame_nums = np.linspace(i,min(i+frames_per_segment, num_frames-1),self.frames_per_clip,dtype=int)
			frames = load_video_frames_cv2(video_path,frame_nums)
			print(frames.shape)
			narrations = self.narrate(frames,narrations_per_clip)
			print(narrations)
			narr_dict = {}
			narr_dict['start_frame'] = i
			narr_dict['stop_frame'] = i+frames_per_segment
			narr_dict['narrations'] = narrations
			out.append(narr_dict)
		return out
	
	'''
	Given a prompt and a set of predicates representing the environment, outputs the next most probable action
	Input:
		prompt: str - task to complete
		predicates: List[str] - environment state
		actions: List[Dict] - potential actions to take
	Output: str - most likely next action
	'''
	def generate_next_action(self, prompt:str, predicates: List[str], actions: List[Dict], action_history: List[str]) -> str:
		final_prompt = "Here is the task you are assigned to complete: " + prompt + "\nHere is the current state of the environment:\n"
		for predicate in predicates:
			final_prompt += predicate + "\n"

		final_prompt += "Here is your action history:\n"
		for act in action_history:
			final_prompt += act + "\n"

		final_prompt += "Based on the previous information, determine the next action needed to complete the task. Do not repeat actions."
		#  If you are done with the task, respond with \"done\"
		actions_list = []
		for act in actions:
			actions_list.append(act["action"].replace("Object", " " + act["objectType"] + " ").lower())

		actions_list.append("done")
		print(final_prompt)
		# print(actions_list)
		
		action_probabilities = self.llm.eval_log_probs(final_prompt, actions_list, batch_size=1)
		res_idx = max(enumerate(action_probabilities), key=lambda x: x[1])[0]
		if (res_idx == len(actions_list) - 1):
			return None
		return actions[res_idx]


def test_generate_next_action_blt():
	agent = Base_Agent(load_llm=True, load_narrator=False)
	env = CookingEnv()
	with open('Tasks/Make_A_BLT_0.json') as json_file:
		task_dict = json.load(json_file)
	env.load_task_state(task_dict, 8)
	action_history = []

	for i in range(9):
		action_history.append(task_dict["action_segments"][i]["action_text"])

	def filter_by_object(pred):
		pred = pred.lower()
		return "bacon" in pred or "lettuce" in pred or "tomato" in pred or "bread" in pred or "knife" in pred or "plate" in pred
	
	prompt = "Make a BLT"

	actions_list = env.generate_possible_actions(False)
	filtered_actions = list(filter(lambda x: filter_by_object(x["objectType"]), actions_list))

	predicates = env.generate_language_predicates()
	filtered_predicates = list(filter(filter_by_object, predicates))

	next_action = agent.generate_next_action(prompt, filtered_predicates, filtered_actions, action_history)
	
	if (next_action is None):
		return
	action_history.append(next_action["action"].replace("Object", " " + next_action["objectType"] + " ").lower())
	print(next_action)

	for _ in range(5):
		env.step(next_action)
		actions_list = env.generate_possible_actions(False)
		filtered_actions = list(filter(lambda x: filter_by_object(x["objectType"]), actions_list))

		predicates = env.generate_language_predicates()
		filtered_predicates = list(filter(filter_by_object, predicates))

		next_action = agent.generate_next_action(prompt, filtered_predicates, filtered_actions, action_history)
		print(next_action)
		if (next_action == None):
			return
		action_history.append(next_action["action"].replace("Object", " " + next_action["objectType"] + " ").lower())



def main():
	agent = Base_Agent(load_llm = False, load_narrator = True)
	video_path = "Videos/make_a_blt_0.mp4"
	narrate_frequency = 2
	narrations_per_clip = 10
	out = agent.narrate_video_unsegmented(video_path,narrate_frequency,narrations_per_clip)
	print(out)
	with open("blt_0_narrations.json", 'w') as file:
		json.dump(out,file)

	video_path = "Videos/make_a_blt_1.mp4"
	narrate_frequency = 2
	narrations_per_clip = 10
	out = agent.narrate_video_unsegmented(video_path,narrate_frequency,narrations_per_clip)
	print(out)
	with open("blt_1_narrations.json", 'w') as file:
		json.dump(out,file)

	video_path = "Videos/make_a_latte_0.mp4"
	narrate_frequency = 2
	narrations_per_clip = 10
	out = agent.narrate_video_unsegmented(video_path,narrate_frequency,narrations_per_clip)
	print(out)
	with open("latte_0_narrations.json", 'w') as file:
		json.dump(out,file)
if __name__ == "__main__":
	# main()
	test_generate_next_action_blt()





