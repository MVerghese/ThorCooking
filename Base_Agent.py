import numpy as np
from LLM_Interface import transformers_interface
import LaViLa_Interface
from utils import load_video_frames_cv2, get_video_metadata
import json
import torch
from sentence_transformers import SentenceTransformer
from utils import compute_cos_similarity
import json

LLAMA_PATH = "/media/mverghese/Mass Storage/models/Llama-3-8b-hf/"
MODEL_PATH = "/media/mverghese/Mass Storage/models/"


class Base_Agent:
	def __init__(self,language_model = "Llama-3-8b-chat-hf/", load_llm = True, load_narrator = True):
		if load_llm:
			self.llm = transformers_interface(model_name = LLAMA_PATH, cuda_devices = [0])
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
		cross_task_narrations_path = "cross_task_narrations.json"
		with open(cross_task_narrations_path, 'r') as file:
			self.cross_task_narrations = json.load(file)

		self.text_embedd_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1',device='cuda')

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

	def select_few_shot_examples(self,task, num_retrievals):
		tasks = [self.cross_task_narrations[video]["task"] for video in self.cross_task_narrations.keys()]
		tasks = list(set(tasks))
		task_embedds = self.text_embedd_model.encode(tasks)
		task_embedd = self.text_embedd_model.encode(task)
		task_similarity = compute_cos_similarity(task_embedds,task_embedd)
		sorted_idxs = np.argsort(task_similarity)[::-1]
		selected_idxs = sorted_idxs[:num_retrievals]
		selected_tasks = [tasks[i] for i in selected_idxs]
		selected_narrations = {}
		for task in selected_tasks:
			for video in self.cross_task_narrations.keys():
				if self.cross_task_narrations[video]["task"] == task:
					selected_narrations[task] = self.cross_task_narrations[video]["steps"]
					break
		return selected_narrations

	def build_prompt(self,task,narration_history,examples):
		prompt = "Your job is to plan the next steps to complete the task, here are some examples of task plans: \n"
		for example in examples.keys():
			example_prompt = "Task: " + example + "\n"
			example_prompt += "Steps: \n"
			example_prompt += "\n".join(examples[example])
			example_prompt += "\n"
			prompt += example_prompt

		prompt += "\n"
		prompt += "Current Task: " + task + "\n"
		prompt += "Steps: \n"
		for i, step in enumerate(narration_history):
			prompt += "Step {}: ".format(i+1) + step + "\n"
		prompt += "Step {}: ".format(len(narration_history)+1)
		return prompt

	def group_narrations_unordered(self, narrations, narration_embeddings, threshold=0.7):
		if isinstance(narration_embeddings, list):
			narration_embeddings = np.array(narration_embeddings)
		# print(narration_embeddings.shape[0])
		if narration_embeddings.shape[0] < 2:
			return np.zeros(narration_embeddings.shape[0],dtype=int), narrations
		cluster_ids = np.ones(narration_embeddings.shape[0],dtype=int)*-1
		cluster_ids[0] = 0
		new_cluster_index = 1
		for i in range(1,narration_embeddings.shape[0]):
			similarities = compute_cos_similarity(narration_embeddings[:i],narration_embeddings[i])
			max_similarity = np.max(similarities)
			if max_similarity > threshold:
				cluster_ids[i] = cluster_ids[np.argmax(similarities)]
			else:
				cluster_ids[i] = new_cluster_index
				new_cluster_index += 1
		
		cluster_data = []
		for i in range(np.max(cluster_ids)+1):
			rel_ids = np.where(cluster_ids == i)[0]
			rel_narrations = [narrations[j] for j in rel_ids]
			# Find the longest narration
			longest_narration = ""
			for narration in rel_narrations:
				if len(narration) > len(longest_narration):
					longest_narration = narration
			cluster_data.append(longest_narration)

		return cluster_ids, cluster_data

	def summarize_narrations(self, narrations, goal = None, token_limit = 4096):
		if goal == None:
			goal = "complete a task"
		
		num_prompt_tokens = token_limit +1
		frequency = 1
		while num_prompt_tokens > token_limit:
			print("Building prompt with frequency: ", frequency)
			
			if not isinstance(narrations, list):
				narration_list = narrations.split("\n")
				for i in range(len(narration_list)):
					if ". " in narration_list[i]:
						narration_list[i] = narration_list[i].split(". ")[1]
			else:
				narration_list = narrations
			
			keep_narrations = []
			for i in range(0,len(narration_list),frequency):
				keep_narrations.append(narration_list[i])
				
			narration_text = ""
			for i, narration in enumerate(keep_narrations):
				narration_text = narration_text + "%i. "%(i+1) + narration + "\n"
			

			
			prompt = "A person is currently attempting to " + goal + ". Their task is in progress and their goal is not yet complete. The following are low level narrations of their actions.\n\n"
			prompt = prompt + narration_text
			prompt = prompt + "\n\nPlease summarize these into a smaller set of high-level narrations. Focus on narrations that are relevant to the goal and do not include irrelevant narrations in your high-level summary. \nSummary: "
			# prompt = prompt + "\n\nFirst generate the steps to complete the users goal to " + goal + ", then list the steps towards the goal that the user has already completed. Begin the general task steps with '* A person' and begin the completed steps with the text, '1. A person ... 2. A person ...': \n"
			# with open("Llama_chat_prompt.txt", "r") as f:
			# 	system_prompt = f.read()
			# prompt = system_prompt.replace("{prompt}", prompt)
			# print(prompt)

			num_prompt_tokens = self.llm.get_num_tokens(prompt)
			print(num_prompt_tokens)
			frequency *= 2

		print(prompt)
		generated_text = self.llm.generate(prompt,num_tokens=400,use_cache=False)
		print("GENERATED TEXT")
		print(generated_text)
		# generated_text = "1. A person " + generated_text
		text_lines = generated_text.split("\n")
		good_lines = []
		for line in text_lines:
			# if the line starts with a number, keep it
			if len(line) > 0 and line[0].isdigit():
				good_lines.append(line)
		output_text = "\n".join(good_lines)
		return output_text

	def cure_narrations(self, narrations, goal = None, token_limit = 4096):
		object_prompt = "List all objects or nouns that might be involved in the task {} separated by commas: ".format(goal)
		generated_text = self.llm.generate(object_prompt,num_tokens=400,use_cache=False)
		print("GENERATED TEXT")
		print(generated_text)
		objects = generated_text.split(",")
		objects = [object.lstrip().rstrip() for object in objects]
		object_embeddings = self.text_embedd_model.encode(objects)
		

	def group_and_summarize_narration_history(self,task,narration_history):
		grouped_narrations = []
		for narration_block in narration_history:
			narration_embeddings = self.text_embedd_model.encode(narration_block)
			cluster_ids, cluster_data = self.group_narrations_unordered(narration_block,narration_embeddings)
			grouped_narrations += cluster_data
		print(grouped_narrations)
		summarized_narrations = self.summarize_narrations(grouped_narrations,goal = task)
		return summarized_narrations







def main():
	agent = Base_Agent(load_llm = True, load_narrator = False)
	# video_path = "Videos/make_a_blt_0.mp4"
	# narrate_frequency = 2
	# narrations_per_clip = 10
	# out = agent.narrate_video_unsegmented(video_path,narrate_frequency,narrations_per_clip)
	# print(out)
	# with open("blt_0_narrations.json", 'w') as file:
	# 	json.dump(out,file)

	# video_path = "Videos/make_a_blt_1.mp4"
	# narrate_frequency = 2
	# narrations_per_clip = 10
	# out = agent.narrate_video_unsegmented(video_path,narrate_frequency,narrations_per_clip)
	# print(out)
	# with open("blt_1_narrations.json", 'w') as file:
	# 	json.dump(out,file)

	# video_path = "Videos/make_a_latte_0.mp4"
	# narrate_frequency = 2
	# narrations_per_clip = 10
	# out = agent.narrate_video_unsegmented(video_path,narrate_frequency,narrations_per_clip)
	# print(out)
	# with open("latte_0_narrations.json", 'w') as file:
	# 	json.dump(out,file)
	# examples = agent.select_few_shot_examples("make a blt",3)
	# print(examples)
	# prompt = agent.build_prompt("make a blt",["Get bread","Get lettuce","Get tomato"],examples)
	# print(prompt)
	narration_path = "blt_0_narrations.json"
	with open(narration_path, 'r') as file:
		narration_list = json.load(file)
	narration_list = [narration['narrations'] for narration in narration_list]
	# print(narration_list)
	summarized_narrations = agent.group_and_summarize_narration_history("make a blt",narration_list)
	print(summarized_narrations)
if __name__ == "__main__":
	main()




			
