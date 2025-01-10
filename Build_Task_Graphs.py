import numpy as np
from LLM_Interface import transformers_interface
from sentence_transformers import SentenceTransformer
import json
from utils import compute_cos_similarity
from tqdm import tqdm



LLAMA_PATH = "/home/mverghese/Models/Llama-3.2-11B-Vision-Instruct/"

def parse_generated_steps(steps):
		separated_steps = steps.split("\n")
		clean_steps = []
		for step in separated_steps:
			if step[0].isdigit() and len(step.split(". ")) >= 2:
				parsed_step = step.split(". ")[1]
				if len(parsed_step) > 0:
					clean_steps.append(parsed_step)
		return clean_steps

def generate_task_sequences(task, num_prompt_examples, num_generated_sequences, lam=.8):
	chat_system_prompt = "You are an assitant designed to help people with household tasks. Do not be verbose. Answer the question with no added qualifications or caveats. Just directly provide the answer."
	llm = transformers_interface(model_name=LLAMA_PATH)
	goal_step_sequences_path = "/home/mverghese/ThorCooking/Dataset_Annotations/goalstep_annontations_processed.json"
	goal_step_goals_path = "/home/mverghese/ThorCooking/Dataset_Annotations/goals.json"
	goal_step_goal_embeddings_path = "/home/mverghese/ThorCooking/Dataset_Annotations/goal_embeddings.npy"
	with open(goal_step_sequences_path, "r") as file:
		goal_step_sequences = json.load(file)
	with open(goal_step_goals_path, "r") as file:
		goal_step_goals = json.load(file)
	goal_step_goal_embeddings = np.load(goal_step_goal_embeddings_path)
	text_embedd_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1", device="cuda")
	self_similarities = np.dot(goal_step_goal_embeddings, goal_step_goal_embeddings.T)
	print(self_similarities.shape)


	prompt = "The following are examples of tasks and a set of steps to complete each task: \n"
	task_embedd = text_embedd_model.encode(task)
	similarities = compute_cos_similarity(goal_step_goal_embeddings, task_embedd)
	selected_indices = [] 
	for i in range(num_prompt_examples):
		if i == 0:
			selected_indices.append(np.argmax(similarities))
		else:
			relevant_similarities = self_similarities[selected_indices,:]
			max_rel_similarities = np.max(relevant_similarities, axis=0)
			mmr_similarities = lam*similarities - (1-lam)*max_rel_similarities
			mmr_similarities[selected_indices] = -1
			selected_indices.append(np.argmax(mmr_similarities))

	selected_goals = [goal_step_goals[i] for i in selected_indices]
	print(selected_goals)
	selected_sequences = []
	for goal in selected_goals:
		for sequence in goal_step_sequences:
			if sequence["goal"] == goal:
				selected_sequences.append(sequence)
				break

	for i in range(num_prompt_examples):
		prompt += "Task: " + selected_goals[i] + "\n"
		prompt += "Steps: \n"
		for i, step in enumerate(selected_sequences[i]["steps"]):
			prompt += "{}. ".format(i + 1) + step + "\n"

		prompt += "\n\n"

	prompt += "Generate the steps for the following task: \n"
	prompt += "Task: " + task + "\n"
	prompt += "Steps: \n"
	print(prompt)
	generated_sequences = []
	for i in tqdm(range(num_generated_sequences)):
		generated_sequence = llm.generate_chat_easy(chat_system_prompt, prompt, num_tokens=400)[0]
		# print(generated_sequence)
		generated_sequences.append(parse_generated_steps(generated_sequence))
	return generated_sequences

def generate_task_sequences_promptless(task, num_generated_sequences):
	chat_system_prompt = "You are an assitant designed to help people with household tasks. Do not be verbose. Answer the question with no added qualifications or caveats. Just directly provide the answer."
	llm = transformers_interface(model_name=LLAMA_PATH)
	prompt = "What are the steps to " + task + "?" + "\n"
	prompt += "List each step in a new line with a number. For example: 1. " + "\n"
	generated_sequences = []
	for i in tqdm(range(num_generated_sequences)):
		generated_sequence = llm.generate_chat_easy(chat_system_prompt, prompt, num_tokens=400)[0]
		generated_sequences.append(parse_generated_steps(generated_sequence))
	return generated_sequences


def generate_task_graph_dict(sequences, cluster_threshold = 0.9):
	text_embedd_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1", device="cuda")
	task_graph_dict = {}
	task_graph_steps = []
	task_graph_embeddings = np.zeros((0,768))
	print("Building task graph")
	for sequence in tqdm(sequences):
		for i, step in enumerate(sequence[:-1]):
			step_embedd = text_embedd_model.encode(step)
			if task_graph_embeddings.shape[0] == 0:
				task_graph_embeddings = np.vstack((task_graph_embeddings, step_embedd))
				task_graph_steps.append(step)
				task_graph_dict[step] = []
				task_graph_dict[step].append(sequence[i + 1])
			else:
				similarities = compute_cos_similarity(task_graph_embeddings, step_embedd)
				max_sim_step_idx = np.argmax(similarities)
				if similarities[max_sim_step_idx] > cluster_threshold:
					task_graph_dict[task_graph_steps[max_sim_step_idx]].append(sequence[i + 1])
				else:
					task_graph_embeddings = np.vstack((task_graph_embeddings, step_embedd))
					task_graph_steps.append(step)
					task_graph_dict[step] = []
					task_graph_dict[step].append(sequence[i + 1])
	print("Task graph has {} keys".format(len(task_graph_dict.keys())))
	print("Cleaning up duplicate steps in task graph")
	next_steps_lens = []
	for key in tqdm(task_graph_dict.keys()):
		next_steps = task_graph_dict[key]
		next_steps_lens.append(len(next_steps))
		if len(next_steps) == 0:
			del task_graph_dict[key]
		next_step_embedds = text_embedd_model.encode(next_steps)
		self_similarities = np.dot(next_step_embedds, next_step_embedds.T)
		step_delete_indices = []
		for i in range(1, len(next_steps)):
			prev_similarities = self_similarities[i,:i]
			if np.max(prev_similarities) > cluster_threshold:
				step_delete_indices.append(i)
		if len(step_delete_indices) > 0:
			step_delete_indices.reverse()
			for i in step_delete_indices:
				del next_steps[i]
		task_graph_dict[key] = next_steps
	print("Average number of next steps is {}".format(np.mean(next_steps_lens)))
	return task_graph_dict



if __name__ == '__main__':
	# generated_sequences = generate_task_sequences_promptless("Make a bacon lettuce and tomato sandwich. The ingredients are uncooked bacon, whole tomatoes, a whole head of lettuce, a loaf of sliced bread, and a jar of mayonnaise.", 100)
	# with open("Generated_Steps_Ingredients.json", "w") as file:
	# 	json.dump(generated_sequences, file)

	with open("Generated_Steps_Ingredients.json", "r") as file:
		generated_sequences = json.load(file)
	task_graph_dict = generate_task_graph_dict(generated_sequences, 0.7)
	with open("BLT_Generated_Task_Graph_Ingredients.json", "w") as file:
		json.dump(task_graph_dict, file)