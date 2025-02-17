import numpy as np
from LLM_Interface import transformers_interface

# import LaViLa_Interface
from utils import load_video_frames_cv2, get_video_metadata, compute_entropy
import json
import torch
from sentence_transformers import SentenceTransformer
from utils import compute_cos_similarity
import json

LLAMA_PATH = "/home/mverghese/Models/Llama-3.1-8B-Instruct/"
LLAMA_PATH = "/home/mverghese/Models/Llama-3.2-11B-Vision-Instruct/"

MODEL_PATH = "/media/mverghese/Mass Storage/models/"




class Base_Agent:
	def __init__(
		self,
		language_model="Llama-3-8b-chat-hf/",
		load_llm=True,
		selection_method="prob",
		use_chat_prompt=False,
		build_task_templates = False
	):
		if load_llm:
			print("Loading LLM {}".format(LLAMA_PATH.split("/")[-2]))
			self.llm = transformers_interface(model_name=LLAMA_PATH, cuda_devices=[0], vision_model=True)
		# system_prompt_path = "Llama_chat_prompt.txt"
		# with open(system_prompt_path, 'r') as file:
		# 	self.system_prompt = file.read()
		cross_task_narrations_path = "Narrations/cross_task_narrations.json"
		goal_step_sequences_path = "/home/mverghese/ThorCooking/Dataset_Annotations/goalstep_annontations_processed.json"
		goal_step_goals_path = "/home/mverghese/ThorCooking/Dataset_Annotations/goals.json"
		goal_step_goal_embeddings_path = "/home/mverghese/ThorCooking/Dataset_Annotations/goal_embeddings.npy"
		with open(cross_task_narrations_path, "r") as file:
			self.cross_task_narrations = json.load(file)
		with open(goal_step_sequences_path, "r") as file:
			self.goal_step_sequences = json.load(file)
		with open(goal_step_goals_path, "r") as file:
			self.goal_step_goals = json.load(file)
		self.goal_step_goal_embeddings = np.load(goal_step_goal_embeddings_path)


		self.text_embedd_model = SentenceTransformer(
			"multi-qa-mpnet-base-cos-v1", device="cuda"
		)
		self.prob_scaling_factor = 3
		self.selection_method = selection_method
		self.use_chat_prompt = use_chat_prompt
		self.chat_system_prompt = "You are an assitant designed to help people with household tasks. Do not be verbose. Answer the question with no added qualifications or caveats. Just directly provide the answer."
		self.step_similarity_threshold = 0.5
		self.task_graph_path = None
		self.video_history_path = None

	def set_video_history_path(self, video_history_path):
		self.video_history_path = video_history_path
	
	def set_task_graph_path(self, task_graph_path):
		self.task_graph_path = task_graph_path
		with open(self.task_graph_path, "r") as file:
			self.task_graph = json.load(file)
		self.task_graph_keys = list(self.task_graph.keys())
		self.task_graph_embedds = self.text_embedd_model.encode(self.task_graph_keys)

	def generate_task_sequences(self,task,num_prompt_examples = 3, num_generated_sequences = 10):
		prompt = "The following are examples of tasks and a set of steps to complete each task: \n"
		task_embedd = self.text_embedd_model.encode(task)
		similarity = compute_cos_similarity(self.goal_step_goal_embeddings, task_embedd)
		sorted_idxs = np.argsort(similarity)[::-1]
		
		selected_idxs = sorted_idxs[:num_prompt_examples]
		selected_goals = [self.goal_step_goals[i] for i in selected_idxs]
		selected_sequences = []
		for goal in selected_goals:
			for sequence in self.goal_step_sequences:
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
		for i in range(num_generated_sequences):
			generated_sequence = self.llm.generate_chat_easy(self.chat_system_prompt, prompt, num_tokens=400)[0]
			print(generated_sequence)
			generated_sequences.append(self.parse_generated_steps(generated_sequence))
		return generated_sequences

	def parse_generated_steps(self,steps):
		separated_steps = steps.split("\n")
		clean_steps = []
		for step in separated_steps:
			if step[0].isdigit():
				parsed_step = step.split(". ")[1]
				if len(parsed_step) > 0:
					clean_steps.append(parsed_step)
		return clean_steps

	def add_step_library(self,step_list):
		if isinstance(step_list[0], list):
			new_step_list = []
			for step in step_list:
				new_step_list += step
			step_list = new_step_list
		self.step_library = step_list
		self.step_library_embedds = self.text_embedd_model.encode(step_list)



	def select_few_shot_examples(self, task, num_retrievals):
		tasks = [
			self.cross_task_narrations[video]["task"]
			for video in self.cross_task_narrations.keys()
		]
		tasks = list(set(tasks))
		task_embedds = self.text_embedd_model.encode(tasks)
		task_embedd = self.text_embedd_model.encode(task)
		task_similarity = compute_cos_similarity(task_embedds, task_embedd)
		sorted_idxs = np.argsort(task_similarity)[::-1]
		selected_idxs = sorted_idxs[:num_retrievals]
		selected_tasks = [tasks[i] for i in selected_idxs]
		selected_narrations = {}
		for task in selected_tasks:
			for video in self.cross_task_narrations.keys():
				if self.cross_task_narrations[video]["task"] == task:
					selected_narrations[task] = self.cross_task_narrations[video][
						"steps"
					]
					break
		return selected_narrations

	def build_prompt(self,task,narration_history,examples,mode="mcq",last_failed_action="", additional_prompt="", use_few_shot_examples = True):
		# Modes are:
		# mcq: for multiple choice questions
		# yn-chat: for yes/no questions using chat prompting
		# yn: for yes/no questions without chat prompting
		if mode == "mcq":
			if use_few_shot_examples:
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
			else: 
				prompt = "A user is attempting to complete the task: " + task + ". The steps completed so far are: \n"
			
			for i, step in enumerate(narration_history):
				prompt += "Step {}: ".format(i + 1) + step + "\n"

			if last_failed_action != "":
				prompt += "The last action, " + last_failed_action + " failed.\n"
			if additional_prompt != "":
				prompt += additional_prompt
			prompt += "Which of the following is the best next step to complete the task? Answer with the corresponding capital letter."

			return prompt

		elif mode == "yn":
			if use_few_shot_examples:
				prompt = "USER: Your job is to plan the next steps to complete the task, here are some examples of task plans: \n"
				for example in examples.keys():
					example_prompt = "Task: " + example + "\n"
					example_prompt += "Steps: \n"
					example_prompt += "\n".join(examples[example])
					example_prompt += "\n"
					prompt += example_prompt

				prompt += "\n"
				prompt += "Current Task: " + task + "\n"
				prompt += "Steps: \n"
			else: 
				prompt = "USER: I am attempting to complete the task: " + task + ". The steps completed so far are: \n"
			for i, step in enumerate(narration_history):
				prompt += "Step {}: ".format(i + 1) + step + "\n"

			if last_failed_action != "":
				prompt += "The last action, " + last_failed_action + " failed.\n"
			if additional_prompt != "":
				prompt += additional_prompt
			prompt += "Is [action] an appropriate next step to complete the task? ASSITANT: "

			return prompt
		elif mode == "yn-chat":
			if use_few_shot_examples:
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
			else: 
				prompt = "A user is attempting to complete the task: " + task + ". The steps completed so far are: \n"
			for i, step in enumerate(narration_history):
				prompt += "Step {}: ".format(i + 1) + step + "\n"
			
			if last_failed_action != "":
				prompt += "The last action, " + last_failed_action + " failed.\n"
			if additional_prompt != "":
				prompt += additional_prompt
			prompt += "Is [action] an appropriate next step to complete the task?"

			return prompt	

	def select_action_prob(self,task,action_list,history,segment_dict,probablistic_selection=False,verbose=False,last_failed_action="",additional_prompt="",return_probs=False, num_candidate_actions = 5):
		examples = self.select_few_shot_examples(task, 3)
		print("Action History: ", history)

		if len(action_list) > num_candidate_actions:

			prompt = self.build_prompt(task,history,examples,mode="yn",last_failed_action=last_failed_action,additional_prompt=additional_prompt, use_few_shot_examples = True)
			# print("Prompt: ", prompt)
			initial_action_probs = self.llm.action_probs_yn(prompt, action_list, use_chat = False)
			top_action_idxs = np.argsort(initial_action_probs)[::-1]
			candidate_actions = [action_list[i] for i in top_action_idxs[:num_candidate_actions]]

		else:
			candidate_actions = action_list
			initial_action_probs = np.ones(len(action_list))

		if verbose:
			# print("Candidate action rankings")
			# for i in top_action_idxs:
			# 	print("Action: ", action_list[i], "Prob: ", initial_action_probs[i])
			print("Candidate Actions: ", candidate_actions)

		prompt = self.build_prompt(task,history,examples,mode="mcq",last_failed_action=last_failed_action,additional_prompt=additional_prompt,  use_few_shot_examples = False)
		# print("Prompt: ", prompt)
		action_probs = self.llm.action_probs_mcq(prompt, candidate_actions, use_chat = True)
		if verbose:
			top_action_idxs = np.argsort(action_probs)[::-1]
			print("Top action probs")
			for i in top_action_idxs[:10]:
				print("Action: ", candidate_actions[i], "Prob: ", action_probs[i])
		# action_probs = initial_action_probs
		if last_failed_action in candidate_actions:
			action_probs[candidate_actions.index(last_failed_action)] = 0

		if not probablistic_selection:
			best_action_probs = np.max(action_probs)
			# best_action = candidate_actions[np.argmax(action_probs)]
			best_action = candidate_actions[np.argmax(action_probs)]


			# print("selection_action_prob thinks best action is: ", best_action)
			best_index = action_list.index(best_action)

		else:
			probs = np.exp(similarity - 1)
			best_index = np.random.choice(len(action_list), p=probs)
		# print("Action: ", action_list[best_index])
		# print("Action list at the end: ", action_list, len(action_list))
		if return_probs:
			return action_list[best_index], best_action_probs, best_index, action_probs
		else:
			return action_list[best_index], best_action_probs, best_index


	def select_action_prob_test(self,task,action_list,history,segment_dict,probablistic_selection=False,verbose=False,last_failed_action="",additional_prompt="",return_probs=False):
		print("Verbose is set to: ", verbose)
		prompt = "A user is attempting to complete the task: " + task + ". The steps completed so far are: \n"
		for i, step in enumerate(history):
			prompt += "Step {}: ".format(i + 1) + step + "\n"
		if additional_prompt != "":
			prompt += "Here is some additional information about the state of the task: \n" + additional_prompt
		prompt += "Which of the following is the best next step to complete the task? Answer with a single capital letter."

		print("Action History: ")
		print(history)
		# if verbose:
		print("Prompt: ", prompt)
		action_probs = self.llm.action_probs_mcq(prompt, action_list, use_chat = True)
		if verbose:
			top_action_idxs = np.argsort(action_probs)[::-1]
			for i in top_action_idxs[:10]:
				print("Action: ", action_list[i], "Prob: ", action_probs[i])
		similarity = action_probs
		# probs = np.exp(similarity-1)
		if last_failed_action in action_list:
			similarity[action_list.index(last_failed_action)] = 0

		if not probablistic_selection:
			best_index = np.argmax(similarity)
		else:
			probs = np.exp(similarity - 1)
			best_index = np.random.choice(len(action_list), p=probs)
		# print("Action: ", action_list[best_index])
		if return_probs:
			return action_list[best_index], similarity[best_index], best_index, action_probs
		else:
			return action_list[best_index], similarity[best_index], best_index

	def select_action_iterative(self,task,action_list,history,segment_dict,probablistic_selection=False,verbose=False,last_failed_action="",additional_prompt="",return_probs=False, num_candidate_actions = 5):
		assert self.video_history_path != None, "Video history path not set"
		if len(action_list) <= 1:
			return self.select_action_prob(task,candidate_actions,history,segment_dict,probablistic_selection=False,verbose=False,last_failed_action=last_failed_action,additional_prompt="",return_probs=return_probs)

		examples = self.select_few_shot_examples(task, 3)
		print("Action History: ", history)
		prompt = self.build_prompt(task,history,examples,mode="yn",last_failed_action=last_failed_action,additional_prompt=additional_prompt, use_few_shot_examples = True)
		# print("Prompt: ", prompt)
		initial_action_probs = self.llm.action_probs_yn(prompt, action_list, use_chat = False)
		top_action_idxs = np.argsort(initial_action_probs)[::-1]
		candidate_actions = [action_list[i] for i in top_action_idxs[:num_candidate_actions]]

		initial_prompt = "Your job is to plan a set of steps to complete the task: " + task + ". The actions completed so far are: \n"
		for i, action in enumerate(history):
			initial_prompt += "{}. ".format(i + 1) + action + "\n"
		initial_prompt += "The next possible actions are: " + ", ".join(candidate_actions) + "\n"
		initial_prompt += "What question can you ask about the steps completed so far that would help decide what action to take next?"
		messages = []
		messages.append({"role": "system", "content": self.chat_system_prompt})
		messages.append({"role": "user", "content": initial_prompt})
		results = self.select_action_prob(task,candidate_actions,history,segment_dict,probablistic_selection=False,verbose=False,last_failed_action=last_failed_action,additional_prompt="",return_probs=True)
		inital_action_probs = results[3]
		initial_action_probs /= np.sum(initial_action_probs)
		if np.max(inital_action_probs) > 0.7:
			return results
		for i, action in enumerate(candidate_actions):
			print("Action: ", action, "Probability: ", inital_action_probs[i])

		inital_entropy = compute_entropy(inital_action_probs)
		print("Initial Entropy: ", inital_entropy)
		best_overall_question = ""
		best_overall_answer = ""
		best_overall_entropy = np.inf
		for i in range(5):
			print("Iteration: ", i)
			print("Prompt: ", messages)
			question = self.llm.generate_chat(messages, num_tokens=400)[0]
			messages.append({"role": "assitant", "content": question})
			answers = self.question_answer(question,5,segment_dict)
			print("Question: ", question)
			print("Answers: ", answers)
			best_entropy = np.inf
			best_answer = ""
			for answer in answers:
				additional_prompt = "Question: " + question + "\n" + "Answer: " + answer + "\n"
				_, _, _, action_probs = self.select_action_prob(task,candidate_actions,history,segment_dict,probablistic_selection=False,verbose=False,last_failed_action=last_failed_action,additional_prompt=additional_prompt,return_probs=True)
				entropy = compute_entropy(action_probs)
				if entropy < best_entropy:
					best_entropy = entropy
					best_answer = answer
			print("Question: ", question)
			print("Answer: ", best_answer)
			print("Entropy: ", best_entropy)
			if best_entropy < best_overall_entropy:
				best_overall_entropy = best_entropy
				best_overall_question = question
				best_overall_answer = best_answer
			if best_entropy >= inital_entropy:
				next_prompt = "Answer: " + best_answer + "\n" + "This was not a helpful question and answer, what's another question you can ask?"
				messages.append({"role": "user", "content": next_prompt})
			else:
				if (inital_entropy - best_entropy)/inital_entropy < 0.25:
					next_prompt = "Answer: " + best_answer + "\n" + "This was a helpful question and answer, what's another question you can ask?"
					messages.append({"role": "user", "content": next_prompt})
				else:
					break
		print("Best Question: ", best_overall_question)
		print("Best Answer: ", best_overall_answer)
		additional_prompt = "Question: " + best_overall_question + "\n" + "Answer: " + best_overall_answer + "\n"
		return self.select_action_prob(task,candidate_actions,history,segment_dict,probablistic_selection=False,verbose=False,last_failed_action=last_failed_action,additional_prompt=additional_prompt,return_probs=return_probs)

	def question_answer(self, question, num_responses,segment_dict):
		question_embedd = self.text_embedd_model.encode(question)
		history_segments = [key for key in segment_dict.keys()]
		history_segments_embedds = self.text_embedd_model.encode(history_segments)
		similarity = compute_cos_similarity(history_segments_embedds,question_embedd.flatten())
		best_index = np.argmax(similarity)
		best_segment = history_segments[best_index]
		best_segment_range = segment_dict[best_segment]
		answers, _ = self.llm.SeViLa_VQA(question,best_segment_range,self.video_history_path,num_responses = num_responses)
		print("Num answers: ", len(answers))
		return answers

	def select_action_iterative_test(self,task,action_list,history,segment_dict,probablistic_selection=False,verbose=False,last_failed_action="",additional_prompt="",return_probs=False):
		assert self.video_history_path != None, "Video history path not set"
		# initial_prompt = "Your job is to plan a set of steps to complete the task: " + task + ". The actions completed so far are: \n"
		# for i, action in enumerate(history):
		# 	initial_prompt += "{}. ".format(i + 1) + action + "\n"
		# initial_prompt += "The next possible actions are: " + ", ".join(action_list) + "\n"
		# initial_prompt += "What question can you ask about the steps completed so far that would help decide what action to take next?"
		# messages = []
		# messages.append({"role": "system", "content": self.chat_system_prompt})
		# messages.append({"role": "user", "content": initial_prompt})
		_, _, _, inital_action_probs = self.select_action_prob_test(task,action_list,history,segment_dict,probablistic_selection=False,verbose=False,last_failed_action=last_failed_action,additional_prompt="",return_probs=True)
		sorted_action_idxs = np.argsort(inital_action_probs)[::-1]
		for i in sorted_action_idxs:
			print("Action: ", action_list[i], "Probability: ", inital_action_probs[i])
		
		inital_entropy = compute_entropy(inital_action_probs)
		print("Initial Entropy: ", inital_entropy)

		additional_prompt = "Question: " + "Can the tomato be added to the sandwich" + "\n" + "Answer: " + "No, the tomato is whole it must be sliced before being added" + "\n"
		# additional_prompt = "Choose Slice the Lettuce next\n"
		_, _, _, action_probs =  self.select_action_prob_test(task,action_list,history,segment_dict,probablistic_selection=False,verbose=False,last_failed_action=last_failed_action,additional_prompt=additional_prompt,return_probs=return_probs)
		sorted_action_idxs = np.argsort(action_probs)[::-1]
		for i in sorted_action_idxs:
			print("Action: ", action_list[i], "Probability: ", action_probs[i])
		entropy = compute_entropy(action_probs)
		print("Entropy: ", entropy)

	def select_action(self,task,action_list,history,segment_dict,probablistic_selection=False,verbose=False,last_failed_action="",additional_prompt="",return_probs=False, filter_actions = False):
		# This function is a bit of a mess right now as it hasy varuible numbers of return values based on the arguments
		# TODO: Fix this so that it returns a consistent number of values (requires updating evaluation.py as well  )
		# The filter actions flag filters actions based on the task graph
		if filter_actions:
			last_action = history[-1]
			last_action_embedd = self.text_embedd_model.encode(last_action)
			task_graph_keys_similariity = compute_cos_similarity(self.task_graph_embedds,last_action_embedd)
			last_action_task_graph = self.task_graph_keys[np.argmax(task_graph_keys_similariity)]
			possible_actions = self.task_graph[last_action_task_graph]
			possible_action_embedds = self.text_embedd_model.encode(possible_actions)
			action_list_embedds = self.text_embedd_model.encode(action_list)
			similarity_matrix = np.dot(possible_action_embedds,action_list_embedds.T)
			similarity = np.max(similarity_matrix,axis=0)
			valid_actions = [action_list[i] for i in np.where(similarity > self.step_similarity_threshold)[0]]
			print("Valid Actions: ", valid_actions)
			if self.selection_method == "sim":
				results =  self.select_action_sim(task,valid_actions,history,segment_dict,probablistic_selection,verbose,last_failed_action,additional_prompt,return_probs)
			elif self.selection_method == "prob":
				results =  self.select_action_prob(task,valid_actions,history,segment_dict,probablistic_selection,verbose,last_failed_action,additional_prompt,return_probs)
			elif self.selection_method == "iter":
				results =  self.select_action_iterative(task,valid_actions,history,segment_dict,probablistic_selection,verbose,last_failed_action,additional_prompt,return_probs)
			elif self.selection_method == "iter_test":
				results =  self.select_action_iterative_test(task,valid_actions,history,segment_dict,probablistic_selection,verbose,last_failed_action,additional_prompt,return_probs)
			best_index = action_list.index(results[0])
			# print("selection_action thinks best action is: ", action_list[best_index])
			if return_probs:
				return results[0], results[1], best_index, valid_actions, results[3]
			else:
				return results[0], results[1], best_index, valid_actions
		else:
			if self.selection_method == "sim":
				return self.select_action_sim(task,action_list,history,segment_dict,probablistic_selection,verbose,last_failed_action,additional_prompt,return_probs)
			elif self.selection_method == "prob":
				return self.select_action_prob(task,action_list,history,segment_dict,probablistic_selection,verbose,last_failed_action,additional_prompt,return_probs)
			elif self.selection_method == "iter":
				return self.select_action_iterative(task,action_list,history,segment_dict,probablistic_selection,verbose,last_failed_action,additional_prompt,return_probs)
			elif self.selection_method == "iter_test":
				results =  self.select_action_iterative_test(task,valid_actions,history,segment_dict,probablistic_selection,verbose,last_failed_action,additional_prompt,return_probs)
			else:
				raise ValueError("Invalid selection method")


def main():
	agent = Base_Agent(load_llm=True, load_narrator=False)
	print(agent.generate_task_sequences("make a bacon lettuce and tomato sandwich", num_prompt_examples=5, num_generated_sequences=10))
	1/0	
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
	with open(narration_path, "r") as file:
		narration_list = json.load(file)
	narration_list = [narration["narrations"] for narration in narration_list]
	# print(narration_list)
	summarized_narrations = agent.group_and_summarize_narration_history(
		"make a blt", narration_list
	)
	print(summarized_narrations)


if __name__ == "__main__":
	main()
