import Environment
import Base_Agent
import numpy as np
import os
import traceback

blt_action_segmentation = {}
blt_action_segmentation["cook the bacon"] = [0,637]
blt_action_segmentation["add the bread to the plate"] = [638, 811]
blt_action_segmentation["spread the mayonnaise"] = [812, 1355]
blt_action_segmentation["add the bacon to the sandwich"] = [1356, 1600]
blt_action_segmentation["slice the tomato"] = [1601, 2160]
blt_action_segmentation["add the tomato to the sandwich"] = [2161, 2327]


def single_eval(agent, task, environment, start_index, num_gt_actions, timeout=10):
	env, video_path, task_graph_path = Environment.setup_environment("FloorPlan" + str(environment), task, start_index, history="gt", return_paths=True)
	agent.set_video_history_path(video_path)
	agent.set_task_graph_path(task_graph_path)
	print("Environment set up")
	try:
		history = env.get_history()
		successful_actions = []
		num_actions = num_gt_actions - start_index + timeout
		last_failed_action = ""
		print("Num actions: ", num_actions)
		for i in range(num_actions):
			possible_actions, action_language_tags = env.generate_possible_actions(
				return_language_tags=True
			)
			history, segment_dict = env.get_observations()
			# action, probs, index, valid_actions, all_probs = agent.select_action(task,action_language_tags,history,segment_dict,probablistic_selection=False,verbose=True,last_failed_action=last_failed_action,return_probs = True, filter_actions = True)
			action, probs, index, all_probs = agent.select_action(task,action_language_tags,history,segment_dict,probablistic_selection=False,verbose=True,last_failed_action=last_failed_action,return_probs = True, filter_actions = False)
			print("Action: ", action)
			action_dict = possible_actions[index]
			print("Action dict: ", action_dict)
			action_success, _, _ = env.step(action_dict)
			if action_success:
				successful_actions.append(action)
			else:
				# pass
				last_failed_action = action
				print("LAST FAILED ACTION: ", last_failed_action)

			task_success = env.check_success()
			if task_success:
				return True, successful_actions
		task_success = env.check_success()
	except Exception as e:
		print(traceback.format_exc())
		task_success = False
		successful_actions = []
	finally:
		env.close()

	return task_success, successful_actions


def full_eval(task, environments, start_indices, evals=5, selection_strategy="sim",save_path = ""):
	agent = Base_Agent.Base_Agent(
		load_llm=True, selection_method=selection_strategy, use_chat_prompt=False,
	)
	success_array = np.zeros((len(environments), len(start_indices)))
	eval_counter = 0
	for i in range(len(environments)):
		for j in range(len(start_indices)):
			success_counter = 0
			for k in range(evals):
				print("Evaluating environment {} with staring index {}".format(environments[i], start_indices[j]))
				task_success, successful_actions = single_eval(
					agent, task, environments[i], start_indices[j], 8, timeout=10
				)
				print("Task success: ", task_success)
				print("Successful actions: ", successful_actions)
				success_counter += int(task_success)
				eval_counter += 1
			success_array[i, j] = success_counter / evals
			if save_path != "":
				np.save(save_path + "success_array.npy",success_array)
				progress = "Finished {}/{} evaluations, {}%".format(eval_counter, len(environments) * len(start_indices) * evals, 100*eval_counter/(len(environments) * len(start_indices) * evals))
				with open(save_path + "progress.txt", "w") as f:
					f.write(progress)
	success_per_env = np.mean(success_array, axis=1)
	success_per_start = np.mean(success_array, axis=0)
	success_rate = np.mean(success_array)
	return success_array, success_per_env, success_per_start, success_rate


def test_single_eval(task, env, start_idx, selection_strategy="sim"):
	agent = Base_Agent.Base_Agent(
		load_llm=True, selection_method=selection_strategy, use_chat_prompt=False,
	)
	task_success, successful_actions = single_eval(agent, task, env, start_idx, 8)
	print(task_success)
	print(successful_actions)

def specific_eval(task, env, start_idx, selection_strategy="prob", num_gt_actions = 8, timeout = 10):
	agent = Base_Agent.Base_Agent(load_llm=True, selection_method=selection_strategy, use_chat_prompt=False)

	env, video_path, task_graph_path = Environment.setup_environment("FloorPlan" + str(env), task, start_idx, history="gt", return_paths=True)
	agent.set_video_history_path(video_path)
	agent.set_task_graph_path(task_graph_path)
	print("Environment set up")
	try:
		history = env.get_history()
		successful_actions = []
		num_actions = num_gt_actions - start_idx + timeout
		last_failed_action = ""
		print("Num actions: ", num_actions)
		for i in range(num_actions):
			possible_actions, action_language_tags = env.generate_possible_actions(
				return_language_tags=True
			)
			# print("BASE")
			history, segment_dict = env.get_observations()
			# print("segment_dict: ", segment_dict)
			# action, probs, index, valid_actions, all_probs = agent.select_action(task,action_language_tags,history,segment_dict,probablistic_selection=False,verbose=True,last_failed_action=last_failed_action,return_probs = True, filter_actions = True)
			action, probs, index, all_probs = agent.select_action(task,action_language_tags,history,segment_dict,probablistic_selection=False,verbose=True,last_failed_action=last_failed_action,return_probs = True, filter_actions = False)
			# extra_prompt = "Question: Is the tomato sliced? Answer: No the tomato is not sliced.\n"
			# print("QUESTIONS")
			# _, probs, index, all_probs = agent.select_action(task,action_language_tags,history + successful_actions,probablistic_selection=False,verbose=True,last_failed_action=last_failed_action,additional_prompt=extra_prompt,return_probs = True, filter_actions = True)

			print("Executed Action: ", action)
			action_dict = possible_actions[index]
			print("Action dict: ", action_dict)
			action_success, _, _ = env.step(action_dict)
			if action_success:
				successful_actions.append(action)
			else:
				# pass
				last_failed_action = action

			task_success = env.check_success()
			if task_success:
				return True, successful_actions
		task_success = env.check_success()
	except Exception as e:
		print(traceback.format_exc())
		task_success = False
		successful_actions = []
	finally:
		env.close()
	return task_success, successful_actions

def test_eval():
	task = "make a bacon, lettuce, and tomato sandwich"
	last_failed_action = ""
	agent = Base_Agent.Base_Agent(load_llm=True, selection_method="iter_test", use_chat_prompt=False)
	agent.set_video_history_path("Videos/make_a_blt_0.mp4")
	env = Environment.setup_environment("FloorPlan" + str(1), task, 4, history="gt")
	possible_actions, action_language_tags = env.generate_possible_actions(return_language_tags=True)
	valid_language_tags = []
	for language_tag in action_language_tags:
		if "Tomato" in language_tag or "Lettuce" in language_tag:
			valid_language_tags.append(language_tag)
	history, segment_dict = env.get_observations()
	print("segment_dict: ", segment_dict)
	action, probs, index, valid_actions, all_probs = agent.select_action(task,valid_language_tags,history,segment_dict,probablistic_selection=False,verbose=True,last_failed_action=last_failed_action,return_probs = True, filter_actions = True)





def main():
	# test_eval()
	# 1/0
	# print(specific_eval("make a fried egg and serve it in a plate",2,5, selection_strategy = "prob"))
	# # # # # test_single_eval("make a blt",3,5, selection_strategy = "prob")
	# 1/0
	successful_floorplans = [1, 2, 3, 8, 9, 11, 12, 14, 16, 22, 23, 28, 29, 30]
	# # successful_floorplans = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
	# successful_floorplans = [1]
	# successful_floorplans = [2,3,9,13,15,25]
	# successful_floorplans = [1, 2, 3, 5, 7, 8, 9, 10, 11, 16, 18, 19, 20, 23, 25, 26]
	start_indices = [4,5,6,7]
	# start_indices = [5,7,8]
	# start_indices = [3,5,6,13]
	# start_indices = [2,11,13,16]

	# start_indices = [8,10,13,15]
	# successful_floorplans = [2]
	# start_indices = [4]
	save_path = "Results/GT_History_Llama3-2-11B_Prob_MCQ_Selection_BLT_Move_Action/"
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	success_array, success_per_env, success_per_start, success_rate = full_eval("make a bacon, lettuce, and tomato sandwich",successful_floorplans,start_indices, evals = 1, selection_strategy = "prob", save_path = save_path)
	print(success_array)
	print(success_per_env)
	print(success_per_start)
	print(success_rate)
	
	np.save(save_path + "success_array.npy",success_array)
	np.save(save_path + "success_per_env.npy",success_per_env)
	np.save(save_path + "success_per_start.npy",success_per_start)
	np.save(save_path + "success_rate.npy",success_rate)

	save_path = "Results/GT_History_Llama3-2-11B_Iter_MCQ_Selection_BLT_Move_Action/"
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	success_array, success_per_env, success_per_start, success_rate = full_eval("make a bacon, lettuce, and tomato sandwich",successful_floorplans,start_indices, evals = 5, selection_strategy = "iter", save_path = save_path)
 	print(success_array)
	print(success_per_env)
	print(success_per_start)
	print(success_rate)
	
	np.save(save_path + "success_array.npy",success_array)
	np.save(save_path + "success_per_env.npy",success_per_env)
	np.save(save_path + "success_per_start.npy",success_per_start)
	np.save(save_path + "success_rate.npy",success_rate)

if __name__ == "__main__":
	main()
