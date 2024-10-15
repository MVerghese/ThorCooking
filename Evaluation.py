import Environment
import Base_Agent
import numpy as np

def single_eval(agent, task,environment,start_index, num_gt_actions, timeout = 10):
	env = Environment.setup_environment("FloorPlan" + str(environment), task, start_index, history="Narrations/blt_0_narrations_video-llava_gt.json")
	try:
		history = env.get_history()[:start_index]
		successful_actions = []
		num_actions = num_gt_actions - start_index + timeout
		print("Num actions: ", num_actions)
		for i in range(num_actions):
			possible_actions, action_language_tags = env.generate_possible_actions(return_language_tags = True)
			action, probs, index = agent.select_action(task,action_language_tags,history + successful_actions,probablistic_selection=False, verbose=True)
			print("Action: ", action)
			action_dict = possible_actions[index]
			print("Action dict: ", action_dict)
			action_success, _, _ = env.step(action_dict)
			if action_success:
				successful_actions.append(action)

			task_success = env.check_success()
			if task_success:
				return True, successful_actions
		task_success = env.check_success()
	except Exception as e:
		print(e)
		task_success = False
		successful_actions = []
	finally:
		env.close()

	return task_success, successful_actions



def full_eval(task,environments,start_indices, evals = 5):
	agent = Base_Agent.Base_Agent(load_llm = True, load_narrator = False)
	success_array = np.zeros((len(environments),len(start_indices)))
	for i in range(len(environments)):
		for j in range(len(start_indices)):
			success_counter = 0
			for k in range(evals):
				task_success, _ = single_eval(agent,task,environments[i],start_indices[j], 18)
				success_counter += int(task_success)
			success_array[i,j] = success_counter / evals
	success_per_env = np.mean(success_array,axis=1)
	success_per_start = np.mean(success_array,axis=0)
	success_rate = np.mean(success_array)
	return success_array, success_per_env, success_per_start, success_rate


def main():
	successful_floorplans = [2, 3, 6, 9, 11, 12, 14, 15, 17, 23, 28, 29, 30]
	start_indices = [8,10,13,15]
	# agent = Base_Agent.Base_Agent(load_llm = True, load_narrator = False)
	success_array, success_per_env, success_per_start, success_rate = full_eval("make a blt",successful_floorplans,start_indices)
	print(success_array)
	print(success_per_env)
	print(success_per_start)
	print(success_rate)
	np.save("success_array.npy",success_array)
	np.save("success_per_env.npy",success_per_env)
	np.save("success_per_start.npy",success_per_start)
	np.save("success_rate.npy",success_rate)
if __name__ == '__main__':
	main()