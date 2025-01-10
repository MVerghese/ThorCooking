import numpy as np
import json
from sentence_transformers import SentenceTransformer

def build_goalstep_sequences():
	goalstep_path = "/home/mverghese/ThorCooking/Dataset_Annotations/goalstep_train.json"
	with open(goalstep_path, "r") as f:
		goalstep_annotations = json.load(f)
	result_dict = []
	for goalstep in goalstep_annotations["videos"]:
		result_dict = goalstep_sequence_helper(result_dict, goalstep)
	goals = []
	for goal in result_dict:
		goals.append(goal["goal"])
	text_embedd_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1", device="cuda")
	goal_embeddings = text_embedd_model.encode(goals)
	return result_dict, goal_embeddings, goals

def goalstep_sequence_helper(result_dict, goalstep_annotations):
	# print("Goalstep annotations: ", goalstep_annotations.keys())
	assert "segments" in goalstep_annotations.keys() and len(goalstep_annotations["segments"]) > 0
	if "goal_description" in goalstep_annotations.keys():
		new_dict = {}
		new_dict["goal"] = goalstep_annotations["goal_description"]
		new_dict["top_level_goal"] = True
	elif "step_description" in goalstep_annotations.keys():
		new_dict = {}
		new_dict["goal"] = goalstep_annotations["step_description"]
		new_dict["top_level_goal"] = False
	new_dict["steps"] = []
	for segment in goalstep_annotations["segments"]:
		new_dict["steps"].append(segment["step_description"])
		if "segments" in segment.keys() and len(segment["segments"]) > 0:
			result_dict = goalstep_sequence_helper(result_dict, segment)
	print(new_dict)
	result_dict.append(new_dict)	
	return result_dict
			

	

def main():
	goalstep_sequences, goal_embeddings, goals = build_goalstep_sequences()
	save_path = "/home/mverghese/ThorCooking/Dataset_Annotations/"
	with open(save_path + "goalstep_annontations_processed.json", "w") as f:
		json.dump(goalstep_sequences, f)
	np.save(save_path + "goal_embeddings.npy", goal_embeddings)
	with open(save_path + "goals.json", "w") as f:
		json.dump(goals, f)



if __name__ == "__main__":
	main()

