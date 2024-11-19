import LaViLa_Interface
import numpy as np
from utils import load_video_frames_cv2, get_video_metadata
import json


class Grounding:

	def __init__(self,load_lavila = True, lavila_cuda_device = 0):
		if load_lavila:
			self.lavila = LaViLa_Interface.LaViLa_Interface(cuda_device = lavila_cuda_device)
		self.language_tags = []
		self.language_tags_embedding = np.zeros(0)
		self.frames_per_eval = 16

	def set_language_tags(self,language_tags):
		self.language_tags = language_tags
		self.language_tags_embedding = self.lavila.eval_text(language_tags).cpu().numpy()

	def compute_similarity(self, video_path, frame_range):
		frame_nums = np.linspace(frame_range[0], frame_range[1], self.frames_per_eval).astype(int)
		clip = load_video_frames_cv2(video_path, frame_nums)
		clip_embedding = self.lavila.eval_frames(clip).cpu().numpy()
		similarities = clip_embedding @ self.language_tags_embedding.T
		print(similarities)


	def multi_bisection_search(self, video_path, target_actions, search_range, max_iterations=5, top_k = 5):
		result_dict = self.multi_bisection_helper(video_path, target_actions, search_range, max_iterations)
		return result_dict

	def multi_bisection_helper(self, video_path, target_actions, frame_range, max_iterations):
		# prompt = "USER: <video>What task is the user performing in this video. ASSISTANT:"
		# print(frame_range)
		indices = np.linspace(frame_range[0], frame_range[1], 8).astype(int)
		clip = load_video_frames_cv2(video_path,indices)
		probabilities = self.action_probs_yn(clip, target_actions, add_no_option = False)

		result_dict = {}
		result_dict[frame_range] = probabilities
		# print("Range: {}, Probabilities: {}".format(frame_range, probabilities))
		if max_iterations == 0:
			return result_dict
		first_half = (frame_range[0], int((frame_range[0] + frame_range[1]) / 2))
		second_half = (int((frame_range[0] + frame_range[1]) / 2), frame_range[1])
		result_dict = result_dict | self.multi_bisection_helper(video_path, target_actions, first_half, max_iterations-1) | self.multi_bisection_helper(video_path, target_actions, second_half, max_iterations-1)
		return result_dict

def main():
	grounding = Grounding()
	task_dict_path = "Tasks/Make_A_BLT_0_abridged.json"
	video_path = "Videos/make_a_blt_0.mp4"
	with open(task_dict_path, 'r') as file:
		task_dict = json.load(file)
	language_tags = [step["action_text"] for step in task_dict["action_segments"]]
	language_tags = ['cook bacon', 'add bread to plate', 'spread mayonnaise', 'add the bacon to the sandwich', 'slice tomato', 'add tomato to sandwich', 'slice lettuce', 'add lettuce to sandwich', 'add bread to sandwich']
	print(language_tags)
	grounding.set_language_tags(language_tags)
	frame_range = (0, 3570)
	frame_range = (0, 637)

	grounding.compute_similarity(video_path, frame_range)


if __name__ == '__main__':
	main()
