import numpy as np
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from utils import load_video_frames_cv2, get_video_metadata
from accelerate import infer_auto_device_map
import torch
import json

# model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf",device_map='auto',low_cpu_mem_usage=True, torch_dtype=torch.float16)
# model.half()
# processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

# prompt = "USER: <video>What task is the user performing in this video. ASSISTANT:"
# video_path = "Videos/make_a_blt_0.mp4"

# # sample uniformly 8 frames from the video
# total_frames = 3570
# indices = np.arange(0, total_frames, total_frames / 8).astype(int)
# clip = load_video_frames_cv2(video_path,indices)

# inputs = processor(text=prompt, videos=clip, return_tensors="pt")
# inputs = {k: v.cuda() for k, v in inputs.items()}

# # Generate
# generate_ids = model.generate(**inputs, max_length=80)
# print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

class MMLLM_Interface:
	def __init__(self):
		self.model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf",device_map='auto',low_cpu_mem_usage=True, torch_dtype=torch.float16)
		self.model.half()
		self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

	def gen_inputs(self, prompt, video_frames):
		inputs = self.processor(text=prompt, videos=video_frames, return_tensors="pt")
		inputs = {k: v.cuda() for k, v in inputs.items()}
		return inputs

	def generate(self, prompt, video_frames, max_length=80):
		inputs = self.gen_inputs(prompt, video_frames)
		generate_ids = self.model.generate(**inputs, max_length=max_length)
		return self.processor.batch_decode(generate_ids[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

	def to_tokens_and_logprobs(self,input_texts, return_tokens=False):
		batch = self.processor(text = input_texts, padding=True, return_tensors="pt")
		input_ids = batch['input_ids']
		batch = {k: v.cuda() for k, v in batch.items()}
		outputs = self.model(**batch)
		probs = torch.log_softmax(outputs.logits, dim=-1).detach().cpu()

		# collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
		probs = probs[:, :-1, :]
		input_ids = input_ids[:, 1:]
		gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

		batch = []
		for input_sentence, input_probs in zip(input_ids, gen_probs):
			text_sequence = []
			for token, p in zip(input_sentence, input_probs):
				if token not in self.processor.all_special_ids:
					if return_tokens:
						text_sequence.append((self.processor.decode(token), p.item()))
					else:
						text_sequence.append(p.item())
			batch.append(text_sequence)
		return batch

	def eval_log_probs(self, video_frames, prompt, queries, normalize_by_length = True, batch_size = None):
		prompt_tokens = self.gen_inputs(prompt, video_frames)['input_ids']
		num_prompt_tokens = np.sum([1 if prompt_tokens[0,i] not in self.processor.all_special_ids else 0 for i in range(prompt_tokens.shape[1])])-1
		print("Num prompt tokens: {}".format(num_prompt_tokens))
		sequences = [prompt + query for query in queries]
		if batch_size is not None:
			log_probs = []
			for i in range(0, len(sequences), batch_size):
				log_probs += self.to_tokens_and_logprobs(sequences[i:i+batch_size])

		else:
			log_probs = self.to_tokens_and_logprobs(sequences)
		probs = np.zeros(len(queries))
		for i in range(len(queries)):
			# print(len(log_probs[i]),num_prompt_tokens)
			# print(log_probs[i][num_prompt_tokens:])
			prob = np.sum(log_probs[i][num_prompt_tokens:])
			if normalize_by_length:
				# print(len(log_probs[i])-num_prompt_tokens)
				prob = prob / (len(log_probs[i])-num_prompt_tokens)
			# print("normalized log prob: ", prob)
			probs[i] = np.exp(prob)
		return probs

def narrate_video_segments(video_path, task_dict):
	MMLLM = MMLLM_Interface()
	out = []
	for segment in task_dict["action_segments"]:
		start_frame = segment["start_frame"]
		end_frame = segment["end_frame"]
		print("Start frame {}, End frame {}, GT Narration {}".format(start_frame, end_frame,segment["action_text"]))
		indices = np.linspace(start_frame, end_frame, 8).astype(int)
		clip = load_video_frames_cv2(video_path,indices)
		prompt = "USER: <video>Breifly describe what the user is doing in this video. Start your response with 'a person is': ASSISTANT:"
		narration = MMLLM.generate(prompt, clip)
		# remove the text ".Ъ" from the end of the narration
		if narration[-1] == '.':
			narration = narration[:-1]
		print(narration)
		narr_dict = {}
		narr_dict['start_frame'] = start_frame
		narr_dict['stop_frame'] = end_frame
		narr_dict['narrations'] = narration
		out.append(narr_dict)
	return out


if __name__ == '__main__':
	# MMLLM = MMLLM_Interface()
	# video_path = "Videos/make_a_blt_0.mp4"

	# # sample uniformly 8 frames from the video
	# total_frames = 3570
	# indices = np.arange(0, total_frames, total_frames / 8).astype(int)
	# clip = load_video_frames_cv2(video_path,indices)
	# prompt = "USER: <video>What task is the user performing in this video. ASSISTANT:"
	# print(MMLLM.generate(prompt, clip))
	video_path = "Videos/make_a_blt_0.mp4"
	task_path = "Tasks/Make_A_BLT_0.json"
	with open(task_path, 'r') as f:
		task_dict = json.load(f)
	out = narrate_video_segments(video_path, task_dict)
	with open("Narrations/blt_0_narrations_video-llava_gt.json", 'w') as file:
		json.dump(out,file)



