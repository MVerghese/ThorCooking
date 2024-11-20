import numpy as np
from transformers import (
    VideoLlavaProcessor,
    VideoLlavaForConditionalGeneration,
    MllamaForConditionalGeneration,
    AutoProcessor,
)
from utils import load_video_frames_cv2, get_video_metadata
from accelerate import infer_auto_device_map
import torch
import json
import pickle
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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
# # generate_ids = model.generate(**inputs, max_length=80)
# out_dict = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True,)
# print(out_dict.keys())
# print(out_dict['scores'].shape)

# # print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


def calc_entropy(probs):
    entropy = -1 * np.sum(probs * np.log(probs))
    return entropy


def reverse_norm(val, min_val, max_val):
    return (max_val - val) / (max_val - min_val)

class VLM_Interface:
    def __init__(self, model_path="/home/mverghese/Models/Llama-3.2-11B-Vision/"):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def gen_inputs(self, prompt, image):
        inputs = self.processor(image, prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        return inputs

    def generate(self, prompt, image, max_new_tokens=80):
        inputs = self.gen_inputs(prompt, image)
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


class MMLLM_Interface:
    def __init__(self):
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf",
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.model.half()
        self.processor = VideoLlavaProcessor.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf"
        )
        self.yes_token_id = self.processor.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.processor.tokenizer.convert_tokens_to_ids("no")
        self.A_token_id = self.processor.tokenizer.convert_tokens_to_ids("A")
        self.B_token_id = self.processor.tokenizer.convert_tokens_to_ids("B")
        self.C_token_id = self.processor.tokenizer.convert_tokens_to_ids("C")
        self.D_token_id = self.processor.tokenizer.convert_tokens_to_ids("D")
        self.E_token_id = self.processor.tokenizer.convert_tokens_to_ids("E")
        self.mcq_tokens = [
            self.A_token_id,
            self.B_token_id,
            self.C_token_id,
            self.D_token_id,
            self.E_token_id,
        ]

    def gen_inputs(self, prompt, video_frames):
        inputs = self.processor(text=prompt, videos=video_frames, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        return inputs

    def generate(self, prompt, video_frames, max_new_tokens=80):
        inputs = self.gen_inputs(prompt, video_frames)
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def to_tokens_and_logprobs(self, batch, return_tokens=False):
        input_ids = batch["input_ids"]
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = self.model(**batch)
        probs = torch.log_softmax(outputs.logits, dim=-1).detach()

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        batch = []
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            text_sequence = []
            for token, p in zip(input_sentence, input_probs):
                if return_tokens:
                    text_sequence.append((self.processor.decode(token), p.item()))
                else:
                    text_sequence.append(p.item())
            batch.append(text_sequence)
        return batch

    def eval_log_probs(self, video_frames, prompt, queries, normalize_by_length=True):
        prompt_tokens = self.gen_inputs(prompt, video_frames)["input_ids"]
        num_prompt_tokens = len(prompt_tokens[0])
        print("Num prompt tokens: {}".format(num_prompt_tokens))
        sequences = [self.gen_inputs(prompt + query, video_frames) for query in queries]
        # print(sequences)
        log_probs = []
        for sequence in sequences:
            log_prob = self.to_tokens_and_logprobs(sequence)
            # print(log_prob)
            log_probs.append(log_prob[0])
        probs = np.zeros(len(queries))
        for i in range(len(queries)):
            # print(len(log_probs[i]),num_prompt_tokens)
            # print(log_probs[i][num_prompt_tokens:])
            prob = np.sum(log_probs[i][num_prompt_tokens:])
            if normalize_by_length:
                # print(len(log_probs[i])-num_prompt_tokens)
                prob = prob / (len(log_probs[i]) - num_prompt_tokens)
            # print("normalized log prob: ", prob)
            probs[i] = np.exp(prob)
        return probs

    def yes_no_question(self, video_frames, prompt):
        inputs = self.gen_inputs(prompt, video_frames)
        outputs = self.model.generate(
            **inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True
        )
        logits = outputs["scores"][0]
        # print(logits)
        log_probs = torch.log_softmax(logits, dim=-1).detach()
        yes_prob = np.exp(log_probs[0][self.yes_token_id].item())
        no_prob = np.exp(log_probs[0][self.no_token_id].item())
        total_prob = yes_prob + no_prob
        yes_prob = yes_prob / total_prob
        no_prob = no_prob / total_prob
        return yes_prob, no_prob

    def mcq_question(self, video_frames, prompt, num_options=5):
        inputs = self.gen_inputs(prompt, video_frames)
        outputs = self.model.generate(
            **inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True
        )
        logits = outputs["scores"][0]
        # print(logits)
        log_probs = torch.log_softmax(logits, dim=-1).detach()
        probs = [
            np.exp(log_probs[0][token_id].item())
            for token_id in self.mcq_tokens[:num_options]
        ]
        probs = np.array(probs)
        total_prob = np.sum(probs)
        probs = probs / total_prob
        return probs

    def action_probs_yn(self, video_frames, actions, add_no_option=False):
        prompt = "USER: <video>Is this a video of [action] ASSISTANT:"
        probabilities = [
            self.yes_no_question(video_frames, prompt.replace("[action]", action))[0]
            for action in actions
        ]
        probabilities = np.array(probabilities)
        probabilities = probabilities / np.sum(probabilities)
        return probabilities

    def action_probs_mcq(self, video_frames, actions, add_no_option=False):
        mcq_letters = ["A", "B", "C", "D", "E"]
        assert len(actions) <= len(mcq_letters)
        prompt = "USER: <video>Is this a video of"
        for i in range(len(actions)):
            prompt += " " + mcq_letters[i] + ". " + actions[i]
        if add_no_option:
            prompt += " " + mcq_letters[len(actions)] + ". None of the above"
        prompt += " ASSISTANT:"
        num_options = len(actions) + add_no_option
        probabilities = self.mcq_question(video_frames, prompt, num_options)
        return probabilities

    def action_probs_LLM(self, video_frames, actions, add_no_option=False):
        with open("LLM_Action_Prompt.txt", "r") as file:
            prompt = file.read()
        prompt = prompt.replace("[actions]", "{}".format(actions))
        print(prompt)
        out = self.generate(prompt, video_frames)
        try:
            results = json.loads(out)
        except Exception as e:
            print(e)
            print(out)
            return None
        return results

    # def action_probs_mcq_robust(self, video_frames, actions, add_no_option = False):

    def bisection_search(
        self, video_path, target_action, search_range, max_iterations=5, top_k=5
    ):
        # total_frames = int(get_video_metadata(video_path)['nb_frames'])
        # frame_range = (0, total_frames)
        result_dict = self.bisection_helper(
            video_path, target_action, search_range, max_iterations
        )
        best_range = None
        best_prob = 0
        probs = result_dict.keys()
        probs = np.array(list(probs))
        best_probs = np.sort(probs)[-top_k:]
        return best_probs, [result_dict[prob] for prob in best_probs]

    def bisection_helper(self, video_path, target_action, frame_range, max_iterations):
        prompt = "USER: <video>Is this a video of [action] ASSISTANT:"
        prompt = prompt.replace("[action]", target_action)
        # print(frame_range)
        indices = np.linspace(frame_range[0], frame_range[1], 8).astype(int)
        clip = load_video_frames_cv2(video_path, indices)
        yes_prob, _ = self.yes_no_question(clip, prompt)
        result_dict = {}
        result_dict[yes_prob] = frame_range
        print("Range: {}, Probability: {}".format(frame_range, yes_prob))
        if max_iterations == 0:
            return result_dict
        first_half = (frame_range[0], int((frame_range[0] + frame_range[1]) / 2))
        second_half = (int((frame_range[0] + frame_range[1]) / 2), frame_range[1])
        result_dict = (
            result_dict
            | self.bisection_helper(
                video_path, target_action, first_half, max_iterations - 1
            )
            | self.bisection_helper(
                video_path, target_action, second_half, max_iterations - 1
            )
        )
        return result_dict

    def multi_bisection_search(
        self, video_path, target_actions, search_range, max_iterations=5, top_k=5
    ):
        # total_frames = int(get_video_metadata(video_path)['nb_frames'])
        # frame_range = (0, total_frames)
        result_dict = self.multi_bisection_helper(
            video_path, target_actions, search_range, max_iterations
        )
        return result_dict

    def multi_bisection_helper(
        self, video_path, target_actions, frame_range, max_iterations
    ):
        # prompt = "USER: <video>What task is the user performing in this video. ASSISTANT:"
        # print(frame_range)
        indices = np.linspace(frame_range[0], frame_range[1], 8).astype(int)
        clip = load_video_frames_cv2(video_path, indices)
        probabilities = self.action_probs_yn(clip, target_actions, add_no_option=False)

        result_dict = {}
        result_dict[frame_range] = probabilities
        # print("Range: {}, Probabilities: {}".format(frame_range, probabilities))
        if max_iterations == 0:
            return result_dict
        first_half = (frame_range[0], int((frame_range[0] + frame_range[1]) / 2))
        second_half = (int((frame_range[0] + frame_range[1]) / 2), frame_range[1])
        result_dict = (
            result_dict
            | self.multi_bisection_helper(
                video_path, target_actions, first_half, max_iterations - 1
            )
            | self.multi_bisection_helper(
                video_path, target_actions, second_half, max_iterations - 1
            )
        )
        return result_dict


def probability_visualizer(
    result_dict, prob_index, row_scale, cells_per_row, mode="entropy", gt_range=None
):
    rows = []
    if mode == "entropy":
        min_entropy = np.inf
        max_entropy = 0
        for key in result_dict.keys():
            entropy = calc_entropy(result_dict[key])
            min_entropy = min(min_entropy, entropy)
            max_entropy = max(max_entropy, entropy)
    for key in result_dict.keys():
        start_frame = key[0]
        if mode == "probability":
            key_dict = {"range": key, "value": result_dict[key][prob_index]}
        if mode == "entropy":
            probabilities = result_dict[key]
            value = (
                reverse_norm(calc_entropy(probabilities), min_entropy, max_entropy)
                if np.argmax(probabilities) == prob_index
                else 0
            )
            key_dict = {"range": key, "value": value}
        if start_frame == 0:
            rows.append([key_dict])
        else:
            found = False
            for row in rows:
                print(row[-1]["range"][1], start_frame)
                if row[-1]["range"][1] == start_frame:
                    row.append(key_dict)
                    found = True
                    break
            if not found:
                print("No row found")
    vis_array = np.zeros((len(rows), rows[0][0]["range"][1]))
    for i, row in enumerate(rows):
        for key_dict in row:
            print(i, key_dict["range"], key_dict["value"])
            vis_array[i, key_dict["range"][0] : key_dict["range"][1]] = key_dict[
                "value"
            ]
    if gt_range is not None:
        gt_array = np.zeros((1, vis_array.shape[1]))
        gt_array[0, gt_range[0] : gt_range[1]] = 1.2
        vis_array = np.concatenate((vis_array, gt_array), axis=0)
    vis_array = cv2.resize(
        vis_array, (cells_per_row, vis_array.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    vis_array = np.repeat(vis_array, row_scale, axis=0)

    return vis_array


def narrate_video_segments(video_path, task_dict):
    MMLLM = MMLLM_Interface()
    out = []
    for segment in task_dict["action_segments"]:
        start_frame = segment["start_frame"]
        end_frame = segment["end_frame"]
        print(
            "Start frame {}, End frame {}, GT Narration {}".format(
                start_frame, end_frame, segment["action_text"]
            )
        )
        indices = np.linspace(start_frame, end_frame, 8).astype(int)
        clip = load_video_frames_cv2(video_path, indices)
        prompt = "USER: <video>Breifly describe what the user is doing in this video. Start your response with 'a person is' ASSISTANT:"
        narration = MMLLM.generate(prompt, clip)
        # remove the text ".Ъ" from the end of the narration
        if narration[-1] == ".":
            narration = narration[:-1]
        print(narration)
        narr_dict = {}
        narr_dict["start_frame"] = start_frame
        narr_dict["stop_frame"] = end_frame
        narr_dict["narrations"] = narration
        out.append(narr_dict)
    return out

<<<<<<< Updated upstream
=======
def generate_prediction_sets(MMLLM, qhat, possible_actions, video_path, frame_range):
	result_dict = MMLLM.multi_bisection_search(video_path, possible_actions, frame_range, max_iterations=5)
	prediction_sets = {}
	for key in result_dict.keys():
		probabilities = result_dict[key]
		prediction_set = np.where(probabilities >= 1 - qhat)[0]
		prediction_sets[key] = prediction_set
	return prediction_sets



	








if __name__ == '__main__':

	VLM = VLM_Interface()
	video_path = "Videos/make_a_blt_0.mp4"
	frame = 2160
	clip = load_video_frames_cv2(video_path, [frame])[0]
	print(clip.shape)
	prompt = "<|image|><|begin_of_text|>Is the tomato in this image sliced?"
	output = VLM.generate(prompt, clip)
	print(output)
	1/0



	# MMLLM = MMLLM_Interface()
	# video_path = "Videos/make_a_blt_0.mp4"
	# frame_range = (1601, 2160)
	# frame_nums = np.linspace(frame_range[0], frame_range[1], 8).astype(int)
	# clip = load_video_frames_cv2(video_path, frame_nums)
	# output = MMLLM.generate("USER: <video>Is the tomato in this video in the sandwhich or outside the sandwhich? ASSISTANT:", clip, max_new_tokens=80)
	# print(output)
	# 1/0

	# task_dict_path = "Tasks/Make_A_BLT_0_abridged.json"
	# with open(task_dict_path, 'r') as f:
	# 	task_dict = json.load(f)
	# # target_action = "slicing tomato"
	# # # probs, ranges = MMLLM.bisection_search(video_path, target_action, (0,1765))
	# # # for prob, frame_range in zip(probs, ranges):
	# # # 	print("Yes probability: {}, Range: {}".format(prob, frame_range))
	# out = narrate_video_segments(video_path, task_dict)
	# with open("Narrations/blt_0_narrations_video-llava_gt_abridged.json", 'w') as file:
	# 	json.dump(out,file)


	# queries = ["cooking bacon", "slicing a tomato", "chopping lettuce", "spreading mayonnaise"]
	# # queries = ["cooking bacon", "slicing a tomato", "chopping lettuce"]
	# indices = np.linspace(812, 1355, 8).astype(int)
	# 1/0

	# result_dict = MMLLM.multi_bisection_search(video_path, queries, (0,3623))
	# with open("Narrations/blt_0_narrations_video-llava_segmentation_mcq.pkl", 'wb') as file:
		# pickle.dump(result_dict,file)

	# with open("Narrations/blt_0_narrations_video-llava_segmentation.pkl", 'rb') as file:
	# 	result_dict = pickle.load(file)

	# for key in result_dict.keys():
	# 	print("Range: {}, Probabilities: {}, Entropy: {}".format(key, result_dict[key], calc_entropy(result_dict[key])))

	# test_conformal_prediction(result_dict)
	task_dict_path = "Tasks/Make_A_BLT_0_abridged.json"
	video_path = "Videos/make_a_blt_0.mp4"
	MMLLM = MMLLM_Interface()
	with open(task_dict_path, 'r') as f:
		task_dict = json.load(f)
	qhat, possible_actions = conformal_prediction_with_task_dict(MMLLM, task_dict, video_path, alpha = .35,verbose = True, filter_options = True)
	print(qhat, possible_actions)
	# qhat = 0.9534917229117484
	# possible_actions = ['put the tomato slice on the sliced bread in the plate', 'slice the lettuce', 'pick up the lettuce slice', 'put the lettuce on the counter top', 'spread mayonnaise on the sliced bread in the plate', 'put the sliced bread in the plate', 'pick up the sliced bread', 'pick up the tomato slice', 'pick up the bacon', 'put the lettuce slice on the sliced bread in the plate', 'put the bacon on the sliced bread in the plate', 'put the sliced bread on the lettuce slice in the plate', 'pick up the tomato', 'slice the tomato', 'put the tomato on the counter top', 'cook bacon', 'pick up the lettuce']
	frame_range = (0, 3570)
	prediction_sets = generate_prediction_sets(MMLLM, qhat, possible_actions, video_path, frame_range)
	for key in prediction_sets.keys():
		print("Range: {}, Prediction Set: {}".format(key, prediction_sets[key]))
	with open("Narrations/blt_0_narrations_prediction_sets.json", 'w') as file:
		json.dump(prediction_sets,file)
	1/0


	vis_array = probability_visualizer(result_dict, 0, 10, 100, mode="entropy", gt_range=(0,637))
	plt.imshow(vis_array)
	plt.show()
	vis_array = probability_visualizer(result_dict, 1, 10, 100, mode="entropy", gt_range=(1766,2160))
	plt.imshow(vis_array)
	plt.show()
	vis_array = probability_visualizer(result_dict, 2, 10, 100, mode="entropy", gt_range=(2578,3309))
	plt.imshow(vis_array)
	plt.show()
	vis_array = probability_visualizer(result_dict, 3, 10, 100, mode="entropy", gt_range=(812,1355))
	plt.imshow(vis_array)
	plt.show()



	# plt.imshow(vis_array)
	# plt.show()
	1/0

	# sample uniformly 8 frames from the video
	# total_frames = 637
	# indices = np.arange(0, total_frames, total_frames / 8).astype(int)
	# clip = load_video_frames_cv2(video_path,indices)
	# prompt = "USER: <video>Is this a video of [action] ASSISTANT:"
	queries = ["cooking bacon", "slicing a tomato", "chopping lettuce", "spreading mayonnaise"]
	# probabilities = [MMLLM.yes_no_question(clip, prompt.replace('[action]',query))[0] for query in queries]
	# probabilities = np.array(probabilities)
	# probabilities = probabilities / np.sum(probabilities)

	# print("probabilities with yes/no")
	# for query, prob in zip(queries, probabilities):
	# 	print("Query: {}, Probability: {}".format(query, prob))

	# prompt = "USER: <video>What task is the user performing in this video. ASSISTANT:"
	# queries = ["cooking bacon", "slicing a tomato", "chopping lettuce", "toasting bread", "making a sandwich"]
	# probabilities = MMLLM.eval_log_probs(clip, prompt, queries)
	# probabilities = np.array(probabilities)
	# probabilities = probabilities / np.sum(probabilities)

	# print("probabilities without options")
	# for query, prob in zip(queries, probabilities):
	# 	print("Query: {}, Probability: {}".format(query, prob))

	# prompt = "USER: The user is making a bacon lettuce and tomato sandwhich. In this video, <video>, which of the following tasks are they performing: cooking bacon, slicing a tomato, chopping lettuce, toasting bread, making a sandwich?  ASSISTANT:"
	# queries = ["cooking bacon", "slicing a tomato", "chopping lettuce", "toasting bread", "making a sandwich"]
	# probabilities = MMLLM.eval_log_probs(clip, prompt, queries)
	# probabilities = np.array(probabilities)
	# probabilities = probabilities / np.sum(probabilities)

	# print("probabilities with options")
	# for query, prob in zip(queries, probabilities):
	# 	print("Query: {}, Probability: {}".format(query, prob))

	total_frames = 437
	indices = np.arange(0, total_frames, total_frames / 8).astype(int)
	clip = load_video_frames_cv2(video_path,indices)
	prompt = "USER: <video>Is this a video of cooking bacon ASSISTANT:"
	yes_prob, no_prob = MMLLM.yes_no_question(clip, prompt)
	print("Yes probability: {}, No probability: {}".format(yes_prob, no_prob))
	probabilities = MMLLM.action_probs_yn(clip, queries)
	print(probabilities)

	total_frames = 637
	indices = np.arange(0, total_frames, total_frames / 8).astype(int)
	clip = load_video_frames_cv2(video_path,indices)
	prompt = "USER: <video>Is this a video of cooking bacon ASSISTANT:"
	yes_prob, no_prob = MMLLM.yes_no_question(clip, prompt)
	print("Yes probability: {}, No probability: {}".format(yes_prob, no_prob))
	probabilities = MMLLM.action_probs_yn(clip, queries)
	print(probabilities)

	total_frames = 837
	indices = np.arange(0, total_frames, total_frames / 8).astype(int)
	clip = load_video_frames_cv2(video_path,indices)
	prompt = "USER: <video>Is this a video of cooking bacon ASSISTANT:"
	yes_prob, no_prob = MMLLM.yes_no_question(clip, prompt)
	print("Yes probability: {}, No probability: {}".format(yes_prob, no_prob))
	probabilities = MMLLM.action_probs_yn(clip, queries)
	print(probabilities)

	total_frames = 1037
	indices = np.arange(0, total_frames, total_frames / 8).astype(int)
	clip = load_video_frames_cv2(video_path,indices)
	prompt = "USER: <video>Is this a video of cooking bacon ASSISTANT:"
	yes_prob, no_prob = MMLLM.yes_no_question(clip, prompt)
	print("Yes probability: {}, No probability: {}".format(yes_prob, no_prob))
	probabilities = MMLLM.action_probs_yn(clip, queries)
	print(probabilities)





	# video_path = "Videos/make_a_blt_0.mp4"
	# task_path = "Tasks/Make_A_BLT_0.json"
	# with open(task_path, 'r') as f:
	# 	task_dict = json.load(f)
	# out = narrate_video_segments(video_path, task_dict)
	# with open("Narrations/blt_0_narrations_video-llava_gt.json", 'w') as file:
	# 	json.dump(out,file)
>>>>>>> Stashed changes

def test_conformal_prediction(result_dict):
    alpha = 0.1
    class_ranges = [(0, 637), (1766, 2160), (2578, 3309), (812, 1355)]
    cal_ranges = []
    cal_probabilities = []
    cal_labels = []
    for key in result_dict.keys():
        if key[1] - key[0] < 150:

            # found_label = 0
            for i, class_range in enumerate(class_ranges):
                if key[0] >= class_range[0] and key[1] <= class_range[1]:
                    cal_ranges.append(key)
                    cal_probabilities.append(result_dict[key])
                    cal_labels.append(i)
                    # found_label = 1
                    break
            # if not found_label:
            # 	cal_labels.append(len(class_ranges))
    n = len(cal_labels)
    cal_probabilities = np.array(cal_probabilities)
    cal_scores = 1 - cal_probabilities[np.arange(n), cal_labels]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method="higher")
    print("qhat: ", qhat)
    for key in result_dict.keys():
        probabilities = result_dict[key]
        prediction_set = np.where(probabilities >= 1 - qhat)[0]
        print(
            "Range: {}, Probabilities: {}, Prediction Set: {}".format(
                key, probabilities, prediction_set
            )
        )

    return qhat


def conformal_prediction_with_task_dict(
    MMLLM, task_dict, video_path, alpha=0.1, verbose=False, filter_options=False
):

    possible_actions = [
        segment["action_text"] for segment in task_dict["action_segments"]
    ]
    possible_actions = list(set(possible_actions))
    if filter_options:
        text_embedd_model = SentenceTransformer(
            "multi-qa-mpnet-base-cos-v1", device="cuda"
        )
        action_embeddings = text_embedd_model.encode(possible_actions)
    cal_ranges = []
    cal_probabilities = []
    cal_labels = []
    top_k_accuracy = []
    for segment in tqdm(task_dict["action_segments"]):
        start_frame = segment["start_frame"]
        end_frame = segment["end_frame"]
        indices = np.linspace(start_frame, end_frame, 8).astype(int)
        clip = load_video_frames_cv2(video_path, indices)
        probabilities = MMLLM.action_probs_yn(clip, possible_actions)
        if filter_options:
            prompt = "USER: <video>Breifly describe what the user is doing in this video. Start your response with 'a person is' ASSISTANT:"
            narration = MMLLM.generate(prompt, clip)
            narration_embedding = text_embedd_model.encode(narration)
            scores = np.dot(action_embeddings, narration_embedding)
            top_k = 3
            top_k_predictions = np.argsort(scores)[-top_k:]
            top_k_actions = [possible_actions[i] for i in top_k_predictions]
            top_k_accuracy.append(int(segment["action_text"] in top_k_actions))
            zero_indices = np.argsort(scores)[:-top_k]
            probabilities[zero_indices] = 0
            probabilities = probabilities / np.sum(probabilities)

        if verbose:
            print(
                "Start frame {}, End frame {}, GT Narration {}".format(
                    start_frame, end_frame, segment["action_text"]
                )
            )
            for action, prob in zip(possible_actions, probabilities):
                print("Action: {}, Probability: {}".format(action, prob))
        cal_ranges.append((start_frame, end_frame))
        cal_probabilities.append(probabilities)
        cal_labels.append(possible_actions.index(segment["action_text"]))
    print("Top k accuracy: ", np.mean(top_k_accuracy))
    if 1 - np.mean(top_k_accuracy) > alpha:
        print(
            "Top k error {} is greater than alpha {}".format(
                1 - np.mean(top_k_accuracy), alpha
            )
        )
        alpha = 1 - np.mean(top_k_accuracy)

    n = len(cal_labels)
    cal_probabilities = np.array(cal_probabilities)
    cal_scores = 1 - cal_probabilities[np.arange(n), cal_labels]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method="higher")
    predicition_sets = []
    for i in range(n):
        probabilities = cal_probabilities[i]
        prediction_set = np.where(probabilities >= 1 - qhat)[0]
        if verbose:
            print(
                "Range: {}, Probabilities: {}, Prediction Set: {}".format(
                    cal_ranges[i], probabilities, prediction_set
                )
            )
        predicition_sets.append(prediction_set)
    print(
        "average prediction set size: ",
        np.mean([len(prediction_set) for prediction_set in predicition_sets]),
    )
    return qhat, possible_actions


def generate_prediction_sets(MMLLM, qhat, possible_actions, video_path, frame_range):
    result_dict = MMLLM.multi_bisection_search(
        video_path, possible_actions, frame_range, max_iterations=5
    )
    prediction_sets = {}
    for key in result_dict.keys():
        probabilities = result_dict[key]
        prediction_set = np.where(probabilities >= 1 - qhat)[0]
        prediction_sets[key] = prediction_set
    return prediction_sets


if __name__ == "__main__":
    MMLLM = MMLLM_Interface()
    video_path = "Videos/make_a_blt_0.mp4"
    frame_range = (1601, 2160)
    frame_nums = np.linspace(frame_range[0], frame_range[1], 8).astype(int)
    clip = load_video_frames_cv2(video_path, frame_nums)
    output = MMLLM.generate(
        "USER: <video>Is the tomato in this video in the sandwhich or outside the sandwhich? ASSISTANT:",
        clip,
        max_new_tokens=80,
    )
    print(output)

    # task_dict_path = "Tasks/Make_A_BLT_0_abridged.json"
    # with open(task_dict_path, 'r') as f:
    # 	task_dict = json.load(f)
    # # target_action = "slicing tomato"
    # # # probs, ranges = MMLLM.bisection_search(video_path, target_action, (0,1765))
    # # # for prob, frame_range in zip(probs, ranges):
    # # # 	print("Yes probability: {}, Range: {}".format(prob, frame_range))
    # out = narrate_video_segments(video_path, task_dict)
    # with open("Narrations/blt_0_narrations_video-llava_gt_abridged.json", 'w') as file:
    # 	json.dump(out,file)

    # queries = ["cooking bacon", "slicing a tomato", "chopping lettuce", "spreading mayonnaise"]
    # # queries = ["cooking bacon", "slicing a tomato", "chopping lettuce"]
    # indices = np.linspace(812, 1355, 8).astype(int)

    # result_dict = MMLLM.multi_bisection_search(video_path, queries, (0,3623))
    # with open("Narrations/blt_0_narrations_video-llava_segmentation_mcq.pkl", 'wb') as file:
    # pickle.dump(result_dict,file)

    # with open("Narrations/blt_0_narrations_video-llava_segmentation.pkl", 'rb') as file:
    # 	result_dict = pickle.load(file)

    # for key in result_dict.keys():
    # 	print("Range: {}, Probabilities: {}, Entropy: {}".format(key, result_dict[key], calc_entropy(result_dict[key])))

    # test_conformal_prediction(result_dict)
    task_dict_path = "Tasks/Make_A_BLT_0_abridged.json"
    video_path = "Videos/make_a_blt_0.mp4"
    MMLLM = MMLLM_Interface()
    with open(task_dict_path, "r") as f:
        task_dict = json.load(f)
    qhat, possible_actions = conformal_prediction_with_task_dict(
        MMLLM, task_dict, video_path, alpha=0.35, verbose=True, filter_options=True
    )
    print(qhat, possible_actions)
    # qhat = 0.9534917229117484
    # possible_actions = ['put the tomato slice on the sliced bread in the plate', 'slice the lettuce', 'pick up the lettuce slice', 'put the lettuce on the counter top', 'spread mayonnaise on the sliced bread in the plate', 'put the sliced bread in the plate', 'pick up the sliced bread', 'pick up the tomato slice', 'pick up the bacon', 'put the lettuce slice on the sliced bread in the plate', 'put the bacon on the sliced bread in the plate', 'put the sliced bread on the lettuce slice in the plate', 'pick up the tomato', 'slice the tomato', 'put the tomato on the counter top', 'cook bacon', 'pick up the lettuce']
    frame_range = (0, 3570)
    prediction_sets = generate_prediction_sets(
        MMLLM, qhat, possible_actions, video_path, frame_range
    )
    for key in prediction_sets.keys():
        print("Range: {}, Prediction Set: {}".format(key, prediction_sets[key]))
    with open("Narrations/blt_0_narrations_prediction_sets.json", "w") as file:
        json.dump(prediction_sets, file)

    vis_array = probability_visualizer(
        result_dict, 0, 10, 100, mode="entropy", gt_range=(0, 637)
    )
    plt.imshow(vis_array)
    plt.show()
    vis_array = probability_visualizer(
        result_dict, 1, 10, 100, mode="entropy", gt_range=(1766, 2160)
    )
    plt.imshow(vis_array)
    plt.show()
    vis_array = probability_visualizer(
        result_dict, 2, 10, 100, mode="entropy", gt_range=(2578, 3309)
    )
    plt.imshow(vis_array)
    plt.show()
    vis_array = probability_visualizer(
        result_dict, 3, 10, 100, mode="entropy", gt_range=(812, 1355)
    )
    plt.imshow(vis_array)
    plt.show()

    # plt.imshow(vis_array)
    # plt.show()

    # sample uniformly 8 frames from the video
    # total_frames = 637
    # indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    # clip = load_video_frames_cv2(video_path,indices)
    # prompt = "USER: <video>Is this a video of [action] ASSISTANT:"
    queries = [
        "cooking bacon",
        "slicing a tomato",
        "chopping lettuce",
        "spreading mayonnaise",
    ]
    # probabilities = [MMLLM.yes_no_question(clip, prompt.replace('[action]',query))[0] for query in queries]
    # probabilities = np.array(probabilities)
    # probabilities = probabilities / np.sum(probabilities)

    # print("probabilities with yes/no")
    # for query, prob in zip(queries, probabilities):
    # 	print("Query: {}, Probability: {}".format(query, prob))

    # prompt = "USER: <video>What task is the user performing in this video. ASSISTANT:"
    # queries = ["cooking bacon", "slicing a tomato", "chopping lettuce", "toasting bread", "making a sandwich"]
    # probabilities = MMLLM.eval_log_probs(clip, prompt, queries)
    # probabilities = np.array(probabilities)
    # probabilities = probabilities / np.sum(probabilities)

    # print("probabilities without options")
    # for query, prob in zip(queries, probabilities):
    # 	print("Query: {}, Probability: {}".format(query, prob))

    # prompt = "USER: The user is making a bacon lettuce and tomato sandwhich. In this video, <video>, which of the following tasks are they performing: cooking bacon, slicing a tomato, chopping lettuce, toasting bread, making a sandwich?  ASSISTANT:"
    # queries = ["cooking bacon", "slicing a tomato", "chopping lettuce", "toasting bread", "making a sandwich"]
    # probabilities = MMLLM.eval_log_probs(clip, prompt, queries)
    # probabilities = np.array(probabilities)
    # probabilities = probabilities / np.sum(probabilities)

    # print("probabilities with options")
    # for query, prob in zip(queries, probabilities):
    # 	print("Query: {}, Probability: {}".format(query, prob))

    total_frames = 437
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = load_video_frames_cv2(video_path, indices)
    prompt = "USER: <video>Is this a video of cooking bacon ASSISTANT:"
    yes_prob, no_prob = MMLLM.yes_no_question(clip, prompt)
    print("Yes probability: {}, No probability: {}".format(yes_prob, no_prob))
    probabilities = MMLLM.action_probs_yn(clip, queries)
    print(probabilities)

    total_frames = 637
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = load_video_frames_cv2(video_path, indices)
    prompt = "USER: <video>Is this a video of cooking bacon ASSISTANT:"
    yes_prob, no_prob = MMLLM.yes_no_question(clip, prompt)
    print("Yes probability: {}, No probability: {}".format(yes_prob, no_prob))
    probabilities = MMLLM.action_probs_yn(clip, queries)
    print(probabilities)

    total_frames = 837
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = load_video_frames_cv2(video_path, indices)
    prompt = "USER: <video>Is this a video of cooking bacon ASSISTANT:"
    yes_prob, no_prob = MMLLM.yes_no_question(clip, prompt)
    print("Yes probability: {}, No probability: {}".format(yes_prob, no_prob))
    probabilities = MMLLM.action_probs_yn(clip, queries)
    print(probabilities)

    total_frames = 1037
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = load_video_frames_cv2(video_path, indices)
    prompt = "USER: <video>Is this a video of cooking bacon ASSISTANT:"
    yes_prob, no_prob = MMLLM.yes_no_question(clip, prompt)
    print("Yes probability: {}, No probability: {}".format(yes_prob, no_prob))
    probabilities = MMLLM.action_probs_yn(clip, queries)
    print(probabilities)

    # video_path = "Videos/make_a_blt_0.mp4"
    # task_path = "Tasks/Make_A_BLT_0.json"
    # with open(task_path, 'r') as f:
    # 	task_dict = json.load(f)
    # out = narrate_video_segments(video_path, task_dict)
    # with open("Narrations/blt_0_narrations_video-llava_gt.json", 'w') as file:
    # 	json.dump(out,file)
