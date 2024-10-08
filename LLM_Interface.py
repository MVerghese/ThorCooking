import sys
# sys.path.append("/private/home/mverghese/anaconda3/envs/vqa_react_env/lib/python3.8/site-packages")
import os
import time
import threading

import torch
import transformers
from accelerate import infer_auto_device_map
import numpy as np

from typing import Dict, List
import cv2

class transformers_interface:
	def __init__(self, model_name="/checkpoint/mverghese/pretrained_models/llama2/Llama-2-7b-hf", cuda_devices = [0,1]):
		self.model_name = model_name
		self.cuda_devices = cuda_devices
		self.tokenizer = transformers.AutoTokenizer.from_pretrained(
			model_name,
			# add_bos_token=False,
			# add_eos_token=False,
			# use_fast=False,
			# padding_side='left',
		)
		# self.tokenizer.add_special_tokens(
		#     {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"},
		# )

		mem_map = {}
		for i in range(torch.cuda.device_count()):
			device_mem = torch.cuda.get_device_properties(i).total_memory
			if i in cuda_devices:
				mem_map[i] = "%iGiB" % (device_mem // 1024 ** 3)
			else:
				mem_map[i] = "0GiB"

		# IF YOU ARE RUNNING A 70B MODEL, YOU NEED TO CHANGE THIS TO A FOLDER IN YOUR PRIVATE SPACE (and make sure the folder exists)
		save_dir = "/private/home/mverghese/LLM_MM_aligner/experiments/mrinal/llama2_cache"
		# try:
		#     os.mkdir(save_dir)
		# except FileExistsError:
		#     pass
		self.model = transformers.AutoModelForCausalLM.from_pretrained(
			model_name, low_cpu_mem_usage=True, device_map="balanced", max_memory=mem_map, offload_folder=save_dir, torch_dtype=torch.bfloat16
		)
		self.model.eval()
		self.model.half()

		self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		self.model.resize_token_embeddings(self.model.config.vocab_size + 1)

	def get_num_tokens(self, prompt):
		assert isinstance(prompt, str)
		batch = self.tokenizer(prompt, return_tensors="pt")
		batch = {k: v.cuda() for k, v in batch.items()}
		return batch['input_ids'].shape[1]

	def generate(self, prompt, num_tokens = 400, top_p = .9, sampling = True, use_cache = True, stopword = None):
		# self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
		# self.model.config.pad_token_id = 32000
		is_batch = isinstance(prompt, list)
		if is_batch:
			context = [str(e).rstrip().lstrip() for e in prompt]
		else:
			context = str(prompt).rstrip().lstrip()
		# print("Prompt: {}".format(context))
		if is_batch:
			batch = self.tokenizer(context, return_tensors="pt", padding=True)
		else:
			batch = self.tokenizer(context, return_tensors="pt")
			# Print num tokens in prompt
			print("Num tokens in prompt: {}".format(batch['input_ids'].shape[1]))
		batch = {k: v.cuda() for k, v in batch.items()}
		# print("INPUT IDS:", batch['input_ids'])

		output = self.model.generate(
			**batch,
			do_sample=sampling,
			top_p=top_p,
			repetition_penalty=1.1,
			max_new_tokens=num_tokens,
			eos_token_id=self.tokenizer.eos_token_id,
		)
		# if the prompt is a string
		if is_batch:
			output_text = self.tokenizer.batch_decode(output[:,batch['input_ids'].shape[1]:], skip_special_tokens=True)
		else:
			output_text = self.tokenizer.batch_decode(output[:,batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
			if stopword is not None and stopword in output_text:
				output_text = output_text[:output_text.index(stopword)] + stopword

		

		return(output_text)
	

	def generate_choice(self,prompt,choices,sample = False):
		probs = self.eval_log_probs(prompt,choices)
		probs = probs / np.sum(probs)
		if not sample:
			return choices[np.argmax(probs)], probs
		else:
			probs = probs / np.sum(probs)
			return np.random.choice(choices,p=probs), probs
		
	def generate_stopword_and_verify(self,prompt,stopword, max_attempts = 5):
		success = False
		while not success and max_attempts > 0:
			new_text = self.generate(prompt,num_tokens=64,use_cache=False,stopword=stopword)
			if stopword in new_text:
				success = True
				break
			else:
				max_attempts -= 1
		# print("Prompt: {}".format(prompt))
		# print("Generated: {}".format(new_text))
		return new_text, success
	
	def eval_log_probs_old_old(self, prompt, queries, normalize_by_length = True):
		batch = self.tokenizer(prompt, return_tensors="pt")
		batch = {k: v.cuda() for k, v in batch.items()}
		input_length = batch['input_ids'].shape[1]
		# print("Input length: {}".format(input_length))
		force_tokens = self.tokenizer(queries)["input_ids"]

		all_transition_scores = []
		all_generated_tokens = []
		for i in range(len(queries)):

			f_tokens = [force_tokens[i]]

			output = self.model.generate(
				**batch,
				do_sample=False,
				# top_p=0.9,
				repetition_penalty=1.1,
				max_new_tokens=len(f_tokens[0]),
				eos_token_id=self.tokenizer.eos_token_id,
				return_dict_in_generate=True,
				output_scores=True,
				force_words_ids = f_tokens,
				num_beams=2,
				num_return_sequences=1,
			)
			transition_scores = self.model.compute_transition_scores(output.sequences, output.scores, output.beam_indices, normalize_logits=False)
			all_transition_scores.append(transition_scores[0,:].cpu())
			all_generated_tokens.append(output.sequences[0,batch['input_ids'].shape[1]:].cpu())

		probs = np.zeros(len(queries))
		for i in range(len(queries)):
			tokens = all_generated_tokens[i].reshape(-1,1)
			scores = all_transition_scores[i]
			decoded_tokens = np.array(self.tokenizer.batch_decode(tokens, skip_special_tokens=True))
			scores[np.where(decoded_tokens == '')] = 0.
			log_prob_sum = np.sum(scores.numpy())
			if normalize_by_length:
				# normalize by number of nonzero tokens
				log_prob_sum = log_prob_sum / np.count_nonzero(scores)
			probs[i] = np.exp(log_prob_sum)
		return probs
	
	

	def eval_log_probs_old(self, prompt, queries, normalize_by_length = True, verbose = False):
		batch = self.tokenizer(prompt, return_tensors="pt")
		batch = {k: v.cuda() for k, v in batch.items()}
		input_length = batch['input_ids'].shape[1]
		print("Num input tokens: {}".format(input_length))

		query_trie, tokened_queries = get_query_trie(self.tokenizer,queries)
		max_token_len = max([len(e) for e in tokened_queries])
		prefix_allowed_tokens_fn=filter_phrases_by_trie(input_length, query_trie, self.tokenizer, verbose = verbose)

		# output = self.model.generate(
		#     **batch,
		#     return_dict_in_generate=True,
		#     output_scores=True,
		#     num_beams=len(queries),
		#     num_return_sequences=len(queries),
		#     no_repeat_ngram_size=3,
		#     max_new_tokens=256,
		#     remove_invalid_values=True,
		#     prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
		#     # stopping_criteria=stopping_criteria,
		#     early_stopping=True,
		#     repetition_penalty=1.2,
		#     top_p=0.9,
		#     top_k=len(queries),
		#     temperature=0.0
		# )
		output = self.model.generate(
			**batch,
			return_dict_in_generate=True,
			output_scores=True,
			num_beams=len(queries),
			num_return_sequences=len(queries),
			# no_repeat_ngram_size=None,
			max_new_tokens=max_token_len,
			remove_invalid_values=True,
			prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
		)

		transition_scores = self.model.compute_transition_scores(
			output.sequences, output.scores, output.beam_indices, normalize_logits=False
		)
		# print(transition_scores)

		generated_tokens = output.sequences[:, input_length:]
		transition_scores = transition_scores.cpu()
		generated_tokens = generated_tokens.cpu()

		probs = np.zeros(len(queries))
		for i in range(len(queries)):
			tokens = generated_tokens[i,:].reshape(-1,1)
			scores = transition_scores[i]
			decoded_tokens = np.array(self.tokenizer.batch_decode(tokens, skip_special_tokens=True))
			print("Decoded tokens: ", decoded_tokens)
			scores[np.where(decoded_tokens == '')] = 0.
			log_prob_sum = np.sum(scores.numpy())
			if normalize_by_length:
				# normalize by number of nonzero tokens
				log_prob_sum = log_prob_sum / np.count_nonzero(scores)
			probs[i] = np.exp(log_prob_sum)
		return probs

	def to_tokens_and_logprobs(self,input_texts, return_tokens=False):
		batch = self.tokenizer(input_texts, padding=True, return_tensors="pt")
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
				if token not in self.tokenizer.all_special_ids:
					if return_tokens:
						text_sequence.append((self.tokenizer.decode(token), p.item()))
					else:
						text_sequence.append(p.item())
			batch.append(text_sequence)
		return batch

	def eval_log_probs(self, prompt, queries, normalize_by_length = True, batch_size = None):
		prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
		# print(self.tokenizer.decode(prompt_tokens[0]))
		num_prompt_tokens = np.sum([1 if prompt_tokens[0,i] not in self.tokenizer.all_special_ids else 0 for i in range(prompt_tokens.shape[1])])-1
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



			
				

	
	def save_cache(self):
		pass

class Base_Server:
	def __init__(self,prompt_read_file,result_write_file, image_read_folder = None):
		self.prompt_read_file = prompt_read_file
		self.result_write_file = result_write_file
		self.image_read_folder = image_read_folder
		self.interface = None
		self.image_buffer = []
	
	def prompt_listener(self):
		print("Listening for prompt changes")

		with open(self.prompt_read_file,'r') as f:
			text = f.read()
		prev_uuid = text.split("\n")[0]
		# prompt = "\n".join(text.split("\n")[1:])

		# prev_mod_time = os.stat(self.prompt_read_file).st_mtime
		while True:
			# mod_time = os.stat(self.prompt_read_file).st_mtime
			with open(self.prompt_read_file,'r') as f:
				text = f.read()
			# print("Text: {}".format(text))
			uuid = text.split("\n")[0]
			prompt = "\n".join(text.split("\n")[1:])
			if uuid != prev_uuid:
				# with open(self.prompt_read_file,'r') as f:
				#     text = f.read()
				print("Prompt: {}".format(prompt))
				if prompt == "":
					print("recieved empty prompt, continuing")
					continue
				prev_uuid = uuid
				start_time = time.time()
				result = self.generate(prompt)
				print("Time taken: {}".format(time.time() - start_time))
				out = uuid + "\n" + result
				with open(self.result_write_file,'w') as f:
					f.write(out)
				print("Result: {}".format(result))
			time.sleep(.25)
	
	def image_listener(self):
		# clear the image read folder
		for file in os.listdir(self.image_read_folder):
			if not file == "index.txt":
				os.remove(os.path.join(self.image_read_folder,file))
		self.image_buffer = []
		index_file = os.path.join(self.image_read_folder,"index.txt")
		with open(index_file,'r') as f:
			text = f.read()
		prev_uuid = text.split("\n")[0]
		print("Listening for new images")
		while True:
			with open(index_file,'r') as f:
				text = f.read()
			# print("Text: {}".format(text))
			uuid = text.split("\n")[0]
			image_indicies = "\n".join(text.split("\n")[1:])
			if uuid != prev_uuid and image_indicies != "":
				prev_uuid = uuid
				image_indicies = [int(e) for e in image_indicies.split(",")]
				for image_index in image_indicies:
					image_file = os.path.join(self.image_read_folder,str(image_index).zfill(4) + ".jpg")
					print("New image: {}".format(image_file))
					image = cv2.imread(image_file)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					if image is None:
						print("Image is none")
						continue
					else:
						print("Image shape: {}".format(image.shape))
					self.image_buffer.append(image)
			time.sleep(.25)


	
	def generate(self,prompt):
		# raise NotImplementedError
		return("This is a test result")
	
	def run(self):
		self.prompt_listener()
	
	def run_image_listener(self):
		self.image_listener()

class LLM_Server(Base_Server):
	def __init__(self,prompt_read_file,result_write_file):
		super().__init__(prompt_read_file,result_write_file)
		self.interface = transformers_interface()
	
	def generate(self,prompt):
		return self.interface.generate(prompt,num_tokens = 128)

if __name__ == "__main__":
	# prompt_file = "/private/home/mverghese/prompt.txt"
	# result_file = "/private/home/mverghese/result.txt"
	# image_read_folder = "/private/home/mverghese/server_images"
	# server = Base_Server(prompt_file,result_file,image_read_folder)

	# # import threading
	# # prompt_thread = threading.Thread(target=server.prompt_listener)
	# # prompt_thread.start()

	# server.run_image_listener()

	# # interface = transformers_interface()
	# # prompt = ["this is a test prompt", "this is another test prompt"]
	# # print(interface.generate(prompt,num_tokens = 128))
	# # prompt_file = "socratic_prompt.txt"
	# # with open(prompt_file, 'r') as prompt_file:
	# #     prompt = prompt_file.read()

	# # queries = ['put shoe', 'take shoe', 'give shoe', 'insert shoe', 'move shoe']
	# # print("Queries: ", queries)
	# # query_probs = interface.eval_log_probs_old(prompt, queries, verbose = True)
	# # print(query_probs)
	LLAMA_PATH = "/media/mverghese/Mass Storage/models/Llama-3-8b-hf/"
	llm = transformers_interface(LLAMA_PATH,cuda_devices = [0])
	queries = ["One plus one is two", "Good morning", "Hello, how are you?"]
	batch = llm.to_tokens_and_logprobs(queries)
	print(batch)