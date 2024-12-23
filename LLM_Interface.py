import sys
import os
import time
import threading

import torch
import transformers
from accelerate import infer_auto_device_map, disk_offload
import numpy as np

from typing import Dict, List, AnyStr, Union

# import cv2

"""
Interface for LLM
Contains the following:
- model_name: path to model
- cuda_devices
- tokenizer
- save_dir # IF YOU ARE RUNNING A 70B MODEL, YOU NEED TO CHANGE THIS TO A FOLDER IN YOUR PRIVATE SPACE (and make sure the folder exists)
"""


class transformers_interface:
    def __init__(
        self,
        model_name: str = "/media/atkeonlab-3/Mass Storage/models/Llama-3-8b-chat-hf",
        cuda_devices: List[int] = [0, 1],
        save_dir: str = "/media/atkeonlab-3/Mass Storage/models/cache",
    ):
        self.model_name = model_name
        self.cuda_devices = cuda_devices
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
        )

        mem_map = {}
        for i in range(torch.cuda.device_count()):
            device_mem = torch.cuda.get_device_properties(i).total_memory
            if i in cuda_devices:
                mem_map[i] = "%iGiB" % (device_mem // 1024**3)
            else:
                mem_map[i] = "0GiB"

        self.save_dir = save_dir
        # try:
        #     os.mkdir(save_dir)
        # except FileExistsError:
        #     pass
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            device_map="balanced",
            max_memory=mem_map,
            offload_folder=save_dir,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.model.half()
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")
        print("Yes token id: ", self.yes_token_id)
        print("No token id: ", self.no_token_id)

    def get_num_tokens(self, prompt):
        assert isinstance(prompt, str)
        batch = self.tokenizer(prompt, return_tensors="pt")
        batch = {k: v.cuda() for k, v in batch.items()}
        return batch["input_ids"].shape[1]

    """
    generates next set of tokens from prompt
    """

    def generate(
        self,
        prompt: Union[str, List[str]],
        num_tokens=400,
        top_p=0.9,
        sampling=True,
        stopword=None,
    ) -> Union[str, List[str]]:
        is_batch = isinstance(prompt, list)

        if is_batch:
            context = [str(e).rstrip().lstrip() for e in prompt]
            batch: Dict = self.tokenizer(context, return_tensors="pt", padding=True)
        else:
            context = str(prompt).rstrip().lstrip()
            batch: Dict = self.tokenizer(context, return_tensors="pt")

        batch = {k: v.cuda() for k, v in batch.items()}

        output = self.model.generate(
            **batch,
            do_sample=sampling,
            top_p=top_p,
            repetition_penalty=1.1,
            max_new_tokens=num_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # if the prompt is a list
        if is_batch:
            output_text = self.tokenizer.batch_decode(
                output[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
            )
        else:
            output_text = self.tokenizer.batch_decode(
                output[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
            if stopword is not None and stopword in output_text:
                output_text = output_text[: output_text.index(stopword)] + stopword
        return output_text

    def generate_chat(self, messages, num_tokens=400, sampling=True):

        batch = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ).cuda()
        print("Batch device: ", batch.get_device())

        output = self.model.generate(
            batch,
            do_sample=sampling,
            max_new_tokens=num_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        output_text = self.tokenizer.batch_decode(
            output[:, batch.shape[1] :], skip_special_tokens=True
        )
        return output_text

    def generate_choice(self, prompt, choices, sample=False):
        probs = self.eval_log_probs(prompt, choices)
        probs = probs / np.sum(probs)
        if not sample:
            return choices[np.argmax(probs)], probs
        else:
            probs = probs / np.sum(probs)
            return np.random.choice(choices, p=probs), probs

    """
    Generate next set of tokens from input prompt until stopword is generated
    """

    def generate_stopword_and_verify(
        self, prompt: str, stopword: str, max_attempts: int = 5
    ):
        # loop until we find a stopword or we have reached the maximum number of attempts
        success = False
        while not success and max_attempts > 0:
            new_text = self.generate(
                prompt, num_tokens=64, use_cache=False, stopword=stopword
            )
            if stopword in new_text:
                success = True
            else:
                max_attempts -= 1
        return new_text, success

    """
    return tokens & logprobs of tokens
    """

    def to_tokens_and_logprobs(
        self, input_texts: List[str], return_tokens: bool = False
    ):
        batch = self.tokenizer(input_texts, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
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

    """
    returns the logprobs of generating the specified queries from the given prompt
    """

    def eval_log_probs(
        self,
        prompt: str,
        queries: List[str],
        normalize_by_length: bool = True,
        batch_size: int = None,
    ):
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        num_prompt_tokens = (
            np.sum(
                [
                    (
                        1
                        if prompt_tokens[0, i] not in self.tokenizer.all_special_ids
                        else 0
                    )
                    for i in range(prompt_tokens.shape[1])
                ]
            )
            - 1
        )
        print("Num prompt tokens: {}".format(num_prompt_tokens))
        sequences = [prompt + query for query in queries]
        if batch_size is not None:
            log_probs = []
            for i in range(0, len(sequences), batch_size):
                log_probs += self.to_tokens_and_logprobs(sequences[i : i + batch_size])

        else:
            log_probs = self.to_tokens_and_logprobs(sequences)
        probs = np.zeros(len(queries))
        for i in range(len(queries)):
            prob = np.sum(log_probs[i][num_prompt_tokens:])
            if normalize_by_length:
                prob = prob / (len(log_probs[i]) - num_prompt_tokens)
            probs[i] = np.exp(prob)
        return probs

    def yes_no_question(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        logits = outputs["logits"][0]
        log_probs = logits
        yes_prob = np.exp(log_probs[0][self.yes_token_id].item())
        no_prob = np.exp(log_probs[0][self.no_token_id].item())
        total_prob = yes_prob + no_prob
        # print("Yes prob: {}, No prob: {}".format(yes_prob, no_prob))
        yes_prob = yes_prob / total_prob
        no_prob = no_prob / total_prob
        return yes_prob, no_prob

    def action_probs_yn(self, prompt, actions):
        probabilities = [
            self.yes_no_question(prompt.replace("[action]", action))[0]
            for action in actions
        ]
        probabilities = np.array(probabilities)
        probabilities = probabilities / np.sum(probabilities)
        return probabilities

    def save_cache(self):
        pass

    def generate_chat(self, messages, num_tokens=400, sampling=True):
        batch = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ).cuda()
        print("Batch device: ", batch.get_device())

        output = self.model.generate(
            batch,
            do_sample=sampling,
            max_new_tokens=num_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        output_text = self.tokenizer.batch_decode(
            output[:, batch.shape[1] :], skip_special_tokens=True
        )
        return output_text


class Base_Server:
    def __init__(self, prompt_read_file, result_write_file, image_read_folder=None):
        self.prompt_read_file = prompt_read_file
        self.result_write_file = result_write_file
        self.image_read_folder = image_read_folder
        self.interface = None
        self.image_buffer = []

    def prompt_listener(self):
        print("Listening for prompt changes")

        with open(self.prompt_read_file, "r") as f:
            text = f.read()
        prev_uuid = text.split("\n")[0]

        while True:
            with open(self.prompt_read_file, "r") as f:
                text = f.read()
            uuid = text.split("\n")[0]
            prompt = "\n".join(text.split("\n")[1:])
            if uuid != prev_uuid:
                print("Prompt: {}".format(prompt))
                if prompt == "":
                    print("recieved empty prompt, continuing")
                    continue
                prev_uuid = uuid
                start_time = time.time()
                result = self.generate(prompt)
                print("Time taken: {}".format(time.time() - start_time))
                out = uuid + "\n" + result
                with open(self.result_write_file, "w") as f:
                    f.write(out)
                print("Result: {}".format(result))
            time.sleep(0.25)

    def image_listener(self):
        # clear the image read folder
        for file in os.listdir(self.image_read_folder):
            if not file == "index.txt":
                os.remove(os.path.join(self.image_read_folder, file))
        self.image_buffer = []
        index_file = os.path.join(self.image_read_folder, "index.txt")
        with open(index_file, "r") as f:
            text = f.read()
        prev_uuid = text.split("\n")[0]
        print("Listening for new images")
        while True:
            with open(index_file, "r") as f:
                text = f.read()
            uuid = text.split("\n")[0]
            image_indicies = "\n".join(text.split("\n")[1:])
            if uuid != prev_uuid and image_indicies != "":
                prev_uuid = uuid
                image_indicies = [int(e) for e in image_indicies.split(",")]
                for image_index in image_indicies:
                    image_file = os.path.join(
                        self.image_read_folder, str(image_index).zfill(4) + ".jpg"
                    )
                    print("New image: {}".format(image_file))
                    image = cv2.imread(image_file)  # type: ignore
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
                    if image is None:
                        print("Image is none")
                        continue
                    else:
                        print("Image shape: {}".format(image.shape))
                    self.image_buffer.append(image)
            time.sleep(0.25)

    def generate(self, prompt):
        # raise NotImplementedError
        return "This is a test result"

    def run(self):
        self.prompt_listener()

    def run_image_listener(self):
        self.image_listener()


class LLM_Server(Base_Server):
    def __init__(self, prompt_read_file, result_write_file):
        super().__init__(prompt_read_file, result_write_file)
        self.interface = transformers_interface()

    def generate(self, prompt):
        return self.interface.generate(prompt, num_tokens=128)


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
    LLAMA_PATH = "/media/atkeonlab-3/Mass Storage/models/Llama-3-8b-chat-hf"
    llm = transformers_interface(LLAMA_PATH, cuda_devices=[0, 1])
    messages = [
        {
            "role": "system",
            "content": "You are an assitant designed to help people with household tasks. Do not be verbose. Answer the question with no added qualifications or caveats. Just directly provide the answer.",
        },
        {
            "role": "user",
            "content": "Help me make a Bacon, Lettuce and Tomato sandwich. List the possible steps separated by commas.",
        },
    ]
    # prompt = "List the common steps to make a Bacon, Lettuce and Tomato sandwich. Separate the steps by commas."
    response = llm.generate_chat(messages, num_tokens=512)
    # response = llm.generate(prompt,num_tokens = 512)
    print(response)
    # yes_prob, no_prob = llm.yes_no_question(prompt)
    # print("Yes prob: {}, No prob: {}".format(yes_prob, no_prob))
    # output = llm.generate(prompt,num_tokens = 128)
    # print(output)
