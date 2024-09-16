import numpy as np
# from LLM_Interface import transformers_interface
import LaViLa_Interface
from utils import load_video_frames_cv2, get_video_metadata
import json
import torch
LLAMA_PATH = "/home/mverghese/models/Llama-3-8b-chat-hf/"
MODEL_PATH = "/home/mverghese/models/"


class Base_Agent:
	def __init__(self,language_model = "Llama-3-8b-chat-hf/", load_llm = True, load_narrator = True):
		if load_llm:
			self.llm = transformers_interface(model_name = MODEL_PATH + language_model, cuda_devices = [0])
		if load_narrator:
			self.narrator = LaViLa_Interface.LaViLa_Interface(load_nar = True, load_dual = False)
			self.frames_per_clip = 4
			print("loaded narrator")
		else:
			self.narrator = None
			self.frames_per_clip = 4
		# system_prompt_path = "Llama_chat_prompt.txt"
		# with open(system_prompt_path, 'r') as file:
		# 	self.system_prompt = file.read()

	def narrate(self,frames,num_sequences = 1):
		if isinstance(frames[0],np.ndarray):
			torch_images = []
			for image in images:
				torch_image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()
				torch_images.append(torch_image)
				torch_images = torch.cat(torch_images,dim=0)
		else:
			torch_images = frames
		
		torch_images = torch_images.permute(0,2,3,1)
		narrations = self.narrator.eval(torch_images,num_return_sequences=num_sequences)
		for i, narration in enumerate(narrations):
			narration = narration.replace("#C", "")
			narration = narration.replace("#c", "")
			narration = narration.replace("#O", "")
			narration = narration.replace("X ","")
			narration = narration.replace("Y ", "")
			narration = narration.replace("C ", "A person ")
			narration = narration.replace("c ", "A person ")
			narration = narration.lstrip().rstrip()
			narrations[i] = narration
		return narrations

	def narrate_video_unsegmented(self,video_path,narrate_frequency,narrations_per_clip=1):
		video_metadata = get_video_metadata(video_path)
		frame_rate = int(eval(video_metadata['avg_frame_rate']))
		duration = float(video_metadata['duration'])
		num_frames = int(duration*frame_rate)
		out = []
		frames_per_segment = narrate_frequency*frame_rate
		for i in range(0,num_frames,frames_per_segment):
			frame_nums = np.linspace(i,min(i+frames_per_segment, num_frames-1),self.frames_per_clip,dtype=int)
			frames = load_video_frames_cv2(video_path,frame_nums)
			print(frames.shape)
			narrations = self.narrate(frames,narrations_per_clip)
			print(narrations)
			narr_dict = {}
			narr_dict['start_frame'] = i
			narr_dict['stop_frame'] = i+frames_per_segment
			narr_dict['narrations'] = narrations
			out.append(narr_dict)
		return out

def main():
	agent = Base_Agent(load_llm = False, load_narrator = True)
	video_path = "Videos/make_a_blt_0.mp4"
	narrate_frequency = 2
	narrations_per_clip = 10
	out = agent.narrate_video_unsegmented(video_path,narrate_frequency,narrations_per_clip)
	print(out)
	with open("blt_0_narrations.json", 'w') as file:
		json.dump(out,file)

	video_path = "Videos/make_a_blt_1.mp4"
	narrate_frequency = 2
	narrations_per_clip = 10
	out = agent.narrate_video_unsegmented(video_path,narrate_frequency,narrations_per_clip)
	print(out)
	with open("blt_1_narrations.json", 'w') as file:
		json.dump(out,file)

	video_path = "Videos/make_a_latte_0.mp4"
	narrate_frequency = 2
	narrations_per_clip = 10
	out = agent.narrate_video_unsegmented(video_path,narrate_frequency,narrations_per_clip)
	print(out)
	with open("latte_0_narrations.json", 'w') as file:
		json.dump(out,file)
if __name__ == "__main__":
	main()




			
