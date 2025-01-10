import numpy as np
import cv2
import ffmpeg
import torch
import json


def compute_cos_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2)


def combine_videos(paths, combine_idxs, output_path):
    frames = []
    for i in combine_idxs:
        print(paths[i])
        cap = cv2.VideoCapture(paths[i])
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame = frame[256:-256, :, :]
                frame = cv2.resize(frame, (480, 480))
                frames.append(frame)
            else:
                break
        cap.release()
    frame_height, frame_width, _ = frames[0].shape
    for i in range(len(frames)):
        frames[i] = cv2.resize(frames[i], (frame_width, frame_height))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
    for frame in frames:
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()


def load_video_frames_cv2(video_path, frame_nums):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    # print(len(frame_nums),height,width)
    frames = np.zeros((len(frame_nums), height, width, 3))
    for i in range(len(frame_nums)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nums[i] - 1)
        res, frame = cap.read()
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[i, :, :, :] = frame
    frames = torch.from_numpy(frames)
    frames = frames.permute(0, 3, 1, 2)
    return frames


def get_video_metadata(video_path):
    return ffmpeg.probe(video_path)["streams"][0]


def process_vpa_narrations(narration_list):
    narration_dict = {}
    for narration in narration_list:
        if narration["video"] not in narration_dict:
            narration_dict[narration["video"]] = {}
            narration_dict[narration["video"]]["end_time"] = 0
        if narration["end_time"] > narration_dict[narration["video"]]["end_time"]:
            narration_dict[narration["video"]]["end_time"] = narration["end_time"]
            prompt_steps = narration["prompt"].split("\n")
            narration_dict[narration["video"]]["task"] = prompt_steps[0][6:]
            end_index = prompt_steps.index("")
            narration_dict[narration["video"]]["steps"] = prompt_steps[1:end_index]
    return narration_dict

def compute_entropy(probabilities):
	# remove zero probabilities
	probabilities = probabilities[probabilities>0]
	return -np.sum(probabilities*np.log(probabilities))


if __name__ == "__main__":
    # paths = ["Videos/53_video_492_0_none_display.mp4",
    # 		 "Videos/54_video_502_0_none_display.mp4",
    # 		 "Videos/55_video_496_0_none_display.mp4",
    # 		 "Videos/56_video_504_0_none_display.mp4",
    # 		 "Videos/57_video_490_0_none_display.mp4",
    # 		 "Videos/71_video_640_0_none_display.mp4",
    # 		 "Videos/72_video_646_0_none_display.mp4",
    # 		 "Videos/73_video_648_0_none_display.mp4",
    # 		 "Videos/74_video_650_0_none_display.mp4",
    # 		 "Videos/75_video_634_0_none_display.mp4",
    # 		 ]
    # combine_videos(paths,[0,1,2],"make_a_blt_0.mp4")
    # combine_videos(paths,[0,3,4],"make_a_blt_1.mp4")
    # combine_videos(paths,[5,6,7,8,9],"make_a_latte_0.mp4")
    narration_path = "cross_task_vpa_summarized.jsonl"
    with open(narration_path, "r") as file:
        narration_list = file.readlines()
    narration_list = [json.loads(narration) for narration in narration_list]
    narration_dict = process_vpa_narrations(narration_list)
    with open("cross_task_narrations.json", "w") as file:
        json.dump(narration_dict, file)
