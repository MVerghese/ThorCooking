from lavis.processors import transforms_video
from lavis.processors.blip_processors import ToUint8, ToTHWC
from lavis.models.sevila_models.sevila import SeViLA
import os
import torch
from torchvision import transforms
import numpy as np
import cv2


class SeViLa_Interface:
    def __init__(
        self,
        model_path="/home/mverghese/HypothesisTesting/SeViLA/sevila_checkpoints/sevila_pretrained.pth",
        cuda_device=0,
    ):
        img_size = 224
        self.LOC_propmpt = "Does the information within the frame provide the necessary details to accurately answer the given question?"
        self.QA_prompt = (
            "Considering the information presented in the frame, answer the question."
        )
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        normalize = transforms.Normalize(mean, std)
        self.image_size = img_size
        self.transform = transforms.Compose(
            [ToUint8(), ToTHWC(), transforms_video.ToTensorVideo(), normalize]
        )

        print(
            "Model Loading \nLoading the SeViLA model can take a few minutes (typically 2-3)."
        )
        self.sevila = SeViLA(
            img_size=img_size,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            t5_model="google/flan-t5-xl",
            prompt="",
            max_txt_len=77,
            apply_lemmatizer=False,
            frame_num=4,
            answer_num=5,
            task="freeze_loc_freeze_qa_vid",
        )

        self.sevila.load_checkpoint(url_or_filename=model_path)
        print("Model Loaded")

        if torch.cuda.is_available():
            self.device = cuda_device
        else:
            self.device = "cpu"

        if self.device == "cpu":
            self.sevila = self.sevila.float()
        else:
            self.sevila = self.sevila.to(int(self.device))

    def inference(self, frames, question, num_keyframes=4):
        # frames should be a tensor of shape T,C,H,W
        # question should be a string
        # crop frames to be image_size x image_size
        crop = transforms.CenterCrop(self.image_size)
        frames = crop(frames)
        print("frames shape: ", frames.shape)

        frames = self.transform(frames)
        frames = frames.float().to(int(self.device))
        frames = frames.unsqueeze(0)

        text_input_qa = "Question: " + question + self.QA_prompt

        text_input_loc = "Question: " + question + " " + self.LOC_propmpt

        out = self.sevila.generate_demo(
            frames, text_input_qa, text_input_loc, int(num_keyframes)
        )
        # print(out)
        answer = out["output_text"][0]
        return answer


def main():
    video_path = "/home/mverghese/HypothesisTesting/SeViLA/videos/demo1.mp4"
    video_path = "/home/mverghese/MBLearning/peel_a_carrot.mp4"
    question = "What is the color of the golf cart?"
    question = "What tool is the person using?"

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(num_frames, 100)
    print(num_frames, height, width)
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        res, frame = cap.read()
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[i, :, :, :] = frame
    frames = torch.from_numpy(frames)
    frames = frames.permute(0, 3, 1, 2)
    print(frames.shape)

    # display the loaded frames
    # for i in range(num_frames):
    #     cv2.imshow("frame", frames[i,:,:,:])
    #     cv2.waitKey(100)

    sevila = SeViLa_Interface()
    answer = sevila.inference(frames, question)
    print(answer)


if __name__ == "__main__":
    main()
