import math
import os
import random
import time
import warnings

import cv2
import numpy as np
import pytorchvideo
import torch

warnings.filterwarnings("ignore", category=UserWarning)

from PIL import Image, ImageDraw, ImageFont
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from pytorchvideo.transforms.functional import (
    clip_boxes_to_image,
    short_side_scale_with_boxes,
    uniform_temporal_subsample,
)
from torchvision.transforms._functional_video import normalize

from .deep_sort.deep_sort import DeepSort

from loguru import logger


class MyVideoCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []

    def read(self):
        self.idx += 1
        ret, img = self.cap.read()
        if ret:
            self.stack.append(img)
        else:
            self.end = True
        return ret, img

    def to_tensor(self, img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)

    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must large than 0 !"
        self.stack = [self.to_tensor(img) for img in self.stack]
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)
        del self.stack
        self.stack = []
        return clip

    def release(self):
        self.cap.release()

    def get(self, parameter):
        return self.cap.get(parameter)


class YOLOStream:
    def __init__(self):
        self.imsize = 640
        self.conf = 0.4
        self.iou = 0.4
        super().__init__()

    def __tensor_to_numpy(self, tensor):
        img = tensor.cpu().numpy().transpose((1, 2, 0))
        return img

    def ava_inference_transform(
        self,
        clip,
        boxes,
        num_frames=32,  # if using slowfast_r50_detection, change this to 32, 4 for slow
        crop_size=640,
        data_mean=[0.45, 0.45, 0.45],
        data_std=[0.225, 0.225, 0.225],
        slow_fast_alpha=4,  # if using slowfast_r50_detection, change this to 4, None for slow
    ):
        boxes = np.array(boxes)
        roi_boxes = boxes.copy()
        clip = uniform_temporal_subsample(clip, num_frames)
        clip = clip.float()
        clip = clip / 255.0
        height, width = clip.shape[2], clip.shape[3]
        boxes = clip_boxes_to_image(boxes, height, width)
        clip, boxes = short_side_scale_with_boxes(
            clip,
            size=crop_size,
            boxes=boxes,
        )
        clip = normalize(
            clip,
            np.array(data_mean, dtype=np.float32),
            np.array(data_std, dtype=np.float32),
        )
        boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])
        if slow_fast_alpha is not None:
            fast_pathway = clip
            slow_pathway = torch.index_select(
                clip,
                1,
                torch.linspace(
                    0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
                ).long(),
            )
            clip = [slow_pathway, fast_pathway]

        return clip, torch.from_numpy(boxes), roi_boxes

    def plot_one_box(
        self,
        x,
        img,
        color=[100, 100, 100],
        text_info="None",
    ):
        # Plots one bounding box on image img
        color = [253, 253, 255]  # FDFDFF - GRADE1
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

        cv2.rectangle(img, c1, c2, color, 2, lineType=cv2.LINE_AA)

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        font_path = "yolo/assets/Pretendard-Medium.ttf"
        font = ImageFont.truetype(font_path, 14)

        l, t, r, b = draw.textbbox((c1[0], c1[1]), text_info, font=font)
        draw.rectangle((l, t - 5, r + 10, b + 15), fill=(253, 253, 255))

        draw.text((c1[0] + 5, c1[1] + 5), text_info, font=font, fill=(135, 132, 154))

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def pseduo_tdoa(
        self,
        x,
        img,
        color=[100, 100, 100],
        text_info="None",
    ):
        # Plots one bounding box on image img
        color = [253, 253, 255]  # FDFDFF - GRADE1
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

        cv2.circle(
            img, (int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)), 8, color, -1
        )

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        font_path = "yolo/assets/Pretendard-Medium.ttf"
        font = ImageFont.truetype(font_path, 30)

        l, t, r, b = draw.textbbox((0, 0), text_info, font=font)

        draw.text(
            (int(((c1[0] + c2[0]) / 2) - (r / 2)), int((c1[1] + c2[1]) / 2) + 14),
            text_info,
            font=font,
            fill=(253, 253, 255),
        )

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def deepsort_update(self, Tracker, pred, xywh, np_img):
        outputs = Tracker.update(
            xywh,
            pred[:, 4:5],
            pred[:, 5].tolist(),
            cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB),
        )
        return outputs

    def save_preds_tovideo(
        self,
        yolo_preds,
        id_to_ava_labels,
        args: str,
        vision_frame,
    ):
        for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
            if args == "TDOA":
                im = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2RGB)

            else:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            if pred.shape[0]:
                for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                    if int(cls) != 0:
                        ava_label = ""

                    elif trackid in id_to_ava_labels.keys():
                        ava_label = id_to_ava_labels[trackid].split(" ")[0]

                    else:
                        ava_label = "Unknown"

                    if args == "YOLO":
                        im = self.plot_one_box(
                            box,
                            im,
                            [253, 253, 255],
                            "{} {}".format(yolo_preds.names[int(cls)], ava_label),
                        )

                    else:
                        im = self.pseduo_tdoa(
                            box,
                            im,
                            [253, 253, 255],
                            "{}".format(yolo_preds.names[int(cls)]),
                        )

            self.final_frame = im.astype(np.uint8)
            return self.final_frame

    def setup(self):
        self.imsize = self.imsize

        self.yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5l6").to(0)
        self.yolo_model.conf = self.conf
        self.yolo_model.iou = self.iou
        self.yolo_model.max_det = 100

        self.video_model = slowfast_r50_detection(True).eval().to(0)

        self.deepsort_tracker = DeepSort(
            "yolo/deep_sort/deep_sort/deep/checkpoint/ckpt.t7"
        )
        self.ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map(
            "yolo/selfutils/temp.pbtxt"
        )

        logger.info("[YOLO] Starting YOLO-Slowfast instance ...")

        self.cap = MyVideoCapture(0)
        self.id_to_ava_labels = {}
        self.a = time.time()

    def main(self, args: str):
        while not self.cap.end:
            ret, img = self.cap.read()

            if not ret:
                continue

            self.yolo_preds = self.yolo_model([img], size=self.imsize)
            self.yolo_preds.files = ["img.jpg"]

            deepsort_outputs = []

            for j in range(len(self.yolo_preds.pred)):
                temp = self.__deepsort_update(
                    self.deepsort_tracker,
                    self.yolo_preds.pred[j].cpu(),
                    self.yolo_preds.xywh[j][:, 0:4].cpu(),
                    self.yolo_preds.ims[j],
                )
                if len(temp) == 0:
                    temp = np.ones((0, 8))
                deepsort_outputs.append(temp.astype(np.float32))

            self.yolo_preds.pred = deepsort_outputs

            if len(self.cap.stack) == 25:
                logger.info(f"[YOLO] Processing {self.cap.idx // 25}th second clips")

                clip = self.cap.get_video_clip()

                if self.yolo_preds.pred[0].shape[0]:
                    inputs, inp_boxes, _ = self.__ava_inference_transform(
                        clip, self.yolo_preds.pred[0][:, 0:4], crop_size=self.imsize
                    )
                    inp_boxes = torch.cat(
                        [torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1
                    )

                    if isinstance(inputs, list):
                        inputs = [inp.unsqueeze(0).to(0) for inp in inputs]
                    else:
                        inputs = inputs.unsqueeze(0).to(0)

                    with torch.no_grad():
                        slowfaster_preds = self.video_model(inputs, inp_boxes.to(0))
                        slowfaster_preds = slowfaster_preds.cpu()

                    for tid, avalabel in zip(
                        self.yolo_preds.pred[0][:, 5].tolist(),
                        np.argmax(slowfaster_preds, axis=1).tolist(),
                    ):
                        self.id_to_ava_labels[tid] = self.ava_labelnames[avalabel + 1]

            buffer = self.__save_preds_tovideo(
                self.yolo_preds, self.id_to_ava_labels, args
            )
            ret, new_buf = cv2.imencode(".jpg", buffer)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + bytearray(new_buf.tobytes())
                + b"\r\n"
            )

    def end_instance(self):
        logger.info("[YOLO] Total cost: {:.3f} s".format(time.time() - self.a))
        self.cap.release()
