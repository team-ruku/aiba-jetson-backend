import torch
import time
import numpy as np
import cv2

from loguru import logger

from yolo import YOLOStream

from vision.run import VisionDepth

from app.utils import get_accel_device


class AIBAProcess(YOLOStream, VisionDepth):
    def __init__(self):
        super().__init__()

    def on_startup(self):
        self.setup()
        self.initialize()

    def start_process(
        self,
        args,
        optimize=False,
        side=False,
        grayscale=False,
    ):
        self.current_list = []
        with torch.no_grad():
            fps = 1
            time_start = time.time()
            frame_index = 0

            while not self.cap.end:
                ret, img = self.cap.read()

                if not ret:
                    continue

                self.yolo_preds = self.yolo_model([img], size=self.imsize)
                self.yolo_preds.files = ["img.jpg"]

                deepsort_outputs = []

                for j in range(len(self.yolo_preds.pred)):
                    temp = self.deepsort_update(
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
                    logger.info(
                        f"[YOLO] Processing {self.cap.idx // 25}th second clips"
                    )

                    clip = self.cap.get_video_clip()

                    if self.yolo_preds.pred[0].shape[0]:
                        inputs, inp_boxes, _ = self.ava_inference_transform(
                            clip, self.yolo_preds.pred[0][:, 0:4], crop_size=self.imsize
                        )
                        inp_boxes = torch.cat(
                            [torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1
                        )
                        try:  # failback for Apple Silicon
                            if isinstance(inputs, list):
                                inputs = [
                                    inp.unsqueeze(0).to(get_accel_device())
                                    for inp in inputs
                                ]
                            else:
                                inputs = inputs.unsqueeze(0).to(get_accel_device())

                            with torch.no_grad():
                                slowfaster_preds = self.video_model(
                                    inputs, inp_boxes.to(get_accel_device())
                                )
                                slowfaster_preds = slowfaster_preds.to(
                                    get_accel_device()
                                )

                        except RuntimeError:
                            if isinstance(inputs, list):
                                inputs = [inp.unsqueeze(0).to("cpu") for inp in inputs]
                            else:
                                inputs = inputs.unsqueeze(0).to("cpu")

                            with torch.no_grad():
                                slowfaster_preds = self.video_model(
                                    inputs, inp_boxes.to("cpu")
                                )
                                slowfaster_preds = slowfaster_preds.to("cpu")

                        for tid, avalabel in zip(
                            self.yolo_preds.pred[0][:, 5].tolist(),
                            np.argmax(slowfaster_preds, axis=1).tolist(),
                        ):
                            self.id_to_ava_labels[tid] = self.ava_labelnames[
                                avalabel + 1
                            ]

                self.content = None

                if args == "TDOA":
                    if img is not None:
                        original_image_rgb = np.flip(
                            img, 2
                        )  # in [0, 255] (flip required to get RGB)
                        image = self.transform({"image": original_image_rgb / 255})[
                            "image"
                        ]

                        prediction = self.process(
                            self.vision_device,
                            self.vision_model,
                            self.vision_model_type,
                            image,
                            (self.net_w, self.net_h),
                            original_image_rgb.shape[1::-1],
                            optimize,
                            True,
                        )

                        original_image_bgr = (
                            np.flip(original_image_rgb, 2) if side else None
                        )
                        self.content = self.create_side_by_side(
                            original_image_bgr, prediction, grayscale
                        )

                        alpha = 0.1
                        if time.time() - time_start > 0:
                            fps = (1 - alpha) * fps + alpha * 1 / (
                                time.time() - time_start
                            )  # exponential moving average
                            time_start = time.time()
                        logger.info(f"[Vision] FPS: {round(fps,2)}")

                        frame_index += 1

                buffer = self.save_preds_tovideo(
                    self.yolo_preds, self.id_to_ava_labels, args, self.content
                )
                ret, new_buf = cv2.imencode(".jpg", buffer)

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + bytearray(new_buf.tobytes())
                    + b"\r\n"
                )

    def return_text(self):
        while not self.cap.end:
            yield ",".join(self.current_list)

    def on_shutdown(self):
        self.end_instance()
