import numpy as np
import cv2
import argparse
from ..Model import Model

MODEL_PATH = "results/ChromaGAN_pretrainedWeights/my_model_colorization.h5"
# MODEL_PATH = "results/nude/final_model.pth"
NORMALIZE = True
INPUT_PATH = "tmp_james.jpg"
OUTPUT_PATH = "tmp_style_james.jpg"
VIDEO_IN = "test/kiki.flv"
VIDEO_OUT = "result/kiki.mp4"
TMP_DIR = "tmp"
IMAGE_WIDTH = 800  # 767 // 2
IMAGE_HEIGHT = 600  # 1025 // 2


if __name__ == "__main__":
    # Parse arguments
    arg_parser = argparse.ArgumentParser(description="Neural Style Transfer")
    arg_parser.add_argument(
        "-v",
        "--video",
        default=False,
        action="store_true",
        help="toggle for style transfer on video stream",
    )
    arg_parser.add_argument(
        "-w",
        "--webcam",
        default=False,
        action="store_true",
        help="toggle for style transfer on webcam stream",
    )
    args = vars(arg_parser.parse_args())
    model = Model()
    model.load_weights(MODEL_PATH)
