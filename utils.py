import os

import logging

import requests
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
def get_target_data(dataset, mode=None):
    image_list = []
    if dataset == "advbench":
        csv_file = os.path.join("dataset", "advbench", "harmful_behaviors.csv")
        data = pd.read_csv(csv_file)["goal"]

    elif dataset == "safebench":
        csv_file = os.path.join("dataset", "safebench", "question", "SafeBench-Tiny.csv")
        data = pd.read_csv(csv_file)["question"]

    elif dataset == "safebench_tiny":
        csv_file = os.path.join("dataset", "safebench", "question", "safebench.csv")
        images = os.path.join("dataset", "safebench", "question", "")
        data = pd.read_csv(csv_file)["question"]

    if mode == "mm":
        return data.to_list(), image_list
    else:
        return data.to_list()

def get_logger(log_file_path):
    logger = logging.getLogger("my logger")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    log_file_path = log_file_path + ".log"  # Specify the path to your log file.
    file_handler = logging.FileHandler(log_file_path)

    logger.addHandler(file_handler)

    logger.addHandler(handler)

    return logger


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
# print(get_target_data("advbench"))
# load_image("view.jpg")


def prepare_prompt(inp, model, model_name, image):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt

def exp_img_parse(img_ll):
    exp_imgs = dict()
    if img_ll is None:
        return None
    else:
        indices = set([img.split("_")[1] for img in img_ll])
        for idx in indices:
            exp_imgs[idx] = []

        for img in img_ll:
            idx = img.split("_")[1]
            exp_imgs[idx].append(img)
        for idx in indices:
            exp_imgs[idx] = sorted(exp_imgs[idx], key=lambda x: int(x.split("_")[-2]))
        return exp_imgs

# img_folder = os.path.join("results", "safebench_tiny_llava-v1.5-7b")
# img_ll = os.listdir(img_folder)
#
# parsed_imgs = exp_img_parse(img_ll)
# for idx in sorted(parsed_imgs):
#     print(parsed_imgs[idx])




