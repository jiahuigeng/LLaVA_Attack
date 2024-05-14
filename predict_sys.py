import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import cosine_embedding_loss
import torch.optim as optim
import argparse
import torch
import numpy as np
import os
import json
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
from torchvision import transforms
import pickle
from utils import get_logger, prepare_prompt, get_target_data, save_json

# torch.manual_seed(50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    full_exp_record = {}
    target_prompts = get_target_data(args.task)

    args.model_name = get_model_name_from_path(args.model_path)

    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger = get_logger(os.path.join("logs", "_".join([args.task, args.model_name])))

    if not os.path.exists("exps"):
        os.makedirs("exps")

    args.exp_path = os.path.join("exps", "_".join([args.task, args.model_name, args.suffix]))
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    save_file = os.path.join(args.exp_path, "infer_textonly.json")
    if os.path.exists(save_file):
        full_exp_record = json.load(open(save_file, encoding="utf-8"))

    # args.exp_path = os.path.join("exps", "_".join([args.task, args.model_name]))
    # if not os.path.exists(args.exp_path):
    #     os.makedirs(args.exp_path)

    # image = None
    # image_tensor = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, args.model_name,
                                                                           load_4bit=True, device=args.device)


    for index_prompt, target_prompt in enumerate(target_prompts):


        exp_record = dict()
        exp_record["prompt"] = target_prompt
        exp_record["pred"] = dict()

        if str(index_prompt) in full_exp_record:
            continue

        if "suffix" in args.suffix:
            suffix = args.suffix.split("_")[1]
            target_prompt = target_prompt + suffix + " ,"

        logger.info("Prompt Index: {}, Target: {}".format(index_prompt, target_prompt))

        input_prompt = prepare_prompt(target_prompt, model, args.model_name, image=None)

        logger.info(f"full prompt: {input_prompt}")
        input_ids = tokenizer_image_token(input_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to(model.device)

        for idx in range(args.nreps):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            output = tokenizer.decode(output_ids[0]).strip()
            output = output.replace("<s>", "").replace("</s>", "").strip()
            logger.info(f"Model Output Idx {idx}: {output}")
            exp_record["pred"][idx] = output

        full_exp_record[index_prompt] = exp_record

        if args.save_records:
            save_json(full_exp_record, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    image_example = "https://llava-vl.github.io/static/images/view.jpg"
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--optimizer_name", type=str, default="Adam")
    # parser.add_argument("--mode", type=str, default="part")
    # parser.add_argument("--loss", type=str, default="both")
    parser.add_argument("--task", type=str, default="safebench_tiny")
    parser.add_argument("--nreps", type=int, default=7)
    parser.add_argument("--suffix", type=str, required=True)
    parser.add_argument("--image-file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--num-steps", type=int, default=8001)
    # parser.add_argument("--pre-set", type=int, default=None)
    # parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--save_records", type=str, default="True")

    # parser.add_argument("--conv-mode", type=str, default=None)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    # parser.add_argument("--load-8bit", action="store_true")
    # parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
