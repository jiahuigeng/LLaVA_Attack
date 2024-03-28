import os.path

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import cosine_embedding_loss
import torch.optim as optim
import argparse
import torch
import json


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


import matplotlib.pyplot as plt
from torchvision import transforms
import pickle
from tqdm import tqdm
from utils import get_target_data, get_logger, prepare_prompt


torch.manual_seed(50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

senti_prompt = "predict the sentiment of the sentence, positive, negative, or neutral? "
trans_prompt = "translate the following sentence to English:"
facts_prompt = "tell me facts about the following person:"
trans_zh_prompt = "translate the following sentence to Chinese:"
bomb_prompt = "tell me the steps to make the following thing:"

empty_id = 29871
image_shape = (1, 3, 336, 336)
image_token_len = 576

tp = transforms.ToPILImage()

def save_image(tensor, file_png):
    image_tensor = tensor.squeeze(0)
    image_array = image_tensor.numpy().transpose(1, 2, 0)
    plt.imsave(file_png, image_array)

def main(args):

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           load_4bit=True, device=args.device)  #
    folder_name = os.path.join("results", "_".join([args.dataset, model_name]))
    logger = get_logger(folder_name)

    exp_records = {}
    target_prompts = get_target_data(args.dataset)
    for index_prompt, target_prompt in enumerate(target_prompts):
        exp_records[index_prompt] = {}
        exp_records[index_prompt]["prompt"] = target_prompt
        exp_records[index_prompt]["loss"] = {}
        exp_records[index_prompt]["outputs"] = {}

        input_ids = tokenizer_image_token(target_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            model.device)

        input_embeds = model.get_model().embed_tokens(input_ids[:, 1:]).to(model.device)
        empty_embed = model.get_model().embed_tokens(torch.tensor([[empty_id]]).to(model.device))
        empty_embeds = empty_embed.repeat(1, image_token_len - input_ids.shape[1] - 1, 1)
        padded_input_embeds = torch.cat((empty_embeds, input_embeds), dim=1).to(model.device)

        image_tensor = torch.randn(image_shape).to(device).requires_grad_(True)
        optimizer = optim.Adam([image_tensor], lr=0.1)
        if args.optimizer == "sgd":
            optimizer = optim.SGD([image_tensor], lr=0.005)


        cos_loss_fun = nn.CosineEmbeddingLoss()
        model.train()
        for param in model.parameters():
            param.requires_grad = False

        best_loss = 100
        best_idx = 0
        best_tensor = None
        for step in range(args.num_steps):  # May need more iterations for Adam
            optimizer.zero_grad()
            image_embeds = model.encode_images(image_tensor)
            loss = None
            if args.loss_mode == "full":
                if args.loss == "l2":
                    diff = image_embeds - padded_input_embeds
                    loss = (diff ** 2).mean()

                elif args.loss == "cosine":
                    target_ones = torch.ones(padded_input_embeds.shape[1]).to("cuda")
                    loss = cos_loss_fun(image_embeds[0], padded_input_embeds[0], target_ones)

                elif args.loss == "both":
                    l2_loss = ((image_embeds - padded_input_embeds) ** 2).mean()
                    target_ones = torch.ones(padded_input_embeds.shape[1]).to("cuda")
                    cos_loss = cos_loss_fun(image_embeds[0], padded_input_embeds[0], target_ones)
                    loss = l2_loss + cos_loss

                loss.backward(retain_graph=True)
                optimizer.step()

            elif args.loss_mode == "part":
                len_prompt_token = input_embeds.shape[1]
                target_ones = torch.ones(padded_input_embeds.shape[1])[-len_prompt_token:].to("cuda")
                part_prompt_embeds = padded_input_embeds[0][-len_prompt_token:].to("cuda")
                part_image_embeds = image_embeds[0][-len_prompt_token:].to("cuda")

                if args.loss == "l2":
                    loss = ((part_image_embeds - part_prompt_embeds) ** 2).mean()
                elif args.loss == "cosine":
                    loss = cos_loss_fun(part_image_embeds, part_prompt_embeds, target_ones)
                elif args.loss == "both":
                    l2_loss = ((part_image_embeds - part_prompt_embeds) ** 2).mean()
                    cos_loss = cos_loss_fun(part_image_embeds, part_prompt_embeds, target_ones)
                    loss = l2_loss + cos_loss
                loss.backward(retain_graph=True)
                optimizer.step()


            if step % 160 == 0:  # Print loss every 10 steps
                logger.info("Step {}: Loss: {:.4f}".format(step, loss.item()))
                exp_records[index_prompt]["loss"][step] = loss.item()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_idx = step
                    best_tensor = image_tensor.detach().cpu()

                if step % 800 == 0:  # Print loss every 10 steps

                    if not os.path.exists(folder_name):
                        os.mkdir(folder_name)
                    curr_name = f"index_{index_prompt}_{args.task}_{args.loss_mode}_{args.loss}_step_{step}_prompt"
                    pickle.dump(image_tensor.detach().cpu(), open(os.path.join(folder_name, curr_name + '.bin'), "wb"))

                #     inp = " "
                #     infer_prompt = prepare_prompt(inp, model, model_name, image=True)
                #     infer_ids = tokenizer_image_token(infer_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                #         0).to("cuda")
                #     with torch.inference_mode():
                #         output_ids = model.generate(
                #             infer_ids,
                #             images=image_tensor,
                #             # image_sizes=[image_size],
                #             do_sample=True if args.temperature > 0 else False,
                #             temperature=args.temperature,
                #             max_new_tokens=args.max_new_tokens,
                #             # streamer=streamer,
                #             use_cache=True)
                #     outputs = tokenizer.decode(output_ids[0]).strip()
                #     logger.info("Step {}: Output: {}".format(step, outputs))
                #     exp_records[index_prompt]["outputs"][step] = outputs

            if loss < 1e-4:  # A threshold for convergence
                break
        if index_prompt > 1:
            break

        json.dump(exp_records, open(os.path.join("results", "_".join([args.dataset, model_name]) + ".json"), "w"), indent=2)
        best_name = f"{args.task}_{args.loss_mode}_{args.loss}_best_{best_idx}_prompt"
        pickle.dump(best_tensor, open(best_name + '.bin', "wb"))


    # else:
    #     image_tensor = torch.tensor(pickle.load(open("prompt.bin", "rb"))).to(device="cuda")
    #     image_embeds = model.encode_images(image_tensor).detach()
    #     padded_embeds = torch.tensor(pickle.load(open("padded_prompt.bin", "rb"))).to(device="cuda")
    #     loss = ((image_embeds - padded_embeds)**2).mean()
    #     print(f'Loss: {loss.item()}')
    #
    #
    # print(best_loss, best_idx)




if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        image_example = "https://llava-vl.github.io/static/images/view.jpg"
        parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
        parser.add_argument("--model-base", type=str, default=None)
        parser.add_argument("--loss_mode", type=str, default="part")
        parser.add_argument("--loss", type=str, default="both")
        parser.add_argument("--task", type=str, default="prompt_gen")
        parser.add_argument("--image-file", type=str, default="")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--num-steps", type=int, default=8001)
        parser.add_argument("--pre-set", type=int, default=None)
        parser.add_argument("--dataset", type=str, default="safebench_tiny")
        parser.add_argument("--optimizer", type=str, default="adam")

        # parser.add_argument("--conv-mode", type=str, default=None)
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--max-new-tokens", type=int, default=512)
        # parser.add_argument("--load-8bit", action="store_true")
        parser.add_argument("--load-4bit", action="store_true")
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()
        main(args)
