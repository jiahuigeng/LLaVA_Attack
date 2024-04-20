import torch
import torch.nn as nn
from spacy import vocab
from torch.autograd import Variable
from torch.nn.functional import cosine_embedding_loss
import torch.optim as optim
import argparse
import torch
import os
from transformers import AutoTokenizer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
from torchvision import transforms
import pickle
torch.manual_seed(50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# senti_prompt = "predict the sentiment of the sentence, positive, negative, or neutral? "
# # trans_prompt =
# fact_prompt = "tell me facts about the following person:"
# trans_zh_prompt = "translate the following sentence to Chinese:"
# make_prompt = "tell me the steps to make the following thing:"

prompts = {"trans": "translate the following sentence to English:",
               "fact": "tell me facts about the following person:",
               "make": "tell me the steps to make the following thing:",
}

mpath_15_7b = "liuhaotian/llava-v1.5-7b"
mpath_15_13b = "liuhaotian/llava-v1.5-13b"
mpath_16_7b = "liuhaotian/llava-v1.6-vicuna-7b"
mpath_16_13b = "liuhaotian/llava-v1.6-vicuna-13b"
mpath_16_mistral = "liuhaotian/llava-v1.6-mistral-7b"

# empty_id = 29871
image_shape = (1, 3, 336, 336)
image_token_len = 576

tp = transforms.ToPILImage()

# def get_empty_id(vocab):
#     return next((key for key, value in vocab.items() if value == "▁"), None)

def get_token_embeds(model_path, task):
    mname = model_path.split("/")[-1]
    prompt_embeds_path = "results/prompt_embeds/"
    if not os.path.exists(prompt_embeds_path):
        os.makedirs(prompt_embeds_path)
    input_embeds_file = os.path.join("results/prompt_embeds/", f"task_{task}_{mname}_input_embeds.bin")
    padded_input_embeds_file = os.path.join("results/prompt_embeds/", f"task_{task}_{mname}_padded_input_embeds.bin")
    if os.path.exists(input_embeds_file) and os.path.exists(padded_input_embeds_file):
        input_embeds = torch.load(open(input_embeds_file, "rb")).to("cuda")
        padded_input_embeds = torch.load(open(padded_input_embeds_file, "rb")).to("cuda")

    else:
        target_prompt = prompts[task]
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, _, _ = load_pretrained_model(model_path, None, model_name, device=args.device)
        empty_id = tokenizer.get_vocab()["▁"]
        model.eval()

        input_ids = tokenizer_image_token(target_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            model.device)

        input_embeds = model.get_model().embed_tokens(input_ids[:, 1:]).to(model.device)
        empty_embed = model.get_model().embed_tokens(torch.tensor([[empty_id]]).to(model.device))
        empty_embeds = empty_embed.repeat(1, image_token_len - input_ids.shape[1] - 1, 1)
        padded_input_embeds = torch.cat((empty_embeds, input_embeds), dim=1).to(model.device)

        input_embeds = input_embeds.detach().clone()
        padded_input_embeds = padded_input_embeds.detach().clone()
        pickle.dump(input_embeds, open(input_embeds_file, "wb"))
        pickle.dump(padded_input_embeds, open(padded_input_embeds_file, "wb"))

    print(input_embeds.shape, padded_input_embeds.shape)
    return input_embeds, padded_input_embeds


def main(args):

    task = args.task

    token_embeds_15_7b = get_token_embeds(mpath_15_7b, task)
    # token_embeds_15_13b = get_token_embeds(mpath_15_7b, task)
    # token_embeds_16_7b = get_token_embeds(mpath_16_7b, task)
    # token_embeds_16_13b = get_token_embeds(mpath_16_13b, task)
    # token_embeds_16_mistral = get_token_embeds(mpath_16_mistral, task)


    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, _, _ = load_pretrained_model(args.model_path, args.model_base, model_name,
    #                                                                        device=args.device)  # load_4bit=True,



    # input_ids = tokenizer_image_token(target_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
    #     model.device)
    #
    # input_embeds = model.get_model().embed_tokens(input_ids[:, 1:]).to(model.device)
    # empty_embed = model.get_model().embed_tokens(torch.tensor([[empty_id]]).to(model.device))
    # empty_embeds = empty_embed.repeat(1, image_token_len - input_ids.shape[1] - 1, 1)
    # padded_input_embeds = torch.cat((empty_embeds, input_embeds), dim=1).to(model.device)

    # target_ids = [14240,   278,  1494, 10541,   304,  4223, 29901]
    # padded_ids = [[29871] * (576 - len(target_ids)) + target_ids]
    # padded_input_embeds = model.get_model().embed_tokens(torch.tensor(padded_ids).to(device="cuda"))

    # target_ids = input_ids[:, 1:]
    # padded_ids = [[29871] * (576 - len(target_ids)) + target_ids]

    image_tensor = torch.randn(image_shape).to(device).requires_grad_(True)
    optimizer_name = 'Adam'
    best_loss = 100
    best_idx = 0
    best_tensor = None
    # if optimizer_name == 'LBFGS':
    #     optimizer = optim.LBFGS([image_tensor], lr=1, max_iter=20, line_search_fn='strong_wolfe')
    #     # optimizer = optim.LBFGS([image_tensor])
    #     def closure():
    #         optimizer.zero_grad()
    #         image_embeds = model.encode_images(image_tensor)
    #         diff = image_embeds - padded_input_embeds
    #         loss = (diff**2).mean()
    #         loss.backward(retain_graph=True)
    #         return loss
    #
    #     for step in range(args.num_steps):
    #         optimizer.step(closure)
    #         with torch.no_grad():
    #             image_embeds = model.encode_images(image_tensor).requires_grad_(True)
    #             diff = image_embeds - padded_input_embeds
    #             loss = (diff**2).mean()
    #             print(f'Step {step}, Loss: {loss.item()}')
    #
    #         if loss < 1e-4:
    #             break
    #
    # elif optimizer_name == 'Adam': #(1, 576, 1024)

    # if args.task in ["trans", "fact", "make"]:
    #
    #     optimizer = optim.Adam([image_tensor], lr=0.1)
    #     cos_loss_fun = nn.CosineEmbeddingLoss()
    #     # image_embeds = torch.randn(model.encode_images(image_tensor).shape).to(device).requires_grad_(True)
    #     # optimizer = optim.Adam([image_embeds], lr=0.1)
    #     model.train()
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     for step in range(args.num_steps):  # May need more iterations for Adam
    #         optimizer.zero_grad()
    #         image_embeds = model.encode_images(image_tensor)
    #         loss = None
    #         if args.mode == "full":
    #             if args.loss == "l2":
    #                 diff = image_embeds - padded_input_embeds
    #                 loss = (diff ** 2).mean()
    #
    #             elif args.loss == "cosine":
    #                 target_ones = torch.ones(padded_input_embeds.shape[1]).to("cuda")
    #                 loss = cos_loss_fun(image_embeds[0], padded_input_embeds[0], target_ones)
    #
    #             elif args.loss == "both":
    #                 l2_loss = ((image_embeds - padded_input_embeds) ** 2).mean()
    #                 target_ones = torch.ones(padded_input_embeds.shape[1]).to("cuda")
    #                 cos_loss = cos_loss_fun(image_embeds[0], padded_input_embeds[0], target_ones)
    #                 loss = l2_loss + cos_loss
    #
    #             loss.backward(retain_graph=True)
    #             optimizer.step()
    #
    #         elif args.mode == "part":
    #             len_prompt_token = input_embeds.shape[1]
    #             target_ones = torch.ones(padded_input_embeds.shape[1])[-len_prompt_token:].to("cuda")
    #             part_prompt_embeds = padded_input_embeds[0][-len_prompt_token:].to("cuda")
    #             part_image_embeds = image_embeds[0][-len_prompt_token:].to("cuda")
    #
    #             if args.loss == "l2":
    #                 loss = ((part_image_embeds - part_prompt_embeds) ** 2).mean()
    #             elif args.loss == "cosine":
    #                 loss = cos_loss_fun(part_image_embeds, part_prompt_embeds, target_ones)
    #             elif args.loss == "both":
    #                 l2_loss = ((part_image_embeds - part_prompt_embeds) ** 2).mean()
    #                 cos_loss = cos_loss_fun(part_image_embeds, part_prompt_embeds, target_ones)
    #                 loss = l2_loss + cos_loss
    #             loss.backward(retain_graph=True)
    #             optimizer.step()
    #
    #
    #         if step % 1 == 0:  # Print loss every 10 steps
    #             if loss.item() < best_loss:
    #                 best_loss = loss.item()
    #                 best_idx = step
    #                 best_tensor = image_tensor.detach().cpu()
    #
    #             if args.pre_set == step:
    #                 file_name = f"{args.task}_{args.mode}_{args.loss}_{args.pre_set}_prompt.bin"
    #                 pickle.dump(image_tensor.detach().cpu(), open(file_name, "wb"))
    #             print(f'Step {step}, Loss: {loss.item()}')
    #
    #         if loss < 1e-4:  # A threshold for convergence
    #             break
    #
    #     best_name = f"{args.task}_{args.mode}_{args.loss}_best_{best_idx}_prompt.bin"
    #     pickle.dump(best_tensor, open(best_name, "wb"))
    #
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
        parser.add_argument("--mode", type=str, default="part")
        parser.add_argument("--loss", type=str, default="both")
        parser.add_argument("--task", type=str, default="trans")
        parser.add_argument("--image-file", type=str, default="")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--num-steps", type=int, default=3000)
        parser.add_argument("--pre-set", type=int, default=None)

        # parser.add_argument("--conv-mode", type=str, default=None)
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--max-new-tokens", type=int, default=512)
        # parser.add_argument("--load-8bit", action="store_true")
        # parser.add_argument("--load-4bit", action="store_true")
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()
        main(args)
