from PIL import Image

import numpy as np
import torch
# import argparse

from models.network_swin2sr import Swin2SR as net


def call_model(ckpt, scale, window_size, device):
    model = net(upscale=scale, in_chans=3, img_size=64, window_size=window_size,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')

    pretrained_model = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(pretrained_model, strict=True)
    model.eval()
    model = model.to(device)

    return model

def inference(img_tensor, model, window_size, scale, device):

    with torch.no_grad():
        _, _, h_old, w_old = img_tensor.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
        img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, :w_old + w_pad]

        output = model(img_tensor.to(device))
    
    output = output[..., :h_old * scale, :w_old * scale]
    output = output.squeeze(0).to("cpu").clamp_(0,1)

    return output

from glob import glob
import os
from tqdm import tqdm
def main_call(device="cpu"):
    model = call_model(ckpt="./ckpt/swin2sr_real.pth", scale=4, window_size=8, device=device)
    
    fn_root_path = "/media/mlfavorfit/sdb/generated_templates/generated_template_1check/high_resolution/**/"
    fns = glob(fn_root_path + "*.jpg", recursive=True)

    for idx, fn in tqdm(enumerate(fns), total=len(fns)):
        if "original.jpg" in fn : continue
        img = Image.open(fn).convert("RGB")

        img_tensor = torch.FloatTensor(np.array(img)/255).permute(2,0,1).unsqueeze(0)

        result = inference(img_tensor=img_tensor, model=model, window_size=8, \
                        scale=4, device=device) * 255
        new_img = Image.fromarray(result.numpy().transpose(1,2,0).astype(np.uint8))

        new_img.save(os.path.join(fn))
        if idx == 0:
            print(fn)


if __name__ == '__main__':

    main_call("cuda")
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--device', type=str, default="cpu", help="device cpu or cuda")
    # parser.add_argument('--image_path', type=str, default="./data/milk_of_banana.png", help="path of image")
    # parser.add_argument('--ckpt', type=str, default="./ckpt/swin2sr_real.pth", help="path of pretrained model")
    # parser.add_argument('--save_path', type=str, default="./data/temp.jpg", help="save result path")
    # parser.add_argument('--window_size', type=int, default=8, help="window size for padding")
    # parser.add_argument('--scale', type=int, default=4, help="scale rate")
    # args = parser.parse_args()

    # model = call_model(ckpt=args.ckpt, scale=args.scale, window_size=args.window_size, device=args.device)
    
    # img = Image.open(args.image_path).convert("RGB")
    # img_tensor = torch.FloatTensor(np.array(img)/255).permute(2,0,1).unsqueeze(0)
    # result = inference(img_tensor=img_tensor, model=model, window_size=args.window_size, \
    #                    scale=args.scale, device=args.device) * 255

    # new_img = Image.fromarray(result.numpy().transpose(1,2,0).astype(np.uint8))
    # new_img.save(args.save_path)
