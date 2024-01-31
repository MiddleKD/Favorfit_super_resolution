from PIL import Image
import numpy as np
import torch
from models.network_swin2sr import Swin2SR as net


def call_model(ckpt, scale, window_size, device):
    model = net(upscale=scale, in_chans=3, img_size=64, window_size=window_size,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv', main_device=device)

    pretrained_model = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(pretrained_model, strict=True)
    model.eval()
    model = model.to(device)

    return model

def inference(img_pil, model, window_size=8, scale=4):
    img_tensor = torch.FloatTensor(np.array(img_pil)/255).permute(2,0,1).unsqueeze(0)

    with torch.no_grad():
        _, _, h_old, w_old = img_tensor.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
        img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, :w_old + w_pad]

        output = model(img_tensor.to(model.device))
    
    output = output[..., :h_old * scale, :w_old * scale]
    output = output.squeeze(0).to("cpu").clamp_(0,1)
    output = output * 255

    return Image.fromarray(output.numpy().transpose(1,2,0).astype(np.uint8))


def main_call(model_path, root_dir, save_dir, device="cpu"):
    import os
    from glob import glob
    from tqdm import tqdm

    model = call_model(ckpt=model_path, scale=4, window_size=8, device=device)
    
    fns = glob(os.path.join(root_dir, "*"))

    for idx, fn in tqdm(enumerate(fns), total=len(fns)):
        img_pil = Image.open(fn).convert("RGB")
        result = inference(img_pil=img_pil, model=model, window_size=8, scale=4)
        result.save(os.path.join(save_dir, os.path.basename(fn)))

if __name__ == '__main__':
    main_call(model_path="/home/mlfavorfit/Desktop/lib_link/favorfit/kjg/0_model_weights/super_resolution/super_resolution_x4.pth",
              root_dir="/media/mlfavorfit/sdb/cat_toy/images", 
              save_dir="/media/mlfavorfit/sdb/cat_toy/images2", 
              device="cuda")
