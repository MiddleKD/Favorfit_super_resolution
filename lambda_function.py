from inference import *
import json
from utils import bs64_to_pil, pil_to_bs64

def respond(err, res):
    respond_msg = {'statusCode': 502 if err is not None else 200, 'body': json.dumps(res)}
    # print(f'Respond Message: {respond_msg}')
    return respond_msg

def lambda_handler(event, context):

    print("Loading arguments")
    args = event["body"]
    if isinstance(args, str):
        args = json.loads(args)

    if "image_b64" in args:
        img_bs64 = args["image_b64"]
    else:
        print("Can not found image base64 format")
        raise AssertionError
    
    print(list(args.keys()), img_bs64[:50])
    if not img_bs64.startswith("/"):
        img_bs64 = img_bs64.split(",", 1)[1]

    print("Check gpu")
    if "avail_gpu" in args:
        if args["avail_gpu"] == True:
            target_device = "cuda"
        else:
            target_device = "cpu"
    else:
        target_device = "cpu"
    
    print("Loading image")
    img_pil = bs64_to_pil(img_bs64)

    print("Loading model")
    model = call_model(ckpt="./ckpt/swin2sr_real.pth", scale=4, window_size=8, device=target_device)
    img_tensor = torch.FloatTensor(np.array(img_pil)/255).permute(2,0,1).unsqueeze(0)

    print("Processing model")
    output = inference(img_tensor=img_tensor, model=model, window_size=8, \
                       scale=4, device=target_device) * 255
    new_img_pil = Image.fromarray(output.numpy().transpose(1,2,0).astype(np.uint8))

    print("Respond")
    output_b64 = pil_to_bs64(new_img_pil)
    result = {"image_bs64": "data:application/octet-stream;base64," + output_b64}

    return respond(None, result)
