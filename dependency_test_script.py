from inference import main_call
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cpu", help='device select cpu or cuda')
args = parser.parse_args()

if args.device == "cuda":
    main_call(device=args.device)
else:
    main_call(device="cpu")