import torch
from argparse import ArgumentParser
import os, json
from tqdm import tqdm

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import NWPUTest, Resize2Multiple
from models import get_model
from utils import get_config, sliding_window_predict

parser = ArgumentParser(description="Test a trained model on the NWPU-Crowd test set.")
# Parameters for model
parser.add_argument("--model", type=str, default="vgg19_ae", help="The model to train.")
parser.add_argument("--input_size", type=int, default=448, help="The size of the input image.")
parser.add_argument("--reduction", type=int, default=8, choices=[8, 16, 32], help="The reduction factor of the model.")
parser.add_argument("--regression", action="store_true", help="Use blockwise regression instead of classification.")
parser.add_argument("--truncation", type=int, default=None, help="The truncation of the count.")
parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"], help="The representative count values of bins.")
parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"], help="The prompt type for CLIP.")
parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "dynamic", "coarse"], help="The granularity of bins.")
parser.add_argument("--num_vpt", type=int, default=32, help="The number of visual prompt tokens.")
parser.add_argument("--vpt_drop", type=float, default=0.0, help="The dropout rate for visual prompt tokens.")
parser.add_argument("--shallow_vpt", action="store_true", help="Use shallow visual prompt tokens.")
parser.add_argument("--weight_path", type=str, required=True, help="The path to the weights of the model.")

# Parameters for evaluation
parser.add_argument("--sliding_window", action="store_true", help="Use sliding window strategy for evaluation.")
parser.add_argument("--stride", type=int, default=None, help="The stride for sliding window strategy.")
parser.add_argument("--window_size", type=int, default=None, help="The window size for in prediction.")
parser.add_argument("--resize_to_multiple", action="store_true", help="Resize the image to the nearest multiple of the input size.")
parser.add_argument("--zero_pad_to_multiple", action="store_true", help="Zero pad the image to the nearest multiple of the input size.")

parser.add_argument("--device", type=str, default="cuda", help="The device to use for evaluation.")
parser.add_argument("--num_workers", type=int, default=4, help="The number of workers for the data loader.")


def main(args: ArgumentParser):
    print("Testing a trained model on the NWPU-Crowd test set.")
    device = torch.device(args.device)
    _ = get_config(vars(args).copy(), mute=False)
    if args.regression:
        bins, anchor_points = None, None
    else:
        with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
            config = json.load(f)[str(args.truncation)]["nwpu"]
        bins = config["bins"][args.granularity]
        anchor_points = config["anchor_points"][args.granularity]["average"] if args.anchor_points == "average" else config["anchor_points"][args.granularity]["middle"]
        bins = [(float(b[0]), float(b[1])) for b in bins]
        anchor_points = [float(p) for p in anchor_points]

    args.bins = bins
    args.anchor_points = anchor_points

    model = get_model(
        backbone=args.model,
        input_size=args.input_size, 
        reduction=args.reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        deep_vpt=not args.shallow_vpt
    )
    state_dict = torch.load(args.weight_path, map_location="cpu")
    state_dict = state_dict if "best" in os.path.basename(args.weight_path) else state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    sliding_window = args.sliding_window
    if args.sliding_window:
        window_size = args.input_size
        stride = window_size // 2 if args.stride is None else args.stride
        if args.resize_to_multiple:
            transforms = Resize2Multiple(base=args.input_size)
        else:
            transforms = None
    else:
        window_size, stride = None, None
        transforms = None

    dataset = NWPUTest(transforms=transforms, return_filename=True)

    image_ids = []
    preds = []

    for idx in tqdm(range(len(dataset)), desc="Testing on NWPU"):
        image, image_path = dataset[idx]
        image = image.unsqueeze(0)  # add batch dimension
        image = image.to(device)  # add batch dimension

        with torch.set_grad_enabled(False):
            if sliding_window:
                pred_density = sliding_window_predict(model, image, window_size, stride)
            else:
                pred_density = model(image)

            pred_count = pred_density.sum(dim=(1, 2, 3)).item()

        image_ids.append(os.path.basename(image_path).split(".")[0])
        preds.append(pred_count)

    result_dir = os.path.join(current_dir, "nwpu_test_results")
    os.makedirs(result_dir, exist_ok=True)
    weights_dir, weights_name = os.path.split(args.weight_path)
    model_name = os.path.split(weights_dir)[-1]
    result_path = os.path.join(result_dir, f"{model_name}_{weights_name.split('.')[0]}.txt")

    with open(result_path, "w") as f:
        for idx, (image_id, pred) in enumerate(zip(image_ids, preds)):
            if idx != len(image_ids) - 1:
                f.write(f"{image_id} {pred}\n")
            else:
                f.write(f"{image_id} {pred}")  # no newline at the end of the file


if __name__ == "__main__":
    args = parser.parse_args()
    args.model = args.model.lower()

    if args.regression:
        args.truncation = None
        args.anchor_points = None
        args.bins = None
        args.prompt_type = None
        args.granularity = None

    if "clip_vit" not in args.model:
        args.num_vpt = None
        args.vpt_drop = None
        args.shallow_vpt = None
    
    if "clip" not in args.model:
        args.prompt_type = None

    if args.sliding_window:
        args.window_size = args.input_size if args.window_size is None else args.window_size
        args.stride = args.input_size if args.stride is None else args.stride
        assert not (args.zero_pad_to_multiple and args.resize_to_multiple), "Cannot use both zero pad and resize to multiple."

    else:
        args.window_size = None
        args.stride = None
        args.zero_pad_to_multiple = False
        args.resize_to_multiple = False
    
    main(args)

# Example usage:
# python test_nwpu.py --model vgg19_ae --truncation 4 --weight_path ./checkpoints/sha/vgg19_ae_448_4_1.0_dmcount_aug/best_mae.pth --device cuda:0