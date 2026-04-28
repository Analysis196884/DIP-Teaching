import argparse
import os
import cv2
import numpy as np
import torch
from UN_network import UNetGenerator


def image_to_tensor(image):
    image = np.ascontiguousarray(image)
    tensor = torch.from_numpy(image).float()
    tensor = (tensor / 127.5) - 1.0
    tensor = tensor.permute(2, 0, 1)
    return tensor


def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1.0) * 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def main():
    parser = argparse.ArgumentParser(description="Pix2Pix inference test")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to input images directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Path to output directory",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(args.model, map_location=device))
    generator.eval()
    print(f"Loaded model from {args.model}")

    os.makedirs(args.output_dir, exist_ok=True)

    valid_exts = [".jpg", ".jpeg", ".png"]
    image_files = [
        f
        for f in os.listdir(args.input_dir)
        if any(f.lower().endswith(ext) for ext in valid_exts)
    ]
    image_files.sort()

    print(f"Found {len(image_files)} images")

    with torch.no_grad():
        for img_file in image_files:
            img_path = os.path.join(args.input_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read: {img_path}")
                continue

            h, w, c = img.shape
            if w != 2 * h:
                print(f"Skipping {img_file}: width != 2*height ({w} != {2 * h})")
                continue

            img_rgb = img[:, w // 2 :]
            img_semantic_gt = img[:, : w // 2]

            img_rgb_tensor = image_to_tensor(img_rgb).unsqueeze(0).to(device)
            output_tensor = generator(img_rgb_tensor)
            output_img = tensor_to_image(output_tensor[0])

            result = np.hstack([img_semantic_gt, img_rgb, output_img])
            output_path = os.path.join(args.output_dir, img_file)
            cv2.imwrite(output_path, result)
            print(f"Saved: {output_path}")

    print(f"Done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
