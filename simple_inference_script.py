import os
import sys
import torch
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms

# Add LOCATE to path
LOCATE_PATH = '/ccn2/u/lilianch/external_repos/LOCATE'
sys.path.append(LOCATE_PATH)
os.chdir(LOCATE_PATH)  # Change to LOCATE directory for relative imports

# Now import LOCATE modules
from pathlib import Path
from types import SimpleNamespace
from detectron2.checkpoint import DetectionCheckpointer
import config
import utils as ut
from mask_former_trainer import setup, Trainer


def load_locate_model(checkpoint_path='/ccn2/u/lilianch/external_repos/LOCATE/checkpoints/combined.pth'):
    args = SimpleNamespace(
        config_file='/ccn2/u/lilianch/external_repos/LOCATE/eval/20250915_085420/config.yaml',
        opts=[
            "GWM.DATASET", "INTERNET",
        ],  # Use INTERNET for single images
        wandb_sweep_mode=False,
        resume_path=checkpoint_path,
        eval_only=True
    )

    # Setup config
    cfg = setup(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_state = ut.random_state.PytorchRNGState(seed=cfg.SEED).to(device)
    model = Trainer.build_model(cfg)

    # Load checkpoint
    checkpointer = DetectionCheckpointer(
        model,
        random_state=random_state,
        save_dir=os.path.join(cfg.OUTPUT_DIR, '../..', 'checkpoints')
    )

    print(f'Loading checkpoint from: {checkpoint_path}')
    checkpoint = checkpointer.resume_or_load(checkpoint_path, resume=False)
    model.eval()

    return model, cfg

def process_single_image(model, cfg, rgb_image):
    # Set up transforms (matching LOCATE's preprocessing)
    target_h, target_w = cfg.GWM.RESOLUTION
    transform_image = transforms.Compose([
        transforms.Resize((target_h, target_w), Image.Resampling.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Convert numpy array to PIL if needed
    if isinstance(rgb_image, np.ndarray):
        rgb_image = Image.fromarray(rgb_image)

    # Prepare sample in expected format
    sample = [{
        "image": transform_image(rgb_image),
        "height": target_h,
        "width": target_w,
        "dirname": "h5_data",
        "fname": "image.png"
    }]

    # Run inference
    with torch.no_grad():
        preds = model.forward_base(sample, keys=cfg.GWM.SAMPLE_KEYS, get_eval=True)
    breakpoint()
    mask = torch.sigmoid(preds[0]['sem_seg']).cpu()

    # Resize to original dimensions
    orig_h, orig_w = rgb_image.size[1], rgb_image.size[0]  # PIL uses (width, height)
    mask_resized = F.interpolate(
        mask.unsqueeze(0),
        size=(orig_h, orig_w),
        mode='bilinear',
        align_corners=False
    )[0][0].numpy()

    return mask_resized


def process_h5_dataset(h5_path, output_dir, checkpoint_path=None):
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading LOCATE model...")
    if checkpoint_path is None:
        checkpoint_path = '/ccn2/u/lilianch/external_repos/LOCATE/checkpoints/combined.pth'

    model, cfg = load_locate_model(checkpoint_path)

    print(f"Processing H5 file: {h5_path}")
    predictions = {}

    with h5py.File(h5_path, 'r') as f_in:
        image_names = list(f_in.keys())
        print(f"Found {len(image_names)} images")

        for img_name in tqdm(image_names, desc="Processing images"):
            rgb_array = f_in[img_name]['rgb'][:]

            # Run inference
            soft_mask = process_single_image(model, cfg, rgb_array)
            binary_mask = (soft_mask >= 0.5).astype(np.uint8)

            predictions[img_name] = {
                'soft_mask': soft_mask,
                'binary_mask': binary_mask
            }

            # Save first few as PNG for visualization
            if len(predictions) <= 10:
                mask_vis = (binary_mask * 255).astype(np.uint8)
                Image.fromarray(mask_vis).save(
                    os.path.join(output_dir, f"{img_name}_mask.png")
                )
                # Also save RGB for reference
                Image.fromarray(rgb_array).save(
                    os.path.join(output_dir, f"{img_name}_rgb.png")
                )

    # Save h5
    # output_h5 = os.path.join(output_dir, 'locate_predictions.h5')
    # print(f"Saving predictions to {output_h5}")
    #
    # with h5py.File(output_h5, 'w') as f_out:
    #     for img_name, pred_data in predictions.items():
    #         grp = f_out.create_group(img_name)
    #         grp.create_dataset('segment', data=pred_data['binary_mask'], compression='gzip')

    # Print statistics
    objects_found = sum(1 for p in predictions.values() if p['binary_mask'].sum() > 0)

    return predictions


if __name__ == "__main__":
    # Configuration
    H5_PATH = '/ccn2/u/lilianch/data/550_openx_entity_dataset.h5'
    OUTPUT_DIR = '/ccn2/u/lilianch/external_repos/LOCATE/vis'
    CHECKPOINT = '/ccn2/u/lilianch/external_repos/LOCATE/checkpoints/combined.pth'

    # Run processing
    predictions = process_h5_dataset(
        h5_path=H5_PATH,
        output_dir=OUTPUT_DIR,
        checkpoint_path=CHECKPOINT
    )

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("- locate_predictions.h5: All predictions in H5 format")
    print("- *_mask.png: First 10 binary masks for visualization")
    print("- *_rgb.png: First 10 RGB images for reference")