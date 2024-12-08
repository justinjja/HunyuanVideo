import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler


class ModelSingleton:
    """
    Singleton class to keep the HunyuanVideoSampler model loaded in memory.
    """
    _instance = None
    _model = None

    @staticmethod
    def get_instance(args):
        """
        Get the singleton instance of the model. Initializes the model if not already loaded.
        """
        if ModelSingleton._instance is None:
            ModelSingleton(args)
        return ModelSingleton._model

    def __init__(self, args):
        if ModelSingleton._instance is not None:
            raise Exception("This is a singleton class!")
        models_root_path = Path(args.model_base)
        if not models_root_path.exists():
            raise ValueError(f"`models_root` not exists: {models_root_path}")

        # Load the model once
        logger.info("Loading the HunyuanVideoSampler model...")
        ModelSingleton._model = HunyuanVideoSampler.from_pretrained(
            models_root_path, args=args
        )
        logger.info("Model loaded successfully.")
        ModelSingleton._instance = self


def main():
    # Parse the command-line arguments
    args = parse_args()
    print(args)

    # Get the model instance (singleton)
    model = ModelSingleton.get_instance(args)

    # Prepare save path
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Start sampling
    outputs = model.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
    )
    samples = outputs['samples']

    # Save samples
    for i, sample in enumerate(samples):
        sample = samples[i].unsqueeze(0)
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        video_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/', '')}.mp4"
        save_videos_grid(sample, video_save_path, fps=24)
        logger.info(f"Sample saved to: {video_save_path}")


if __name__ == "__main__":
    main()
