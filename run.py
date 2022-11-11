import argparse
import climatenet.train as train
from climatenet.train import MODELS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Model to use for training.",
        choices=list(MODELS.keys()),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to data directory.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path of a model checkpoint to load",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path of the folder where the generated data will be saved"
    )

    args = parser.parse_args()
    print(vars(args))

    print(f"Running {args.model}...")
    train.run(
        model_name=args.model,
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        save_dir=args.save_dir
    )