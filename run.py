import argparse
import climatenet.trained_cgnet.trained_cgnet as trained_cgnet
import climatenet.trained_upernet.trained_upernet as trained_upernet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Model to use for training.",
        choices=[
            "cgnet",
            "upernet"
        ],
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

    if args.model == "cgnet":
        print("Running baseline example...")
        trained_cgnet.run(
            checkpoint_path=args.checkpoint_path,
            data_dir=args.data_dir,
            save_dir=args.save_dir
        )
    elif args.model == "upernet":
        print("Running UperNet example...")
        trained_upernet.run(
            checkpoint_path=args.checkpoint_path,
            data_dir=args.data_dir,
            save_dir=args.save_dir
        )
    else:
        raise NotImplementedError("Not implemented yet")