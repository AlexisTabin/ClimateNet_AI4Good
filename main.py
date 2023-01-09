import argparse
from climatenet_plus.base_model import train, evaluate
from climatenet_plus.climatenet.utils.utils import Config
from cl.model import curriculum_train, curriculum_evaluate
from configparser import ConfigParser

def run(args):
    if args.model == "base":
        config = Config('base/config.json')
        if args.mode == "train":
            train(config)
        else:
            evaluate(config)
    else:
        if args.mode == "train":
            config = ConfigParser()
            config.read('cl/config.yaml')
            curriculum_train(config)
        else:
            config = ConfigParser()
            config.read('cl/config.yaml')
            curriculum_evaluate(config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "curriculum"], required=True)
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    args = parser.parse_args()
    run(args)

if __name__=="__main__":
    main()