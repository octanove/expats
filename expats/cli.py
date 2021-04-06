
import argparse

from expats.common.config_util import load_from_file, merge_with_dotlist
from expats.common.log import init_setup_log, get_logger
from expats.task import train, evaluate, predict, interpret


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser("EXPATS: A toolkit for explainable automated text scoring")
    sub_parsers = parser.add_subparsers()

    train_parser = sub_parsers.add_parser("train", help="Training profiler")
    train_parser.add_argument("config_path", type=str, help="path to config yaml file")
    train_parser.add_argument("artifact_path", type=str, help="path to training artifacts")
    train_parser.add_argument("--overrides", type=str, nargs='*', help="overide configurations")
    train_parser.set_defaults(mode="train")

    evaluate_parser = sub_parsers.add_parser("evaluate", help="Evaluation of trained profiler")
    evaluate_parser.add_argument("config_path", type=str, help="path to config yaml file")
    evaluate_parser.add_argument("--overrides", type=str, nargs='*', help="overide configurations")
    evaluate_parser.set_defaults(mode="evaluate")

    predict_parser = sub_parsers.add_parser("predict", help="Run inference with trained profiler")
    predict_parser.add_argument("config_path", type=str, help="path to config yaml file")
    predict_parser.add_argument("output_path", type=str, help="path to prediction output file")
    predict_parser.add_argument("--overrides", type=str, nargs='*', help="overide configurations")
    predict_parser.set_defaults(mode="predict")

    interpret_parser = sub_parsers.add_parser("interpret", help="Launch LIT server to interpret trained profiler")
    interpret_parser.add_argument("config_path", type=str, help="path to config yaml file")
    interpret_parser.add_argument("--overrides", type=str, nargs='*', help="overide configurations")
    interpret_parser.set_defaults(mode="interpret")

    args = parser.parse_args()

    if args.mode == "train":
        init_setup_log(args.artifact_path)
        logger.info("##### Training #####")
        logger.info(f"args: {sorted(vars(args).items())}")
        config = load_from_file(args.config_path)
        if args.overrides:
            config = merge_with_dotlist(config, args.overrides)
        logger.info(f"Loaded config: {config}")
        train(config, args.artifact_path)
    elif args.mode == "evaluate":
        init_setup_log()  # FIXME: only stdout logging is better?
        logger.info("##### Evaluation #####")
        logger.info(f"args: {sorted(vars(args).items())}")
        config = load_from_file(args.config_path)
        if args.overrides:
            config = merge_with_dotlist(config, args.overrides)
        logger.info(f"Loaded config: {config}")
        evaluate(config)
    elif args.mode == "predict":
        init_setup_log()  # FIXME: only stdout logging is better?
        logger.info("##### Prediction #####")
        logger.info(f"args: {sorted(vars(args).items())}")
        config = load_from_file(args.config_path)
        if args.overrides:
            config = merge_with_dotlist(config, args.overrides)
        logger.info(f"Loaded config: {config}")
        predict(config, args.output_path)
    elif args.mode == "interpret":
        init_setup_log()  # FIXME: only stdout logging is better?
        logger.info("##### Interactive interpretation #####")
        logger.info(f"args: {sorted(vars(args).items())}")
        config = load_from_file(args.config_path)
        if args.overrides:
            config = merge_with_dotlist(config, args.overrides)
        logger.info(f"Loaded config: {config}")
        interpret(config)

    logger.info("Done")


if __name__ == "__main__":
    main()
