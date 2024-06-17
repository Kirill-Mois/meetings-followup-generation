import argparse
from pathlib import Path
import time
from src.config import SummarizerConfig
from src.data_loader import DataLoader
from src.summarizer import CHAIN_NAME_TO_CLASS
from src.evaluator import Evaluator

from dotenv import load_dotenv

load_dotenv()


def main():
    # Load data
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--markdown_path", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config_path
    markdown_path = args.markdown_path

    loader = DataLoader(markdown_path)
    markdown_text = loader.load()

    config = SummarizerConfig.from_file(config_path)
    summarizer = CHAIN_NAME_TO_CLASS[config.chain_name](config)

    start_time = time.time()
    result = summarizer(markdown_text)
    end_time = time.time()
    time_to_complete = end_time - start_time

    print(result)

    # Evaluate
    evaluator = Evaluator()
    evaluation = evaluator.evaluate(result)

    print("---\nEvaluation:")
    print("Time to complete:", time_to_complete)
    print(evaluation)


if __name__ == "__main__":
    main()
