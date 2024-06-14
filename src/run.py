
import argparse
import time
from config import SummarizerConfig
from src.data_loader import DataLoader
from src.text_splitter import TextSplitter
from src.summarizer import CHAIN_NAME_TO_CLASS
from src.evaluator import Evaluator

from dotenv import load_dotenv
load_dotenv()


def main():
    # Load data
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--markdown_path", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config_path
    markdown_path = args.markdown_path

    loader = DataLoader(markdown_path)
    markdown_text = loader.load_markdown()

    config = SummarizerConfig.from_file(config_path)
    # Split text
    splitter = TextSplitter(markdown_text, config)
    md_docs = splitter.split_recursive()
    # embed_docs = splitter.split_semantic()
    summarizer = CHAIN_NAME_TO_CLASS[config.chain_name](config)

    result = summarizer(md_docs)

    print(result)

    # Evaluate
    evaluator = Evaluator()
    evaluation = evaluator.evaluate(result)

    print("Evaluation:")
    print(evaluation)


if __name__ == "__main__":
    main()
