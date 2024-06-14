import os
from src.config import OPENAI_API_KEY
from src.data_loader import DataLoader
from src.text_splitter import TextSplitter
from src.summarizer import Summarizer
from src.evaluator import Evaluator

def main():
    # Load data
    markdown_path = "path/to/your/markdown_file.md"
    loader = DataLoader(markdown_path)
    markdown_text = loader.load_markdown()

    # Split text
    splitter = TextSplitter(markdown_text)
    md_docs = splitter.split_recursive()
    embed_docs = splitter.split_semantic()

    # Summarize
    summarizer = Summarizer(OPENAI_API_KEY)
    map_reduce_result = summarizer.map_reduce_summarize(md_docs)
    refine_result = summarizer.refine_summarize(md_docs)

    print("Map-Reduce Result:")
    print(map_reduce_result)
    
    print("Refine Result:")
    print(refine_result)

    # Evaluate
    ideal_summary = "Your ideal summary here"
    evaluator = Evaluator(OPENAI_API_KEY)
    map_reduce_evaluation = evaluator.evaluate(map_reduce_result, ideal_summary)
    refine_evaluation = evaluator.evaluate(refine_result, ideal_summary)

    print("Map-Reduce Evaluation:")
    print(map_reduce_evaluation)

    print("Refine Evaluation:")
    print(refine_evaluation)

if __name__ == "__main__":
    main()