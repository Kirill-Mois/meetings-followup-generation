from pathlib import Path
from langchain_community.document_loaders import UnstructuredMarkdownLoader


class DataLoader:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def load_markdown(self):
        loader = UnstructuredMarkdownLoader(self.file_path)
        data = loader.load()
        return data[0].page_content

    def load(self):
        if self.file_path.suffix == ".md":
            return self.load_markdown()
        raise ValueError("Incorrect file format")
