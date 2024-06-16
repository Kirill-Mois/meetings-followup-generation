from config import SummarizerConfig
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


class TextSplitter:
    def __init__(self, text, config: SummarizerConfig):
        self.text = text
        self.config = config

    def split_recursive(self):
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return splitter.create_documents([self.text])

    def split_semantic(self):
        semantic_splitter = SemanticChunker(
            OpenAIEmbeddings(),
            breakpoint_threshold_type=self.config.breakpoint_threshold_type,
        )
        return semantic_splitter.create_documents([self.text])
