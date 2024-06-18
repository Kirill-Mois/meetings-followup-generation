from config import SummarizerConfig
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents.base import Document


class TextSplitter:
    def __init__(self, config: SummarizerConfig):
        self.config = config

    @classmethod
    def from_config(cls, config):
        return SPLIT_METHOD_NAME_TO_CLASS[config.split_method_name](config)

    def __call__(self, text: str) -> list[Document]:
        raise NotImplementedError


class RecursiveTextSplitter(TextSplitter):
    def __call__(self, text: str) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return splitter.create_documents([text])


class SemanticTextSplitter(TextSplitter):
    def __call__(self, text: str) -> list[Document]:
        semantic_splitter = SemanticChunker(
            OpenAIEmbeddings(),
            breakpoint_threshold_type=self.config.breakpoint_threshold_type,
        )
        return semantic_splitter.create_documents([text])


SPLIT_METHOD_NAME_TO_CLASS = {
    "recursive": RecursiveTextSplitter,
    "semantic": SemanticTextSplitter,
}
