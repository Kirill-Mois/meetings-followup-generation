from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

class TextSplitter:
    def __init__(self, text):
        self.text = text

    def split_recursive(self):
        splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=20000, chunk_overlap=4000)
        return splitter.create_documents([self.text])

    def split_semantic(self):
        semantic_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation")
        return semantic_splitter.create_documents([self.text])