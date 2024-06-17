from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from src.config import SummarizerConfig
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents.base import Document


class Summarizer:
    def __init__(self, config: SummarizerConfig):
        self.llm = ChatOpenAI(temperature=0)
        self.config = config

    def split_recursive(self, text: str) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return splitter.create_documents([text])

    def split_semantic(self, text: str) -> list[Document]:
        semantic_splitter = SemanticChunker(
            OpenAIEmbeddings(),
            breakpoint_threshold_type=self.config.breakpoint_threshold_type,
        )
        return semantic_splitter.create_documents([text])

    def split_text(self, text: str) -> list[Document]:
        if self.config.split_method_name == "recursive":
            return self.split_recursive(text)

        if self.config.split_method_name == "semantic":
            return self.split_semantic(text)

    def __call__(self, text: str) -> str:
        raise NotImplementedError


class MapReduceSummarizer(Summarizer):
    def __init__(self, config: SummarizerConfig):
        super().__init__(config)

        # Map chain setup
        map_prompt = PromptTemplate.from_template(config.map_template)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        # Reduce chain setup
        reduce_prompt = PromptTemplate.from_template(config.reduce_template)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,
        )
        self.chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

    def __call__(self, text: str) -> str:
        documents = self.split_text(text)
        return self.chain.run(documents)


class RefineSummarizer(Summarizer):
    def __init__(self, config: SummarizerConfig):
        super().__init__(config)

        question_prompt = PromptTemplate.from_template(config.question_template)

        refine_prompt = PromptTemplate.from_template(config.refine_template)

        self.chain = load_summarize_chain(
            llm=self.llm,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )

    def __call__(self, text):
        documents = self.split_text(text)
        return self.chain({"input_documents": documents}, return_only_outputs=True)[
            "output_text"
        ]


CHAIN_NAME_TO_CLASS = {"map_reduce": MapReduceSummarizer, "refine": RefineSummarizer}
