from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

class Summarizer:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(temperature=0, openai_api_key=api_key)

    def map_reduce_summarize(self, documents):
        # Map chain setup
        map_template = """<MAP PROMPT TEMPLATE HERE>"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        # Reduce chain setup
        reduce_template = """<REDUCE PROMPT TEMPLATE HERE>"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")
        reduce_documents_chain = ReduceDocumentsChain(combine_documents_chain=combine_documents_chain, collapse_documents_chain=combine_documents_chain, token_max=4000)
        map_reduce_chain = MapReduceDocumentsChain(llm_chain=map_chain, reduce_documents_chain=reduce_documents_chain, document_variable_name="docs", return_intermediate_steps=False)

        return map_reduce_chain.run(documents)

    def refine_summarize(self, documents):
        question_template = """<QUESTION PROMPT TEMPLATE HERE>"""
        question_prompt = PromptTemplate.from_template(question_template)

        refine_template = """<REFINE PROMPT TEMPLATE HERE>"""
        refine_prompt = PromptTemplate.from_template(refine_template)

        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )
        return chain({"input_documents": documents}, return_only_outputs=True)["output_text"]