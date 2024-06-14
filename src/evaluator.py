from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

class Evaluator:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(temperature=0, openai_api_key=api_key)

    def evaluate(self, generated_summary, ideal_summary):
        evaluation_template = """<EVALUATION PROMPT TEMPLATE HERE>"""
        evaluation_prompt = PromptTemplate.from_template(evaluation_template)
        evaluation_message = HumanMessage(content=evaluation_prompt.format(docs=generated_summary))

        return self.llm([evaluation_message]).content