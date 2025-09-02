from dis import UNKNOWN

from src.state.emailstate import EmailState
from src.logger.logger import Logger
from src.prompts.emailprompts import EmailPrompt
from src.llms.groqllm import GroqLLM
#from langchain.chains import Chain
import json
from src.workflows.emailworkflow import EmailGraph

logging = Logger.get_logger(__name__)
class EmailProcessor():
    def __init__(self, email_state: EmailState):
        self.email_state = email_state


    def start_email_processing(self):
        graph = EmailGraph()
        graph_builder = graph.create_graph()
        compiled_graph = graph_builder.compile()
        result = compiled_graph.invoke(self.email_state)
        # print(result)









