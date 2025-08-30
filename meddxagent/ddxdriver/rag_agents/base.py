from abc import ABC, abstractmethod
from typing import List, Dict

class RAG(ABC):
    @abstractmethod
    def __init__(self, config):
        "Initialize the rag agent"
        self.config = config

    @abstractmethod
    def get_max_num_searches(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        input_search: str,
        diagnosis_options: List[str] = [],
    ) -> str:
        """
        Retrieves relevant documents given the input text, then returns content from the output (an "answer")
        Params:
            input_search (str): Full search of information the doctor wants (can be formatted in free text)

        Output:
            (str) Text answering the doctor's input search
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_type(self) -> str:
        """Return string description of the rag type being done"""
        raise NotImplementedError

    @abstractmethod
    def synthesize_results(
        self,
        input_search: str,
        search_results: List[Dict[str, str]],
        diagnosis_options: List[str] = [],
    ):
        """
        Given new search results, synthesizes them and returns rag output as a string (using summarization, extra retrieval, etc)
        Search results will be formatted as list of dictionaries in this form:
        {
            "title": <title of search result>
            "content": <content of search result>
        }
        """
