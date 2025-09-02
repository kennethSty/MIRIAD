import yaml
from typing import List, Dict, final
from pathlib import Path
from meddxagent.ddxdriver.models import init_model
from meddxagent.ddxdriver.utils import OutputDict, Constants
from meddxagent.ddxdriver.logger import log
from meddxagent.ddxdriver.rag_agents.miriad_retriever import MiriadRetriever
from meddxagent.ddxdriver.rag_agents.base import RAG as RAGBase
from meddxagent.ddxdriver.rag_agents.utils import (
        extract_and_eval_list,
        get_rag_synthesis_system_prompt, 
        get_rag_synthesis_user_prompt,
        get_create_questions_user_prompt
        )


class MiriadRAG(RAGBase):
    """
    RAG pipeline using data retrieved from MIRIAD.
    """
    def __init__(self, config):
        super().__init__(config)
        self.top_k_search = self.config["retrieval"].get("top_k_search", 3)
        self.max_question_searches = self.config["retrieval"].get("max_question_searches", 3)
        self.retriever = MiriadRetriever(self.config)
        self.model = init_model(
            self.config["model"]["class_name"], **self.config["model"]["config"]
        )
        self.rag_type = "miriad_rag"
    
    def get_type(self) -> str:
        return self.rag_type

    def get_max_num_searches(self) -> int:
        return self.max_question_searches

    @final
    def __call__(
        self,
        input_search: str,
        diagnosis_options: List[str] = [],
    ) -> Dict:
        """
        Process a query through the MIRIAD RAG pipeline.
        
        Args:
            input_search: The search query
            diagnosis_options: List of possible diagnosis options
            
        Returns:
            Dictionary containing the RAG content
        """

        log.info("Retrieving documents from MIRIAD...")
        questions = self.get_questions(input_search)
        retrieval_results = self.retrieve_documents(questions)
        self._log_docs(retrieval_results)
        
        # Summarize retrieved results
        output = self.synthesize_results(
            input_search=input_search,
            retrieval_results=retrieval_results,
            diagnosis_options=diagnosis_options
            )
        log.info("Rag syntesized content \n" + output + "\n")
        log.info("RAG processing completed successfully \n")
        return {OutputDict.RAG_CONTENT: output}
            
    def get_questions(self, input_search: str) -> List[str]: 
        """
        Given an input search formualted by the ddxdriver which contains relevant questions 
        whose answer would help to obtain the correct diagnosis for the patient, 
        extracts these questions into a list.
        """
        retry_counter = 0
        log.info("Start generating questions to guide later retrieval...")
        question_response = self.model(user_prompt=input_search)
        log.info(f"Created questions (raw): {question_response}") 
        extract_questions_prompt = get_create_questions_user_prompt(
            input_search=question_response, max_question_searches=self.max_question_searches
        )
        log.info(f"Prompt to extract parsable list of questions: {extract_questions_prompt}")

        questions = []
        message_history = []
        questions_str = ""
        while retry_counter <= Constants.RAG_RETRIES.value:
            try:
                questions_str = self.model(
                    user_prompt=extract_questions_prompt, message_history=message_history
                )
                questions = extract_and_eval_list(string=questions_str)
                questions = questions[:self.max_question_searches]
            except Exception as e:
                log.info(
                    f"Caught exception trying to generate/parse questions list, trying again: {e}\n"
                )
                questions = []
                if retry_counter <= Constants.RAG_RETRIES.value:
                    log.info(f"Current questions list: {questions_str}, trying again...\n")
                else:
                    log.info(f"Out of retries for questions, returning empty list...\n")
                    return []
            if questions and all(isinstance(x, str) for x in questions):
                break

            retry_counter+=1
            message_history.extend(
                [
                    {"role": "user", "content": extract_questions_prompt},
                    {"role": "assistant", "content": questions_str},
                ]
            )
            extract_questions_prompt = (
                "Your question list was not formatted correctly as a list of strings. "
                "Please edit its format so it can be parsed as such.\n"
                "Here is an example of formatting (replace the placeholders inside the arrow brackets, and do not include the arrow brackets themselves):\n"
                """["<QUESTION_1>", "<QUESTION_2>"]"""
            )

        if not questions or not all(isinstance(x, str) for x in questions):
            error_message = "Questions list was not correctly generated in time \n"
            raise ValueError(error_message)
        
        return questions

    def retrieve_documents(self, questions: List[str]):
        """
        Given a list of questions, uses retrieves the relevant self.top_k_search documents
        from the Miriad Qdrant knowledge base.
        """
        retrieval_results = []
        for question in questions:
            log.info(f"Retrieving documents for question: {question}")
            documents = self.retriever.retrieve_docs(query=question, top_k=self.top_k_search)
            # TODO: add synthetization step for each question
            retrieval_results.extend(documents)
        
        if not retrieval_results:
            # No graceful failing as db is always expected to return sth
            error_message = "Vector db is not returning documents \n"
            raise ValueError(error_message)

        return retrieval_results

    def synthesize_results(
        self,
        input_search: str,
        retrieval_results: List[str],
        diagnosis_options: List[str] = [],
    ) -> str:
        """
        Synthesize retrieval results into a coherent response.
        
        Args:
            input_search: Original search query
            retrieval_results: List of question answer pairs
            diagnosis_options: List of possible diagnoses
            
        Returns:
            Synthesized response string
        """
        
        retrieval_results_text = self._results_to_str(retrieval_results)
        system_prompt = get_rag_synthesis_system_prompt()
        user_prompt = get_rag_synthesis_user_prompt(
            input_search=input_search,
            search_results_text=retrieval_results_text,
            diagnosis_options=diagnosis_options,
        )
        
        print("Generating response with LLM...")
        output = self.model(user_prompt=user_prompt, system_prompt=system_prompt)
        
        return output

    def _log_docs(self, documents):
        """Logs a list of documents"""
        documents_text = self._format_docs_to_str(documents)
        log.info(f"Retrieved {len(documents)} documents:")
        log.info("=" * 60)
        log.info(documents_text)
        log.info("=" * 60)

    def _format_docs_to_str(self, documents) -> str:
        """Format retrieved documents for display."""
        formatted_docs = []
        
        for i, document in enumerate(documents):
            if hasattr(document, 'payload') and 'passage_text' in document.payload:
                text = document.payload['passage_text']
                score = getattr(document, 'score', 'N/A')
                formatted_docs.append(f"Document {i+1} (Score: {score}):\n{text}")
            else:
                raise AttributeError("Missing attributes 'payload' or 'passage_text' in retrieved documents")
        return "\n\n".join(formatted_docs)

    def _results_to_str(self, documents) -> str:
        """Convert documents to retrieval results format expected by synthesize_results."""
        
        retrieval_results = []
        for i, document in enumerate(documents):
            if hasattr(document, 'payload') and 'passage_text' in document.payload:
                text = document.payload['passage_text']
                title = f"Medical Document {i+1}"
            else:
                raise AttributeError("Missing attribute 'payload' in retrieved documents")
            retrieval_results.append(f"{i+1}. {title}\n{text}")
        
        return "\n\n".join(retrieval_results)
