import google.generativeai as genai
from simiir.user.query_generators.base import BaseQueryGenerator
from ifind.search.query import Query
import re

GOOGLE_API_KEY = "" 
genai.configure(api_key=GOOGLE_API_KEY)


class TripleLLMQueryGenerator(BaseQueryGenerator):
    def __init__(self, query_file, user, stopword_file=None, background_file=None, user_type=""):
        super().__init__(stopword_file=stopword_file, background_file=background_file)

        self.__query_filename = query_file
        self.__user = user
        self.user_type = user_type
        self._query_list = []  
        self._query_index = 0

        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def classify_query(self, user_context):
        queries_and_ids = self.generate_query_list(user_context)
        queries = "\n".join([q for q, _ in queries_and_ids])

        if self.user_type == "boolean":
            prompt = f"""
        You are a scientific researcher generating search queries for a research database.

        Query List:
        {queries}

        Task: Based on the Query List, write 10 boolean queries that could be the next logical follow up, with 5 to 10 keywords, using AND/OR to combine them.
        IMPORTANT: Prioritize queries that are most likely to lead to relevant documents. Order these 10 queries from most promising to least promising.
        Only print the queries, do not print any other text.
        Example: "(""Social networks"" OR ""Social media"" OR ""networks"" OR ""Digital platforms"" OR ""Peer networks"") AND (""influence"" OR ""Virtual communities"" OR ""connectivity"" OR ""Social platforms"" OR ""Social sharing sites"")"
        """
        elif self.user_type == "question":
            prompt = f"""
        You are a basic user generating search queries for a research database.

        Query List:
        {queries}

        Task: Based on the Query List, write 10 queries that could be the next logical follow up, each with 5 to 10 keywords including a question mark.
        IMPORTANT: Prioritize queries that are most likely to lead to relevant documents. Order these 10 queries from most promising to least promising.
        Only print the queries, do not print any other text.
        Example: "How affect general anesthesia on blood glucose?", "how did technology change the way students way of learning?"
        """
        elif self.user_type == "basic":
            prompt = f"""
        You are a basic user generating search queries for a research database.

        Query List:
        {queries}

        Task: Based on the Query List, write 10 queries that could be the next logical follow up with 5 to 10 keywords.
        IMPORTANT: Prioritize queries that are most likely to lead to relevant documents. Order these 10 queries from most promising to least promising.
        Only print the queries, do not print any other text.
        Example: "Farmers Income Problems in philippines", "Pierre Bourdieu theory of State cultural capital symbolic capital analysis"
        """
        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
            queries = self.sanitize_and_split_queries(raw_text)
            return queries if queries else []
        except Exception as e:
            print(f"[ERROR] LLM query generation failed: {str(e)}")
            return []

    def get_next_query(self, user_context):
        if not self._query_list or self._query_index >= len(self._query_list):
            new_queries = self.classify_query(user_context)
            self._query_list = new_queries
            self._query_index = 0

        if self._query_index < len(self._query_list):
            raw_query = self._query_list[self._query_index]
            self._query_index += 1

            sanitized = re.sub(r'["()]+', '', raw_query)
            sanitized = re.sub(r'\b\w+:', '', sanitized)
            return sanitized

    def sanitize_and_split_queries(self, llm_response: str):
        cleaned = re.sub(r'[\"\(\)]', '', llm_response)
        queries = re.split(r'[\n,]+|\d+\.\s*', cleaned)
        return [q.strip() for q in queries if q.strip()]

    def generate_query_list(self, user_context):
        """
        Returns list of predetermined queries for the specified user and topic.
        This method is likely called by classify_query to get initial context for the LLM.
        """
        topic = user_context.topic
        queries = []

        with open(self.__query_filename, 'r') as queries_file:
            for line in queries_file:
                line = line.strip()
                if not line: 
                    continue
                parts = line.split(',')
                if len(parts) < 4:
                    continue
                line_qid, line_user, line_topic, *line_terms = parts
                if line_user == self.__user and str(line_topic) == str(topic.id):
                    queries.append((' '.join(line_terms), int(line_qid)))
        queries.sort(key=lambda x: x[1])
        return queries

    def generate_query(self, user_context):
        return self.get_next_query(user_context)
