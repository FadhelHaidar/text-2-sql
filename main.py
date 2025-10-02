from groq import Groq
from vanna.openai import OpenAI_Chat

from urllib.parse import urlparse
from src.settings import settings
from src.text_to_sql import Postgre_VectorStore


class Text2SQL(OpenAI_Chat, Postgre_VectorStore):
    def __init__(self, config=None):

        self.client = Groq(api_key=config["api_key"])
        
        Postgre_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=self.client, config=config)
    
    def log(self, message, title = "Info"):
        return


config = {
    "api_key": settings.groq_api_key,
    "model": "llama-3.1-8b-instant",
    "database_url": settings.postgres_url,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}

text2sql = Text2SQL(config=config)

parsed_url = urlparse(settings.postgres_url)
text2sql.connect_to_postgres(
    host=parsed_url.hostname, 
    dbname=parsed_url.path[1:], 
    user=parsed_url.username, 
    password=parsed_url.password, 
    port=parsed_url.port
)

if __name__ == "__main__":

    while True:
        question = input("Ask Text2SQL: ")
        if question.lower() == "exit":
            break

        sql = text2sql.generate_sql(question, allow_llm_to_see_data=True)
        print(f"\nSQL : {sql}")
        print(f"\nResponse : {text2sql.run_sql(sql)}")