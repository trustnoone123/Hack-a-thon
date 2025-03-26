import os
from langchain_neo4j import Neo4jGraph

from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_openai import ChatOpenAI

# Set up Neo4j credentials
NEO4J_URI = "neo4j+ssc://cdb64ee0.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "hjOjZbewCStXTbJPYgsmgse8mAXxzL1i_mw3l5xDDGM"

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)

llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")

llm_transformer = LLMGraphTransformer(llm=llm)

from langchain_core.documents import Document

text = """
i am jhanani studying 5th std in velammal engineering college.
"""
documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")