from mcp.server.fastmcp  import FastMCP
from pinecone import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
load_dotenv()

#Search tool 
Search_tool = TavilySearch(max_results = 3)


mcp = FastMCP("mcp-tools")


#load Embedding Model
Embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#setting up and loading vectorDB
pinecone_api_key = "PUT YOUR KEY ON .env AND MAKE ENVIRONMENT VARIABLE"
pc = Pinecone(api_key=pinecone_api_key)
mcp = FastMCP("mcp-tools")
index_name = "mcp-powered"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=Embeddings)
retriever = vector_store.as_retriever(search_type ="mmr",k = 7)


# Add Cohere reranker
reranker = CohereRerank(
    top_n=3,
    model="rerank-english-v3.0"
)


#Make final retriever
final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)

#setup LLM
LLM = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct" )
#Make the chain to transform the query
template = ChatPromptTemplate([
    ("system", '''You are a helpful AI bot. Your are provided with a query that failed to retrieve relevant docs from
     vector store. Your have to transform this query so it can get The relevant doc.Answer only the transform query and 
     no other words as this is gonna crash my system'''),
    ("human", "{query}"),])

transform_chain = template | LLM



@mcp.tool()
async def VectorDBsearch(query):
    '''This fuction is used to search the Vector Database'''
    print("searching vector DB")
    response = final_retriever.invoke(query)
    return response

@mcp.tool()
async def TransformQuery(query) -> str:
    '''This function is used to transform query if original query fails to get relevant docs'''
    print("Transforming query")
    response = transform_chain.invoke(query)
    
    # ✅ Handle TextContent list or plain string
    content = response.content

    if isinstance(content, list):
        # Join all text entries safely
        content = " ".join(getattr(c, "text", str(c)) for c in content)
    print(type(content))
    return content.strip()

@mcp.tool()
async def webSearch(query):
    '''This function is used to do Web Search to get information about query from Internet'''
    print("Web Search activated")
    response = Search_tool.invoke(query)
    return response



if __name__ == "__main__":
    mcp.run(transport="streamable-http")


