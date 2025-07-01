from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import asyncio
from fastmcp import Client
from agent.rag_agent import Agent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


class QueryInput(BaseModel):
    query: str
# Global variable to hold the agent
rag_agent_ = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_agent_
    client = Client("http://127.0.0.1:8000/mcp")
    # Startup: Create MCP client and agent
    
    rag_agent_ = Agent(client)
    await rag_agent_.startup()
    print("RAG Agent initialized")
        
    yield  # App runs here
        
    # Shutdown: Cleanup happens automatically
    print("Shutting down RAG Agent")

app = FastAPI(lifespan=lifespan)


@app.post("/query")
async def query_endpoint(input: QueryInput):
    global rag_agent_
    if not rag_agent_:
        raise HTTPException(500, "Agent not initialized")

    result = await rag_agent_.run(input.query)
    return {"answer": result}

app.mount("/", StaticFiles(directory="static", html=True), name="static")