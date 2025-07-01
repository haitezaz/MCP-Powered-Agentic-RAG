from typing import TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import httpx
from fastmcp import Client


load_dotenv()

# Setup LLM
LLM = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

# Generation chain
generation_template = ChatPromptTemplate([
    ("system", '''You are a helpful AI bot. Answer from the context. If context is not enough, say "I don't know"'''),
    ("human", "Query: {query}\nContext: {context}"),
])
generation_chain = generation_template | LLM

# Evaluation model
class ResponseModel(BaseModel):
    response: Literal["Good", "Bad"] = Field(description="Either 'Good' or 'Bad'")

structured_llm = LLM.with_structured_output(ResponseModel)

class GraphState(TypedDict):
    query: str
    documents: List[Dict]
    transformed_query: str
    max_tries: int
    current_try: int
    final_ans: str
    web_documents: List[Dict]
    context_quality: Literal["Good", "Bad"]

class Agent:
    def __init__(self, client : Client):
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile() if hasattr(self, 'workflow') and self.workflow else None
        self.client = client
        
    async def startup(self):
        await self.client.__aenter__()  # Opens connection

    async def shutdown(self):
        await self.client.__aexit__(None, None, None)  # Closes connection

    # =============================================================================
    # NODE FUNCTIONS (these return GraphState dictionaries)
    # =============================================================================
    
    async def get_relevant_docs(self, state: GraphState) -> GraphState:
        print("accessing vector Db")
        """NODE: Get documents from vector database"""
        state["current_try"] += 1
     
        query_to_use = state.get("transformed_query") or state.get("query")
        print(query_to_use)
        if query_to_use:
            print("Error just on call")
            docs = await self.client.call_tool("VectorDBsearch", {"query":query_to_use})
            
            state["documents"] = docs
            
        return state
    
    async def transform_query(self, state: GraphState) -> GraphState:
        print("transforming query")
        """NODE: Transform the original query"""
        transformed = await self.client.call_tool("TransformQuery", {"query":state["query"]})
        if isinstance(transformed, list):
            transformed = " ".join(getattr(t, "text", str(t)) for t in transformed)
        state["transformed_query"] = transformed
        return state
    
    async def web_search(self, state: GraphState) -> GraphState:
        """NODE: Search the web"""
        web_results = await self.client.call_tool("webSearch", {"query":state["query"]})
        state["web_documents"] = web_results
        return state
    
    async def generate_answer(self, state: GraphState) -> GraphState:
        """NODE: Generate final answer"""
        # Choose context source
        if state.get("web_documents"):
            context = state["web_documents"]
        elif state.get("documents"):
            context = state["documents"]
        else:
            context = "No context available"
        
        response = generation_chain.invoke({
            "query": state["query"],
            "context": str(context)
        })
        
        state["final_ans"] = response.content
        return state
    
    async def evaluate_and_route(self, state: GraphState) -> GraphState:
        print("Entered node evaluate_and_route")
        """NODE: Evaluate context quality and update state"""
        if state.get("documents"):
            # Evaluate context
            context_text = "\n".join([str(doc) for doc in state["documents"]])
            prompt = f"""Evaluate if this context can answer the query:

Query: {state["query"]}
Context: {context_text}

Is this context Good or Bad for answering the query?"""
            
            result = structured_llm.invoke(prompt)
            state["context_quality"] = result.response
            
        return state
    
    # =============================================================================
    # ROUTING LOGIC (this returns a string - next node name)
    # =============================================================================
    
    def decide_next_step(self, state: GraphState) -> str:
        print("Entered decision")
        """CONDITIONAL EDGE: Decide which node to go to next"""
        
        # No query? End
        if not state.get("query"):
            return "END"
        
        # No documents yet and haven't tried too much? Get docs
        if not state.get("documents") and state["current_try"] < state["max_tries"]:
            return "get_relevant_docs"
        
        # Have web documents? Generate answer
        if state.get("web_documents"):
            return "generate_answer"
        
        # Have regular documents? Check quality
        if state.get("documents"):
            if state.get("context_quality") == "Good":
                return "generate_answer"
            elif state.get("context_quality") == "Bad":
                if state["current_try"] < state["max_tries"]:
                    return "transform_query"
                else:
                    return "web_search"
        
        # Default
        return "generate_answer"
    
    # =============================================================================
    # BUILD WORKFLOW
    # =============================================================================
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        
        # Add all nodes (functions that return GraphState)
        workflow.add_node("get_relevant_docs", self.get_relevant_docs)
        workflow.add_node("evaluate_and_route", self.evaluate_and_route) 
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Simple edges (always go from A to B)
        workflow.add_edge("get_relevant_docs", "evaluate_and_route")
        workflow.add_edge("transform_query", "get_relevant_docs") 
        workflow.add_edge("web_search", "evaluate_and_route")
        workflow.add_edge("generate_answer", END)
        
        # Conditional edge (decide where to go based on state)
        workflow.add_conditional_edges(
            "evaluate_and_route",  # FROM this node
            self.decide_next_step,  # USE this function to decide
            {  # POSSIBLE destinations
                "get_relevant_docs": "get_relevant_docs",
                "transform_query": "transform_query",
                "web_search": "web_search", 
                "generate_answer": "generate_answer"
            }
        )
        
        # Start here
        workflow.set_entry_point("evaluate_and_route")
        
        return workflow
    
    # =============================================================================
    # RUN THE AGENT
    # =============================================================================
    
    async def run(self, query: str) -> str:
        initial_state = GraphState(
            query=query,
            documents=[],
            transformed_query="",
            max_tries=2,
            current_try=0,
            final_ans="",
            web_documents=[],
            context_quality="Bad"
        )
        
        result = await self.app.ainvoke(initial_state)
        return result["final_ans"]