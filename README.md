![image](https://github.com/user-attachments/assets/dc93f38c-cc5b-4463-83fc-20fc30fdd6ec)
# MCP-Powered Agentic RAG System

A self-correcting Retrieval-Augmented Generation (RAG) system that intelligently evaluates context quality and falls back to web search when needed. Built to solve the common problem of semantic similarity leading to irrelevant retrievals in domain-specific applications.

## 🎯 Problem Statement

Traditional RAG systems often fail in specialized domains like entrepreneurship where:
- Multiple documents contain semantically similar content
- Retrieved context may be topically related but not practically relevant
- Users need precise, actionable answers rather than general information
- Standard similarity search returns "close enough" matches that miss the point

## 🚀 Solution

This agentic RAG system implements a multi-stage evaluation and correction pipeline:

1. **Initial Retrieval**: MMR-based vector search in Pinecone
2. **Smart Reranking**: Cohere API reranks results via MCP server
3. **Context Evaluation**: Agent assesses if retrieved context is sufficient
4. **Query Transformation**: If context is inadequate, agent reformulates the query
5. **Secondary Search**: Re-search vector DB with improved query
6. **Web Search Fallback**: If still insufficient, fall back to web search via MCP
7. **Answer Generation**: Generate response with complete, relevant context

## 🏗️ Architecture

### Core Components
- **LangGraph**: Orchestrates the agentic workflow
- **LangChain**: Handles RAG pipeline components
- **FastMCP**: Enables modular tool integration
- **Pydantic**: Provides structured outputs for flow control

### Infrastructure
- **Vector Database**: Pinecone (cloud-based)
- **Reranking**: Cohere API
- **LLM**: Meta-Llama/Llama-4-Scout-17b-16e-instruct
- **API**: FastAPI endpoints
- **Retrieval Strategy**: MMR + Reranking

## ⚡ Performance

- **Response Time**: < 6-7 seconds (even with full graph execution)
- **Self-Correction**: Automatically detects and corrects inadequate retrievals
- **Fallback Coverage**: Web search ensures no query goes unanswered
- **Context Quality**: Significantly improved relevance over standard RAG

## 🛠️ Technical Highlights

- **Agentic Decision Making**: System autonomously decides when to re-search or fallback
- **MCP Integration**: Seamless tool orchestration despite FastMCP complexity
- **Structured Flow Control**: Pydantic models guide graph execution paths
- **Multi-Modal Retrieval**: Combines vector search with web search capabilities

## 📊 Use Cases

- **Domain-Specific Q&A**: Entrepreneurship, technical documentation, specialized knowledge
- **Research Assistance**: When comprehensive, accurate answers are critical
- **Educational Applications**: Complex topics requiring precise information
- **Professional Support**: Business intelligence and decision support systems

## 🔧 Installation & Usage

```bash
# Clone the repository
git clone [repository-url]
cd agentic-rag-system

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file in the root directory and add your API keys:
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Run the MCP server first
python server.py

# Deploy your server somewhere or keep it running locally
# Update the server URL in app.py to point to your deployed/local server

# Run the FastAPI application
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 🚀 Quick Start
1. **Setup Environment**: Create `.env` file with all required API keys
2. **Start MCP Server**: Run `python server.py` 
3. **Configure Server URL**: Update the server URL in `app.py`
4. **Launch Application**: Use `uvicorn app:app --host 0.0.0.0 --port 8000`

## 📋 API Endpoints

```
POST /query
{
  "question": "Your question here",
  "context_threshold": 0.8
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastMCP team for the challenging but powerful tool integration framework
- Cohere for excellent reranking capabilities
- Pinecone for reliable vector storage
- LangChain community for comprehensive RAG tooling
