#!/usr/bin/env python3
"""Generate LLM Orchestration training dataset.

Covers 8 sub-domains:
1. LangChain (chains, agents, tools, memory, callbacks, LCEL)
2. LlamaIndex (indices, query engine, retrievers, node parsers)
3. RAG patterns (naive, hybrid, reranking, HyDE, RAPTOR, GraphRAG)
4. Multi-agent systems (CrewAI, AutoGen, Swarm, MoA, debate)
5. Prompt engineering (few-shot, chain-of-thought, ReAct, tree-of-thought)
6. Function calling / tool use (OpenAI, Claude, Mistral)
7. MCP protocol (tools, resources, prompts, transports)
8. Evaluation (RAGAS, judge LLM, human eval, automated benchmarks)

Output: JSONL to stdout.

Usage::

    uv run python scripts/gen_llm_orch_dataset.py > /tmp/gen-llm-orch.jsonl
"""
from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stderr)
logger = logging.getLogger(__name__)


def msg(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]}


# ---------------------------------------------------------------------------
# 1. LangChain
# ---------------------------------------------------------------------------
def gen_langchain() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What is LCEL (LangChain Expression Language) and how do I use it?",
        "LCEL is LangChain's declarative syntax for composing chains using the pipe operator (`|`):\n\n"
        "```python\nfrom langchain_core.prompts import ChatPromptTemplate\n"
        "from langchain_openai import ChatOpenAI\nfrom langchain_core.output_parsers import StrOutputParser\n\n"
        "prompt = ChatPromptTemplate.from_template('Tell me a joke about {topic}')\n"
        "model = ChatOpenAI(model='gpt-4o')\nparser = StrOutputParser()\n\n"
        "chain = prompt | model | parser\n\nresult = chain.invoke({'topic': 'programming'})\n```\n\n"
        "LCEL advantages:\n"
        "- Automatic streaming support: `async for chunk in chain.astream({...})`\n"
        "- Automatic batching: `chain.batch([{'topic': 'cats'}, {'topic': 'dogs'}])`\n"
        "- Built-in retry/fallback: `chain.with_retry(stop_after_attempt=3)`\n"
        "- Tracing integration with LangSmith\n"
        "- Type inference across chain components"
    ))
    pairs.append(msg(
        "How do I create a LangChain agent with tools?",
        "```python\nfrom langchain_openai import ChatOpenAI\nfrom langchain.agents import create_tool_calling_agent, AgentExecutor\n"
        "from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder\n"
        "from langchain_core.tools import tool\n\n"
        "@tool\ndef search(query: str) -> str:\n    \"\"\"Search the web for information.\"\"\"\n    return f'Results for: {query}'\n\n"
        "@tool\ndef calculator(expression: str) -> str:\n    \"\"\"Evaluate a math expression.\"\"\"\n    return str(eval(expression))\n\n"
        "llm = ChatOpenAI(model='gpt-4o')\ntools = [search, calculator]\n\n"
        "prompt = ChatPromptTemplate.from_messages([\n"
        "    ('system', 'You are a helpful assistant.'),\n"
        "    MessagesPlaceholder('chat_history', optional=True),\n"
        "    ('human', '{input}'),\n    MessagesPlaceholder('agent_scratchpad'),\n])\n\n"
        "agent = create_tool_calling_agent(llm, tools, prompt)\nexecutor = AgentExecutor(agent=agent, tools=tools, verbose=True)\n\n"
        "result = executor.invoke({'input': 'What is 25 * 47?'})\n```"
    ))
    pairs.append(msg(
        "How does LangChain memory work for multi-turn conversations?",
        "LangChain provides several memory types:\n\n"
        "**1. ConversationBufferMemory (full history):**\n"
        "```python\nfrom langchain.memory import ConversationBufferMemory\nmemory = ConversationBufferMemory(return_messages=True)\n```\n\n"
        "**2. ConversationBufferWindowMemory (last K turns):**\n"
        "```python\nmemory = ConversationBufferWindowMemory(k=5, return_messages=True)\n```\n\n"
        "**3. ConversationSummaryMemory (LLM-summarized):**\n"
        "```python\nmemory = ConversationSummaryMemory(llm=ChatOpenAI(), return_messages=True)\n```\n\n"
        "**4. ConversationTokenBufferMemory (token-limited):**\n"
        "```python\nmemory = ConversationTokenBufferMemory(llm=ChatOpenAI(), max_token_limit=2000)\n```\n\n"
        "Modern approach (LCEL + RunnableWithMessageHistory):\n"
        "```python\nfrom langchain_core.runnables.history import RunnableWithMessageHistory\n"
        "from langchain_community.chat_message_histories import ChatMessageHistory\n\n"
        "store = {}\ndef get_session_history(session_id):\n    if session_id not in store:\n"
        "        store[session_id] = ChatMessageHistory()\n    return store[session_id]\n\n"
        "chain_with_history = RunnableWithMessageHistory(chain, get_session_history)\n```"
    ))
    pairs.append(msg(
        "How do I use LangChain callbacks for tracing and logging?",
        "```python\nfrom langchain_core.callbacks import BaseCallbackHandler\nfrom langchain_openai import ChatOpenAI\n\n"
        "class LoggingHandler(BaseCallbackHandler):\n"
        "    def on_llm_start(self, serialized, prompts, **kwargs):\n"
        "        print(f'LLM call started with {len(prompts)} prompts')\n\n"
        "    def on_llm_end(self, response, **kwargs):\n"
        "        print(f'LLM call completed: {response.generations[0][0].text[:50]}...')\n\n"
        "    def on_tool_start(self, serialized, input_str, **kwargs):\n"
        "        print(f'Tool {serialized[\"name\"]} called with: {input_str}')\n\n"
        "    def on_chain_error(self, error, **kwargs):\n"
        "        print(f'Chain error: {error}')\n\n"
        "llm = ChatOpenAI(callbacks=[LoggingHandler()])\n\n"
        "# Or pass at invocation time:\nchain.invoke({'input': 'hello'}, config={'callbacks': [LoggingHandler()]})\n```\n\n"
        "For production, use LangSmith tracing:\n```python\nimport os\nos.environ['LANGCHAIN_TRACING_V2'] = 'true'\n"
        "os.environ['LANGCHAIN_API_KEY'] = 'ls_...'\n```"
    ))
    pairs.append(msg(
        "How do I build a RAG chain with LangChain?",
        "```python\nfrom langchain_openai import ChatOpenAI, OpenAIEmbeddings\n"
        "from langchain_community.vectorstores import Chroma\n"
        "from langchain_core.prompts import ChatPromptTemplate\n"
        "from langchain_core.runnables import RunnablePassthrough\n"
        "from langchain_core.output_parsers import StrOutputParser\n\n"
        "# Setup retriever\nembeddings = OpenAIEmbeddings()\nvectorstore = Chroma(embedding_function=embeddings, persist_directory='./db')\n"
        "retriever = vectorstore.as_retriever(search_kwargs={'k': 4})\n\n"
        "# RAG prompt\ntemplate = '''Answer based on the following context:\n{context}\n\nQuestion: {question}'''\n"
        "prompt = ChatPromptTemplate.from_template(template)\n\n"
        "# RAG chain with LCEL\ndef format_docs(docs):\n    return '\\n\\n'.join(d.page_content for d in docs)\n\n"
        "rag_chain = (\n    {'context': retriever | format_docs, 'question': RunnablePassthrough()}\n"
        "    | prompt\n    | ChatOpenAI(model='gpt-4o')\n    | StrOutputParser()\n)\n\n"
        "answer = rag_chain.invoke('What is PagedAttention?')\n```"
    ))
    pairs.append(msg(
        "What is the difference between LangChain chains and agents?",
        "**Chains** (deterministic pipeline):\n"
        "- Fixed sequence of steps: prompt → LLM → parser\n"
        "- No decision-making — always follows the same path\n"
        "- Fast, predictable, easy to debug\n"
        "- Use for: RAG, summarization, translation, formatting\n\n"
        "**Agents** (autonomous decision-maker):\n"
        "- LLM decides which tools to call and in what order\n"
        "- Iterative: observe tool results → decide next action\n"
        "- Can handle multi-step reasoning and dynamic workflows\n"
        "- Use for: research, data analysis, multi-tool tasks\n\n"
        "**When to use which:**\n"
        "- Chain: you know the steps in advance\n"
        "- Agent: the steps depend on intermediate results\n"
        "- Chain + conditional routing: hybrid approach\n"
        "- LangGraph: for complex agent workflows with state machines\n\n"
        "Modern recommendation: use LangGraph for agent-like behavior (more control than AgentExecutor)."
    ))
    pairs.append(msg(
        "How do I use LangChain with a local Ollama model?",
        "```python\nfrom langchain_ollama import ChatOllama\nfrom langchain_core.prompts import ChatPromptTemplate\n"
        "from langchain_core.output_parsers import StrOutputParser\n\n"
        "# Connect to local Ollama\nllm = ChatOllama(\n    model='llama3.1:8b',\n    base_url='http://localhost:11434',\n"
        "    temperature=0.7,\n    num_ctx=4096\n)\n\n"
        "# Simple chain\nchain = ChatPromptTemplate.from_template('Explain {topic} simply.') | llm | StrOutputParser()\n"
        "result = chain.invoke({'topic': 'quantum computing'})\n\n"
        "# With structured output\nfrom pydantic import BaseModel\nclass Answer(BaseModel):\n"
        "    summary: str\n    confidence: float\n\n"
        "structured_llm = llm.with_structured_output(Answer)\nresult = structured_llm.invoke('What is Python?')\n```\n\n"
        "Install: `pip install langchain-ollama`. Works with any Ollama model including vision models."
    ))
    pairs.append(msg(
        "How do I use LangGraph for a stateful agent workflow?",
        "```python\nfrom langgraph.graph import StateGraph, MessagesState, START, END\n"
        "from langchain_openai import ChatOpenAI\nfrom langchain_core.tools import tool\n\n"
        "@tool\ndef search(query: str) -> str:\n    \"\"\"Search for information.\"\"\"\n    return f'Found: {query} results'\n\n"
        "model = ChatOpenAI(model='gpt-4o').bind_tools([search])\n\n"
        "def call_model(state: MessagesState):\n"
        "    response = model.invoke(state['messages'])\n    return {'messages': [response]}\n\n"
        "def call_tools(state: MessagesState):\n"
        "    last = state['messages'][-1]\n    results = []\n"
        "    for tc in last.tool_calls:\n        result = search.invoke(tc['args'])\n"
        "        results.append(ToolMessage(content=result, tool_call_id=tc['id']))\n"
        "    return {'messages': results}\n\n"
        "def should_continue(state: MessagesState):\n"
        "    if state['messages'][-1].tool_calls:\n        return 'tools'\n    return END\n\n"
        "graph = StateGraph(MessagesState)\n"
        "graph.add_node('agent', call_model)\ngraph.add_node('tools', call_tools)\n"
        "graph.add_edge(START, 'agent')\ngraph.add_conditional_edges('agent', should_continue)\n"
        "graph.add_edge('tools', 'agent')\n\napp = graph.compile()\nresult = app.invoke({'messages': [('user', 'Search for AI news')]})\n```"
    ))

    # More LangChain patterns
    pairs.append(msg(
        "How do I add fallback and retry logic to a LangChain chain?",
        "LCEL supports built-in fallback and retry:\n\n"
        "```python\nfrom langchain_openai import ChatOpenAI\nfrom langchain_anthropic import ChatAnthropic\n\n"
        "# Fallback: try GPT-4o, fall back to Claude\nprimary = ChatOpenAI(model='gpt-4o')\n"
        "fallback = ChatAnthropic(model='claude-sonnet-4-20250514')\n\n"
        "model_with_fallback = primary.with_fallbacks([fallback])\n\n"
        "# Retry with exponential backoff\nmodel_with_retry = primary.with_retry(\n"
        "    stop_after_attempt=3,\n    wait_exponential_jitter=True\n)\n\n"
        "# Combine both\nrobust_model = primary.with_retry(stop_after_attempt=2).with_fallbacks(\n"
        "    [fallback.with_retry(stop_after_attempt=2)]\n)\n\n"
        "# In a chain\nchain = prompt | robust_model | parser\n```\n\n"
        "The fallback triggers on any exception from the primary. "
        "You can also filter specific exceptions with `exceptions_to_handle`."
    ))
    pairs.append(msg(
        "How do I do structured output parsing in LangChain?",
        "Multiple approaches:\n\n"
        "**1. Pydantic model (recommended):**\n"
        "```python\nfrom pydantic import BaseModel, Field\nfrom langchain_openai import ChatOpenAI\n\n"
        "class MovieReview(BaseModel):\n    title: str = Field(description='Movie title')\n"
        "    rating: float = Field(ge=0, le=10)\n    summary: str\n\n"
        "llm = ChatOpenAI(model='gpt-4o')\nstructured = llm.with_structured_output(MovieReview)\n"
        "result = structured.invoke('Review The Matrix')  # Returns MovieReview instance\n```\n\n"
        "**2. JSON output parser:**\n"
        "```python\nfrom langchain_core.output_parsers import JsonOutputParser\nparser = JsonOutputParser(pydantic_object=MovieReview)\n"
        "chain = prompt | llm | parser\n```\n\n"
        "**3. XML output parser** (for Claude):\n"
        "```python\nfrom langchain_core.output_parsers import XMLOutputParser\nparser = XMLOutputParser(tags=['title', 'rating', 'summary'])\n```\n\n"
        "Pydantic + `with_structured_output` is the most reliable — it uses function calling under the hood."
    ))
    pairs.append(msg(
        "How do I use LangChain document loaders?",
        "LangChain has 100+ document loaders:\n\n"
        "```python\n# PDF\nfrom langchain_community.document_loaders import PyPDFLoader\ndocs = PyPDFLoader('paper.pdf').load()\n\n"
        "# Web page\nfrom langchain_community.document_loaders import WebBaseLoader\ndocs = WebBaseLoader('https://example.com').load()\n\n"
        "# Directory of files\nfrom langchain_community.document_loaders import DirectoryLoader\n"
        "docs = DirectoryLoader('./docs/', glob='**/*.md').load()\n\n"
        "# CSV\nfrom langchain_community.document_loaders import CSVLoader\ndocs = CSVLoader('data.csv').load()\n\n"
        "# GitHub repo\nfrom langchain_community.document_loaders import GithubFileLoader\n"
        "docs = GithubFileLoader(repo='user/repo', access_token='...').load()\n```\n\n"
        "Each loader returns `Document` objects with `page_content` (text) and `metadata` (source, page number, etc.). "
        "Split documents with text splitters before embedding."
    ))
    pairs.append(msg(
        "How do I split documents for RAG with LangChain?",
        "```python\nfrom langchain_text_splitters import (\n    RecursiveCharacterTextSplitter,\n"
        "    TokenTextSplitter,\n    MarkdownHeaderTextSplitter,\n)\n\n"
        "# Recursive character splitter (most common)\nsplitter = RecursiveCharacterTextSplitter(\n"
        "    chunk_size=1000,\n    chunk_overlap=200,\n    separators=['\\n\\n', '\\n', '. ', ' ', '']\n)\n"
        "chunks = splitter.split_documents(docs)\n\n"
        "# Token-based splitter\nsplitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)\n\n"
        "# Markdown-aware splitter\nmd_splitter = MarkdownHeaderTextSplitter(\n"
        "    headers_to_split_on=[('#', 'h1'), ('##', 'h2'), ('###', 'h3')]\n)\n\n"
        "# Code splitter\nfrom langchain_text_splitters import Language\n"
        "code_splitter = RecursiveCharacterTextSplitter.from_language(\n"
        "    language=Language.PYTHON, chunk_size=1000, chunk_overlap=100\n)\n```\n\n"
        "Guidelines: 500-1000 chars for general text, 10-20% overlap, use semantic boundaries."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 2. LlamaIndex
# ---------------------------------------------------------------------------
def gen_llamaindex() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do I build a basic RAG pipeline with LlamaIndex?",
        "```python\nfrom llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings\n"
        "from llama_index.llms.openai import OpenAI\nfrom llama_index.embeddings.openai import OpenAIEmbedding\n\n"
        "# Configure\nSettings.llm = OpenAI(model='gpt-4o', temperature=0.1)\n"
        "Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')\n\n"
        "# Load documents\ndocuments = SimpleDirectoryReader('./data/').load_data()\n\n"
        "# Create index\nindex = VectorStoreIndex.from_documents(documents)\n\n"
        "# Query\nquery_engine = index.as_query_engine(similarity_top_k=5)\nresponse = query_engine.query('What is PagedAttention?')\n"
        "print(response)\n```\n\n"
        "LlamaIndex handles chunking, embedding, indexing, retrieval, and response synthesis automatically. "
        "For production, persist the index: `index.storage_context.persist(persist_dir='./storage')`"
    ))
    pairs.append(msg(
        "What are the different index types in LlamaIndex?",
        "LlamaIndex offers several index types:\n\n"
        "**1. VectorStoreIndex** (most common):\n"
        "- Embeds chunks, retrieves by semantic similarity\n"
        "- Best for: general Q&A, search\n\n"
        "**2. SummaryIndex** (formerly ListIndex):\n"
        "- Stores documents in a list, summarizes all at query time\n"
        "- Best for: summarization tasks\n\n"
        "**3. TreeIndex**:\n"
        "- Hierarchical tree of summaries\n"
        "- Best for: multi-level summarization\n\n"
        "**4. KeywordTableIndex**:\n"
        "- Keyword extraction + lookup\n"
        "- Best for: keyword-based retrieval\n\n"
        "**5. KnowledgeGraphIndex**:\n"
        "- Extracts entities and relations into a graph\n"
        "- Best for: entity-centric Q&A\n\n"
        "**6. ComposableGraph**:\n"
        "- Combines multiple indices with routing\n"
        "- Best for: multi-document/multi-source RAG\n\n"
        "Most projects start with VectorStoreIndex and add complexity as needed."
    ))
    pairs.append(msg(
        "How do I use LlamaIndex with a custom vector store like Qdrant?",
        "```python\nfrom llama_index.core import VectorStoreIndex, StorageContext\n"
        "from llama_index.vector_stores.qdrant import QdrantVectorStore\nimport qdrant_client\n\n"
        "# Connect to Qdrant\nclient = qdrant_client.QdrantClient(url='http://localhost:6333')\n"
        "vector_store = QdrantVectorStore(client=client, collection_name='my_docs')\n\n"
        "# Create index with Qdrant backend\nstorage_context = StorageContext.from_defaults(vector_store=vector_store)\n"
        "index = VectorStoreIndex.from_documents(\n    documents, storage_context=storage_context\n)\n\n"
        "# Query (retrieval uses Qdrant)\nquery_engine = index.as_query_engine(similarity_top_k=5)\nresponse = query_engine.query('How does LoRA work?')\n```\n\n"
        "Supported vector stores: Qdrant, Chroma, Pinecone, Weaviate, Milvus, pgvector, FAISS, "
        "Redis, Elasticsearch. All use the same `VectorStoreIndex` API."
    ))
    pairs.append(msg(
        "How do I use LlamaIndex node parsers for custom chunking?",
        "```python\nfrom llama_index.core.node_parser import (\n    SentenceSplitter,\n"
        "    SentenceWindowNodeParser,\n    SemanticSplitterNodeParser,\n    MarkdownNodeParser,\n)\n\n"
        "# Sentence splitter (most common)\nparser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)\n\n"
        "# Sentence window (stores surrounding context)\nparser = SentenceWindowNodeParser.from_defaults(\n"
        "    window_size=3,  # 3 sentences on each side\n    original_text_metadata_key='original_text'\n)\n\n"
        "# Semantic splitter (splits by embedding similarity)\nfrom llama_index.embeddings.openai import OpenAIEmbedding\n"
        "parser = SemanticSplitterNodeParser(\n    embed_model=OpenAIEmbedding(),\n"
        "    breakpoint_percentile_threshold=95,\n    buffer_size=1\n)\n\n"
        "# Parse documents\nnodes = parser.get_nodes_from_documents(documents)\n"
        "index = VectorStoreIndex(nodes)\n```\n\n"
        "Semantic splitting produces variable-size chunks based on topic boundaries — "
        "better quality than fixed-size chunking but slower."
    ))
    pairs.append(msg(
        "How do I implement a chat engine with LlamaIndex?",
        "```python\nfrom llama_index.core import VectorStoreIndex\nfrom llama_index.core.chat_engine import CondensePlusContextChatEngine\n\n"
        "# Build index\nindex = VectorStoreIndex.from_documents(documents)\n\n"
        "# Chat engine with conversation memory\nchat_engine = index.as_chat_engine(\n"
        "    chat_mode='condense_plus_context',\n    similarity_top_k=5,\n"
        "    system_prompt='You are an expert assistant. Use the context to answer questions.'\n)\n\n"
        "# Multi-turn conversation\nresponse1 = chat_engine.chat('What is LoRA?')\nprint(response1)\n\n"
        "response2 = chat_engine.chat('How does the rank parameter affect it?')\nprint(response2)  # Uses conversation history\n\n"
        "# Reset conversation\nchat_engine.reset()\n```\n\n"
        "Chat modes:\n"
        "- `condense_plus_context`: condenses follow-up questions + retrieves context\n"
        "- `context`: simple context retrieval per turn\n"
        "- `react`: ReAct agent with tools\n"
        "- `best`: auto-selects based on query complexity"
    ))
    pairs.append(msg(
        "How do I use LlamaIndex with a local Ollama model?",
        "```python\nfrom llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings\n"
        "from llama_index.llms.ollama import Ollama\nfrom llama_index.embeddings.ollama import OllamaEmbedding\n\n"
        "# Use local Ollama models\nSettings.llm = Ollama(\n    model='llama3.1:8b',\n"
        "    base_url='http://localhost:11434',\n    request_timeout=120\n)\n"
        "Settings.embed_model = OllamaEmbedding(\n    model_name='nomic-embed-text',\n"
        "    base_url='http://localhost:11434'\n)\n\n"
        "# Build and query\ndocuments = SimpleDirectoryReader('./data/').load_data()\nindex = VectorStoreIndex.from_documents(documents)\n"
        "query_engine = index.as_query_engine()\nresponse = query_engine.query('Summarize the main points.')\n```\n\n"
        "Fully local RAG — no API keys needed. Install: `pip install llama-index-llms-ollama llama-index-embeddings-ollama`"
    ))
    pairs.append(msg(
        "How do I use LlamaIndex retrievers with reranking?",
        "```python\nfrom llama_index.core import VectorStoreIndex\n"
        "from llama_index.core.postprocessor import SentenceTransformerRerank\n\n"
        "# Build index\nindex = VectorStoreIndex.from_documents(documents)\n\n"
        "# Add reranker\nreranker = SentenceTransformerRerank(\n    model='cross-encoder/ms-marco-MiniLM-L-12-v2',\n    top_n=3\n)\n\n"
        "# Query with reranking\nquery_engine = index.as_query_engine(\n"
        "    similarity_top_k=10,  # Retrieve 10 candidates\n"
        "    node_postprocessors=[reranker]  # Rerank to top 3\n)\n"
        "response = query_engine.query('What is continuous batching?')\n```\n\n"
        "The two-stage retrieve-then-rerank pattern:\n"
        "1. Fast retrieval: embedding similarity gets top-K candidates (cheap, recall-focused)\n"
        "2. Reranking: cross-encoder scores relevance for each candidate (expensive, precision-focused)\n\n"
        "This consistently improves RAG quality by 10-20% on benchmarks."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 3. RAG Patterns
# ---------------------------------------------------------------------------
def gen_rag() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What are the main RAG patterns and when should I use each?",
        "RAG pattern taxonomy:\n\n"
        "**1. Naive RAG** (retrieve → read):\n"
        "- Simple embedding similarity retrieval + LLM answer\n"
        "- Use for: simple Q&A, low latency requirements\n\n"
        "**2. Advanced RAG** (pre/post-retrieval optimization):\n"
        "- Query rewriting, HyDE, reranking, metadata filtering\n"
        "- Use for: production systems needing higher accuracy\n\n"
        "**3. Modular RAG** (composable pipeline):\n"
        "- Pluggable retriever, reranker, reader components\n"
        "- Use for: domain-specific customization\n\n"
        "**4. Graph RAG** (knowledge graph + vector):\n"
        "- Entity/relation extraction → graph traversal + embedding retrieval\n"
        "- Use for: multi-hop reasoning, complex entity queries\n\n"
        "**5. RAPTOR** (tree-structured abstractions):\n"
        "- Hierarchical summaries at multiple granularities\n"
        "- Use for: long documents, thematic queries\n\n"
        "**6. Self-RAG** (adaptive retrieval):\n"
        "- Model decides when/if to retrieve, self-evaluates relevance\n"
        "- Use for: mixed queries (some need retrieval, some don't)"
    ))
    pairs.append(msg(
        "What is HyDE (Hypothetical Document Embeddings) in RAG?",
        "HyDE generates a hypothetical answer before retrieval:\n\n"
        "**Standard RAG**: embed user query → find similar documents\n"
        "**HyDE**: generate hypothetical answer → embed the answer → find similar documents\n\n"
        "Why it works: the hypothetical answer is in the same semantic space as stored documents "
        "(both are answers), while the query is in a different space (a question).\n\n"
        "```python\nfrom langchain.chains import HypotheticalDocumentEmbedder\nfrom langchain_openai import ChatOpenAI, OpenAIEmbeddings\n\n"
        "llm = ChatOpenAI(model='gpt-4o-mini')\nbase_embeddings = OpenAIEmbeddings()\n\n"
        "hyde = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, prompt_key='web_search')\n\n"
        "# HyDE generates a hypothetical answer, embeds it, then retrieves\nresult = hyde.embed_query('What causes aurora borealis?')\n```\n\n"
        "Tradeoff: adds one LLM call (latency + cost) but can significantly improve retrieval for "
        "queries that are semantically distant from the stored content."
    ))
    pairs.append(msg(
        "How does hybrid search work in RAG?",
        "Hybrid search combines dense (embedding) and sparse (keyword) retrieval:\n\n"
        "**Dense retrieval**: semantic similarity via embeddings\n- Good for: paraphrased queries, conceptual matching\n"
        "- Weak for: exact terms, acronyms, code identifiers\n\n"
        "**Sparse retrieval** (BM25/TF-IDF): keyword matching\n"
        "- Good for: exact terms, names, numbers\n"
        "- Weak for: semantic understanding, paraphrasing\n\n"
        "**Hybrid**: combine both with Reciprocal Rank Fusion (RRF):\n"
        "```python\ndef reciprocal_rank_fusion(results_list, k=60):\n"
        "    scores = {}\n    for results in results_list:\n"
        "        for rank, doc in enumerate(results):\n"
        "            scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)\n"
        "    return sorted(scores.items(), key=lambda x: -x[1])\n```\n\n"
        "Many vector DBs support hybrid natively:\n"
        "- Qdrant: `search(query_vector=..., query_text=...)`\n"
        "- Weaviate: `hybrid` search mode\n"
        "- Elasticsearch: `knn` + BM25 combined query"
    ))
    pairs.append(msg(
        "What is GraphRAG and how does it differ from standard RAG?",
        "GraphRAG combines knowledge graphs with vector retrieval:\n\n"
        "**Standard RAG**: document → chunks → embed → retrieve by similarity → answer\n"
        "**GraphRAG**: document → extract entities/relations → build knowledge graph → "
        "traverse graph + retrieve chunks → answer\n\n"
        "**Advantages over standard RAG:**\n"
        "1. Multi-hop reasoning: 'Who founded the company that acquired X?'\n"
        "2. Global summaries: 'What are the main themes across all documents?'\n"
        "3. Entity disambiguation: connect mentions of the same entity\n"
        "4. Relationship queries: 'How are A and B related?'\n\n"
        "**Microsoft GraphRAG pipeline:**\n"
        "1. Extract entities and relations via LLM\n"
        "2. Build community hierarchy (Leiden algorithm)\n"
        "3. Generate community summaries at multiple levels\n"
        "4. Local search: entity-centric retrieval\n"
        "5. Global search: community summary map-reduce\n\n"
        "**When to use**: multi-document corpora with rich entity relationships. "
        "Not worth the complexity for simple single-document Q&A."
    ))
    pairs.append(msg(
        "What is RAPTOR in RAG?",
        "RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builds a hierarchical "
        "tree of summaries:\n\n"
        "**Process:**\n"
        "1. Embed all text chunks\n"
        "2. Cluster similar chunks (UMAP + GMM)\n"
        "3. Summarize each cluster → new higher-level nodes\n"
        "4. Repeat: cluster summaries → summarize → even higher level\n"
        "5. Result: tree from leaf chunks to root summary\n\n"
        "**At query time:**\n"
        "- Collapse mode: retrieve from all tree levels\n"
        "- Tree traversal: start at root, navigate down relevant branches\n\n"
        "**When to use:**\n"
        "- Long documents (books, legal contracts, codebases)\n"
        "- Queries at different granularities (detail vs. overview)\n"
        "- Multi-document thematic analysis\n\n"
        "**Limitations:**\n"
        "- Expensive to build (many LLM calls for summarization)\n"
        "- Quality depends on clustering quality\n"
        "- Must rebuild tree when documents change"
    ))
    pairs.append(msg(
        "How do I evaluate RAG pipeline quality?",
        "Use the RAGAS framework:\n\n"
        "```python\nfrom ragas import evaluate\nfrom ragas.metrics import (\n"
        "    answer_relevancy,\n    faithfulness,\n    context_recall,\n    context_precision,\n)\n"
        "from datasets import Dataset\n\ndata = {\n    'question': ['What is LoRA?', ...],\n"
        "    'answer': ['LoRA is...', ...],\n    'contexts': [['LoRA paper excerpt...'], ...],\n"
        "    'ground_truth': ['Low-Rank Adaptation...', ...]\n}\n\n"
        "result = evaluate(\n    Dataset.from_dict(data),\n"
        "    metrics=[answer_relevancy, faithfulness, context_recall, context_precision]\n)\nprint(result)\n```\n\n"
        "Key metrics:\n"
        "- **Faithfulness**: is the answer grounded in retrieved context? (no hallucination)\n"
        "- **Answer Relevancy**: does the answer address the question?\n"
        "- **Context Precision**: are the retrieved documents relevant?\n"
        "- **Context Recall**: were all necessary documents retrieved?\n\n"
        "Target scores: >0.8 for production systems."
    ))
    pairs.append(msg(
        "What is query transformation in RAG?",
        "Query transformation rewrites the user query before retrieval:\n\n"
        "**1. Query Rewriting:**\n"
        "- LLM rephrases the query for better retrieval\n"
        "- 'How does that thing with the attention work?' → 'How does self-attention mechanism work in transformers?'\n\n"
        "**2. Query Decomposition:**\n"
        "- Split complex query into sub-queries\n"
        "- 'Compare LoRA and QLoRA for 70B models' → ['What is LoRA?', 'What is QLoRA?', 'Differences between LoRA and QLoRA for large models']\n\n"
        "**3. Step-Back Prompting:**\n"
        "- Generate a more general question first\n"
        "- 'What temperature should I use for Llama 3.1 coding tasks?' → 'How does temperature affect LLM code generation quality?'\n\n"
        "**4. HyDE** (Hypothetical Document Embeddings):\n"
        "- Generate a hypothetical answer, embed it for retrieval\n\n"
        "**5. Multi-Query:**\n"
        "- Generate N variations of the query, retrieve for each, merge results\n"
        "- Increases recall at the cost of latency\n\n"
        "Query transformation typically improves retrieval quality by 15-25%."
    ))
    pairs.append(msg(
        "How do I implement metadata filtering in RAG?",
        "Metadata filtering combines semantic search with structured filters:\n\n"
        "```python\n# LlamaIndex with metadata filtering\nfrom llama_index.core.vector_stores import (\n"
        "    MetadataFilter, MetadataFilters, FilterOperator\n)\n\nfilters = MetadataFilters(filters=[\n"
        "    MetadataFilter(key='category', value='technical', operator=FilterOperator.EQ),\n"
        "    MetadataFilter(key='date', value='2024-01-01', operator=FilterOperator.GTE),\n])\n\n"
        "query_engine = index.as_query_engine(\n    similarity_top_k=5,\n"
        "    filters=filters\n)\n```\n\n"
        "Common metadata fields:\n"
        "- `source`: document filename or URL\n"
        "- `date`: creation/modification date\n"
        "- `category`: document type or topic\n"
        "- `author`: document author\n"
        "- `page`: page number (for PDFs)\n"
        "- `language`: document language\n\n"
        "Auto-extract metadata with LLM:\n"
        "```python\nfrom llama_index.core.extractors import TitleExtractor, SummaryExtractor\n"
        "transformations = [SentenceSplitter(), TitleExtractor(), SummaryExtractor()]\n```"
    ))
    pairs.append(msg(
        "What is the difference between reranking and retrieval in RAG?",
        "Two-stage retrieval pipeline:\n\n"
        "**Stage 1 — Retrieval (recall-focused):**\n"
        "- Fast approximate search over entire corpus\n"
        "- Methods: embedding similarity, BM25, hybrid\n"
        "- Returns top-K candidates (K=10-100)\n"
        "- Optimized for: not missing relevant documents\n"
        "- Speed: milliseconds over millions of documents\n\n"
        "**Stage 2 — Reranking (precision-focused):**\n"
        "- Precise relevance scoring on K candidates\n"
        "- Methods: cross-encoder models, LLM-based scoring\n"
        "- Returns top-N (N=3-5) most relevant documents\n"
        "- Optimized for: ordering by true relevance\n"
        "- Speed: 10-100ms per candidate\n\n"
        "**Why separate stages?**\n"
        "Cross-encoders are too slow to run over the entire corpus (O(n) per query). "
        "Bi-encoders (embeddings) are fast (pre-computed) but less precise. "
        "Combining both gives the best of both worlds.\n\n"
        "Popular rerankers: Cohere Rerank, bge-reranker, ms-marco cross-encoders, Jina Reranker."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 4. Multi-agent Systems
# ---------------------------------------------------------------------------
def gen_multiagent() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What are the main multi-agent system architectures?",
        "Key multi-agent architectures:\n\n"
        "**1. Sequential Chain** (pipeline):\n"
        "- Agent A → Agent B → Agent C\n"
        "- Each agent handles one step, passes result to next\n"
        "- Example: Researcher → Writer → Editor\n\n"
        "**2. Hierarchical** (manager/worker):\n"
        "- Manager agent delegates tasks to specialist agents\n"
        "- Manager synthesizes results\n"
        "- Example: PM → [Frontend Dev, Backend Dev, QA]\n\n"
        "**3. Debate/Discussion:**\n"
        "- Multiple agents argue different positions\n"
        "- Converge through rounds of discussion\n"
        "- Use for: complex decisions, reducing bias\n\n"
        "**4. Mixture of Agents (MoA):**\n"
        "- Multiple agents generate answers independently\n"
        "- Aggregator synthesizes best answer\n"
        "- Like ensemble methods in ML\n\n"
        "**5. Swarm (OpenAI):**\n"
        "- Lightweight agent handoff based on function calling\n"
        "- Agents transfer control dynamically\n"
        "- Good for: customer service routing, triage"
    ))
    pairs.append(msg(
        "How do I build a multi-agent system with CrewAI?",
        "```python\nfrom crewai import Agent, Task, Crew, Process\nfrom crewai_tools import SerperDevTool\n\n"
        "# Define agents\nresearcher = Agent(\n    role='Senior Researcher',\n"
        "    goal='Find comprehensive information on the topic',\n"
        "    backstory='Expert at finding and analyzing data',\n"
        "    tools=[SerperDevTool()],\n    verbose=True\n)\n\n"
        "writer = Agent(\n    role='Technical Writer',\n"
        "    goal='Write clear, engaging technical content',\n"
        "    backstory='Experienced technical writer',\n    verbose=True\n)\n\n"
        "# Define tasks\nresearch_task = Task(\n    description='Research {topic} thoroughly',\n"
        "    expected_output='Comprehensive research summary with sources',\n    agent=researcher\n)\n\n"
        "write_task = Task(\n    description='Write a blog post based on the research',\n"
        "    expected_output='Well-structured blog post (500-800 words)',\n"
        "    agent=writer,\n    context=[research_task]  # Uses research output\n)\n\n"
        "# Create crew\ncrew = Crew(\n    agents=[researcher, writer],\n    tasks=[research_task, write_task],\n"
        "    process=Process.sequential,\n    verbose=True\n)\n\n"
        "result = crew.kickoff(inputs={'topic': 'LLM quantization methods'})\n```"
    ))
    pairs.append(msg(
        "What is Mixture of Agents (MoA) and how does it work?",
        "MoA uses multiple LLMs in a layered architecture:\n\n"
        "**Layer 1 (Proposers):** N different LLMs independently answer the query\n"
        "**Layer 2+ (Aggregators):** LLMs see previous layer outputs + original query, produce refined answers\n"
        "**Final:** Best answer selected or synthesized\n\n"
        "```python\ndef mixture_of_agents(query, proposers, aggregator, rounds=2):\n"
        "    # Layer 1: independent proposals\n    proposals = [llm.generate(query) for llm in proposers]\n\n"
        "    # Layer 2+: iterative aggregation\n    for _ in range(rounds):\n"
        "        context = '\\n---\\n'.join(proposals)\n"
        "        prompt = f'Given these responses:\\n{context}\\n\\nSynthesize the best answer to: {query}'\n"
        "        proposals = [aggregator.generate(prompt)]\n\n"
        "    return proposals[0]\n```\n\n"
        "Benefits:\n"
        "- Diversity of perspectives (different models have different strengths)\n"
        "- Self-correction through aggregation\n"
        "- Can outperform any single model\n\n"
        "Paper: 'Mixture-of-Agents Enhances Large Language Model Capabilities' (2024)"
    ))
    pairs.append(msg(
        "How does OpenAI Swarm work for agent handoffs?",
        "Swarm uses lightweight function-calling agents with explicit handoff:\n\n"
        "```python\nfrom swarm import Swarm, Agent\n\nclient = Swarm()\n\n"
        "def transfer_to_sales():\n    \"\"\"Transfer to the sales agent.\"\"\"\n    return sales_agent\n\n"
        "def transfer_to_support():\n    \"\"\"Transfer to the support agent.\"\"\"\n    return support_agent\n\n"
        "triage_agent = Agent(\n    name='Triage',\n"
        "    instructions='Route the user to the right department.',\n"
        "    functions=[transfer_to_sales, transfer_to_support]\n)\n\n"
        "sales_agent = Agent(\n    name='Sales',\n"
        "    instructions='Help users with purchasing and pricing.',\n)\n\n"
        "support_agent = Agent(\n    name='Support',\n"
        "    instructions='Help users with technical issues.',\n)\n\n"
        "response = client.run(\n    agent=triage_agent,\n"
        "    messages=[{'role': 'user', 'content': 'I need help with billing'}]\n)\n```\n\n"
        "Key concepts: agents are stateless functions. Handoff = returning another agent. "
        "Context variables are passed explicitly. No complex orchestration needed."
    ))
    pairs.append(msg(
        "How do I implement agent debate for better reasoning?",
        "Agent debate uses multiple perspectives to improve answer quality:\n\n"
        "```python\ndef agent_debate(question, models, rounds=3):\n"
        "    # Initial answers\n    answers = {}\n    for name, model in models.items():\n"
        "        answers[name] = model.generate(question)\n\n"
        "    # Debate rounds\n    for r in range(rounds):\n"
        "        new_answers = {}\n        for name, model in models.items():\n"
        "            other_views = [f'{k}: {v}' for k, v in answers.items() if k != name]\n"
        "            prompt = (\n"
        "                f'Question: {question}\\n\\n'\n"
        "                f'Your previous answer: {answers[name]}\\n\\n'\n"
        "                f'Other perspectives:\\n' + '\\n'.join(other_views) + '\\n\\n'\n"
        "                f'Considering these perspectives, provide your revised answer. '\n"
        "                f'Update your answer if convinced, or defend your position.'\n"
        "            )\n            new_answers[name] = model.generate(prompt)\n"
        "        answers = new_answers\n\n"
        "    # Final synthesis\n    return synthesize(question, answers)\n```\n\n"
        "Works best for: complex reasoning, fact verification, ethical considerations. "
        "Not worth the cost for simple factual questions."
    ))
    pairs.append(msg(
        "What is AutoGen and how does it differ from CrewAI?",
        "**AutoGen** (Microsoft):\n"
        "- Code-execution first: agents can write and run code\n"
        "- Conversation-based: agents talk to each other\n"
        "- Human-in-the-loop: human can participate in conversations\n"
        "- Group chat: multiple agents in one conversation\n"
        "- Good for: coding tasks, data analysis, research\n\n"
        "**CrewAI**:\n"
        "- Role-based: agents have roles, goals, backstories\n"
        "- Task-oriented: explicit task assignment\n"
        "- Process types: sequential, hierarchical, consensual\n"
        "- Tool integration: rich tool ecosystem\n"
        "- Good for: business workflows, content creation\n\n"
        "**Key differences:**\n"
        "| Aspect | AutoGen | CrewAI |\n"
        "|--------|---------|--------|\n"
        "| Paradigm | Conversation | Task execution |\n"
        "| Code exec | Built-in | Via tools |\n"
        "| Complexity | Higher | Simpler |\n"
        "| Flexibility | More | Less |\n"
        "| Learning curve | Steeper | Gentler |\n\n"
        "For most use cases, CrewAI is easier to start with. AutoGen is more powerful for coding-heavy workflows."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 5. Prompt Engineering
# ---------------------------------------------------------------------------
def gen_prompting() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What is chain-of-thought prompting and when should I use it?",
        "Chain-of-thought (CoT) prompting asks the model to reason step by step:\n\n"
        "**Zero-shot CoT:**\n```\nQ: If a car travels at 60 mph for 2.5 hours, how far does it go?\nA: Let's think step by step.\n```\n\n"
        "**Few-shot CoT:**\n```\nQ: Roger has 5 tennis balls. He buys 2 cans of 3. How many now?\n"
        "A: Roger started with 5 balls. 2 cans × 3 balls = 6 new balls. 5 + 6 = 11.\nThe answer is 11.\n\n"
        "Q: The cafeteria had 23 apples. They used 20 and bought 6 more. How many now?\nA:\n```\n\n"
        "**When to use CoT:**\n"
        "- Math and arithmetic problems\n"
        "- Multi-step logical reasoning\n"
        "- Complex decision-making\n"
        "- Tasks where showing work improves accuracy\n\n"
        "**When NOT to use:**\n"
        "- Simple factual questions (adds unnecessary tokens)\n"
        "- Creative writing\n"
        "- Classification tasks (unless borderline cases)\n\n"
        "CoT improves accuracy by 10-40% on reasoning tasks for models ≥7B parameters."
    ))
    pairs.append(msg(
        "What is the ReAct prompting pattern?",
        "ReAct (Reasoning + Acting) interleaves thinking with tool use:\n\n"
        "```\nQuestion: What is the population of the capital of France?\n\n"
        "Thought: I need to find the capital of France first.\n"
        "Action: search(\"capital of France\")\nObservation: The capital of France is Paris.\n\n"
        "Thought: Now I need to find the population of Paris.\n"
        "Action: search(\"population of Paris 2024\")\nObservation: Paris has a population of about 2.1 million.\n\n"
        "Thought: I now have the answer.\nAnswer: The population of Paris, the capital of France, is approximately 2.1 million.\n```\n\n"
        "ReAct format:\n"
        "1. **Thought**: reasoning about what to do next\n"
        "2. **Action**: tool call with arguments\n"
        "3. **Observation**: tool result\n"
        "4. Repeat until ready to answer\n\n"
        "Used by: LangChain agents, LlamaIndex ReAct agents, Claude tool use. "
        "More reliable than pure chain-of-thought because grounded in real data."
    ))
    pairs.append(msg(
        "How do I write effective few-shot prompts?",
        "Few-shot prompting principles:\n\n"
        "**1. Choose representative examples:**\n"
        "- Cover different categories/edge cases\n"
        "- Include both easy and hard examples\n"
        "- Match the distribution of expected inputs\n\n"
        "**2. Format consistently:**\n"
        "```\nInput: The movie was absolutely terrible and boring.\nSentiment: Negative\n\n"
        "Input: I loved every minute of this film!\nSentiment: Positive\n\n"
        "Input: It was okay, nothing special.\nSentiment: Neutral\n\n"
        "Input: {user_input}\nSentiment:\n```\n\n"
        "**3. Order matters:**\n"
        "- Put most relevant examples last (recency bias)\n"
        "- For classification, alternate categories\n"
        "- 3-5 examples is usually sufficient\n\n"
        "**4. Include edge cases:**\n"
        "- Ambiguous inputs with correct labels\n"
        "- Examples that demonstrate reasoning\n\n"
        "**5. Dynamic few-shot:**\n"
        "- Select examples based on similarity to the input\n"
        "- Use embedding similarity to pick the most relevant examples from a pool"
    ))
    pairs.append(msg(
        "What is tree-of-thought prompting?",
        "Tree-of-Thought (ToT) explores multiple reasoning paths:\n\n"
        "**Chain-of-Thought**: single linear path → answer\n"
        "**Tree-of-Thought**: multiple branching paths → evaluate → select best\n\n"
        "```\nProblem: Solve 24 game with numbers 1, 5, 5, 5\n\n"
        "Branch 1: 5 × 5 = 25 → 25 - 1 = 24 ✓\n"
        "Branch 2: 5 + 5 = 10 → 10 × 1 = 10 → stuck ✗\n"
        "Branch 3: 5 × (5 - 1) = 20 → 20 + ? → stuck ✗\n\n"
        "Best path: Branch 1 → 5 × 5 - 1 = 24\n```\n\n"
        "**Implementation approaches:**\n"
        "1. **Breadth-first**: generate all branches at each step, evaluate, prune\n"
        "2. **Depth-first**: explore one path fully, backtrack if stuck\n"
        "3. **Best-first**: score partial paths, expand most promising\n\n"
        "**When to use:**\n"
        "- Combinatorial problems (puzzles, planning)\n"
        "- Tasks where the first attempt often fails\n"
        "- When you can evaluate partial solutions\n\n"
        "Cost: 3-10x more tokens than CoT. Worth it for hard problems."
    ))
    pairs.append(msg(
        "What are effective system prompt patterns for production LLMs?",
        "Production system prompt structure:\n\n"
        "```\n## Role\nYou are [specific role] with expertise in [domain].\n\n"
        "## Instructions\n1. [Primary behavior]\n2. [Secondary behavior]\n3. [Constraints]\n\n"
        "## Output Format\n- Format: [JSON/Markdown/etc.]\n- Length: [constraints]\n- Style: [tone]\n\n"
        "## Examples\n[Few-shot examples]\n\n"
        "## Guardrails\n- Do NOT [prohibited behavior]\n- If uncertain, [fallback behavior]\n- Always [safety requirement]\n```\n\n"
        "Best practices:\n"
        "- Be specific about the role and expertise level\n"
        "- Explicitly define output format (reduces parsing errors)\n"
        "- Include negative instructions (what NOT to do)\n"
        "- Add fallback instructions for edge cases\n"
        "- Keep system prompt under 500 tokens for efficiency\n"
        "- Use XML tags or markdown headers for structure\n"
        "- Test with adversarial inputs"
    ))
    pairs.append(msg(
        "How do I use self-consistency prompting?",
        "Self-consistency samples multiple reasoning paths and takes majority vote:\n\n"
        "```python\ndef self_consistency(question, model, n_samples=5, temperature=0.7):\n"
        "    answers = []\n    for _ in range(n_samples):\n"
        "        response = model.generate(\n"
        "            f'{question}\\nLet\\'s think step by step.',\n"
        "            temperature=temperature  # Higher temp for diversity\n"
        "        )\n        answer = extract_final_answer(response)\n"
        "        answers.append(answer)\n\n"
        "    # Majority vote\n    from collections import Counter\n"
        "    most_common = Counter(answers).most_common(1)[0][0]\n    return most_common\n```\n\n"
        "Why it works:\n"
        "- Different reasoning paths may reach the same correct answer\n"
        "- Wrong answers are usually diverse (don't agree)\n"
        "- Correct answers tend to converge\n\n"
        "Improvements over single CoT: 5-15% accuracy gain on math/reasoning benchmarks. "
        "Cost: N× more tokens. Use temperature 0.5-0.8 for path diversity."
    ))
    pairs.append(msg(
        "What is prompt chaining and when should I use it?",
        "Prompt chaining decomposes complex tasks into sequential LLM calls:\n\n"
        "```\nStep 1: Extract key entities from the document\n        → {entities: ['LoRA', 'QLoRA', 'PEFT']}\n\n"
        "Step 2: For each entity, generate a definition\n        → {definitions: {LoRA: '...', QLoRA: '...'}}\n\n"
        "Step 3: Synthesize into a glossary with examples\n        → Final formatted glossary\n```\n\n"
        "**Advantages over single prompt:**\n"
        "- Each step is simpler → more reliable\n"
        "- Can validate intermediate results\n"
        "- Can use different models per step (cheap for extraction, expensive for synthesis)\n"
        "- Easier to debug (isolate which step fails)\n\n"
        "**When to use:**\n"
        "- Tasks with distinct phases (extract → transform → generate)\n"
        "- When intermediate validation is needed\n"
        "- When different steps need different capabilities\n"
        "- When a single prompt consistently fails\n\n"
        "**When NOT to use:**\n"
        "- Simple single-step tasks (adds unnecessary latency)\n"
        "- Real-time applications (cumulative latency)"
    ))
    pairs.append(msg(
        "How do I handle long context windows effectively?",
        "Strategies for using large context windows (100K+ tokens):\n\n"
        "**1. Place important information at the beginning and end:**\n"
        "- LLMs attend best to the start and end of context (lost-in-the-middle effect)\n"
        "- Put instructions at the start, key data at the end\n\n"
        "**2. Use structured formatting:**\n"
        "```xml\n<document id=\"1\" source=\"paper.pdf\">\nContent here...\n</document>\n"
        "<document id=\"2\" source=\"notes.md\">\nContent here...\n</document>\n```\n\n"
        "**3. Add explicit references:**\n"
        "- 'Refer to Document 1, Section 3 for the answer'\n"
        "- Numbered paragraphs or sections\n\n"
        "**4. Chunk strategically:**\n"
        "- Don't dump everything — select relevant sections\n"
        "- Long context ≠ better quality (dilution effect)\n\n"
        "**5. Use retrieval even with long context:**\n"
        "- Pre-filter documents, then use long context for the filtered set\n"
        "- Better than stuffing everything into context"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 6. Function Calling / Tool Use
# ---------------------------------------------------------------------------
def gen_function_calling() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How does OpenAI function calling work?",
        "OpenAI function calling lets the model invoke structured functions:\n\n"
        "```python\nfrom openai import OpenAI\nclient = OpenAI()\n\ntools = [{\n"
        "    'type': 'function',\n    'function': {\n"
        "        'name': 'get_weather',\n        'description': 'Get current weather for a city',\n"
        "        'parameters': {\n            'type': 'object',\n"
        "            'properties': {\n"
        "                'city': {'type': 'string', 'description': 'City name'},\n"
        "                'units': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}\n"
        "            },\n            'required': ['city']\n        }\n    }\n}]\n\n"
        "response = client.chat.completions.create(\n    model='gpt-4o',\n"
        "    messages=[{'role': 'user', 'content': 'Weather in Paris?'}],\n    tools=tools\n)\n\n"
        "# Check if model wants to call a function\ntool_call = response.choices[0].message.tool_calls[0]\n"
        "print(tool_call.function.name)  # 'get_weather'\nprint(tool_call.function.arguments)  # '{\"city\": \"Paris\"}'\n```\n\n"
        "The model returns a tool call — your code executes the function and sends the result back."
    ))
    pairs.append(msg(
        "How does Claude tool use differ from OpenAI function calling?",
        "Claude tool use syntax:\n\n"
        "```python\nimport anthropic\nclient = anthropic.Anthropic()\n\ntools = [{\n"
        "    'name': 'get_weather',\n    'description': 'Get current weather for a city',\n"
        "    'input_schema': {\n        'type': 'object',\n"
        "        'properties': {\n"
        "            'city': {'type': 'string', 'description': 'City name'},\n"
        "            'units': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}\n"
        "        },\n        'required': ['city']\n    }\n}]\n\n"
        "response = client.messages.create(\n    model='claude-sonnet-4-20250514',\n"
        "    messages=[{'role': 'user', 'content': 'Weather in Paris?'}],\n"
        "    tools=tools, max_tokens=1024\n)\n\n"
        "# Process tool use blocks\nfor block in response.content:\n"
        "    if block.type == 'tool_use':\n        print(block.name, block.input)\n```\n\n"
        "Key differences from OpenAI:\n"
        "- `input_schema` instead of `parameters`\n"
        "- Response uses content blocks (can mix text + tool_use)\n"
        "- `tool_choice` options: `auto`, `any`, `{type: 'tool', name: '...'}`\n"
        "- Claude tends to be more cautious about tool calls"
    ))
    pairs.append(msg(
        "How do I handle parallel tool calls from LLMs?",
        "Both OpenAI and Claude can request multiple tool calls simultaneously:\n\n"
        "```python\n# OpenAI parallel tool calls\nresponse = client.chat.completions.create(\n"
        "    model='gpt-4o',\n    messages=[{'role': 'user', 'content': 'Weather in Paris and Tokyo'}],\n"
        "    tools=tools, parallel_tool_calls=True\n)\n\n"
        "# Process all tool calls\ntool_calls = response.choices[0].message.tool_calls\nresults = []\n"
        "for tc in tool_calls:\n    result = execute_function(tc.function.name, json.loads(tc.function.arguments))\n"
        "    results.append({'role': 'tool', 'tool_call_id': tc.id, 'content': json.dumps(result)})\n\n"
        "# Send results back\nmessages.append(response.choices[0].message)\nmessages.extend(results)\n"
        "final = client.chat.completions.create(model='gpt-4o', messages=messages, tools=tools)\n```\n\n"
        "Tips:\n"
        "- Execute parallel calls concurrently (`asyncio.gather`)\n"
        "- Each tool result must reference its `tool_call_id`\n"
        "- Set `parallel_tool_calls=False` if tools have side effects or ordering matters"
    ))
    pairs.append(msg(
        "How do I implement tool use with Mistral models?",
        "Mistral function calling format:\n\n"
        "```python\nfrom mistralai import Mistral\nclient = Mistral(api_key='...')\n\n"
        "tools = [{\n    'type': 'function',\n    'function': {\n"
        "        'name': 'get_weather',\n        'description': 'Get weather for a city',\n"
        "        'parameters': {\n            'type': 'object',\n"
        "            'properties': {'city': {'type': 'string'}},\n"
        "            'required': ['city']\n        }\n    }\n}]\n\n"
        "response = client.chat.complete(\n    model='mistral-large-latest',\n"
        "    messages=[{'role': 'user', 'content': 'Weather in Paris?'}],\n    tools=tools\n)\n\n"
        "# Process tool calls\nfor tc in response.choices[0].message.tool_calls:\n"
        "    print(tc.function.name, tc.function.arguments)\n```\n\n"
        "Mistral follows OpenAI's tool format closely. Differences:\n"
        "- Uses `client.chat.complete()` (not `completions.create()`)\n"
        "- Tool call arguments may be returned as string or dict\n"
        "- Mistral Large and Codestral support parallel tool calls"
    ))
    pairs.append(msg(
        "What are best practices for designing tool schemas for LLMs?",
        "Tool schema design principles:\n\n"
        "**1. Clear descriptions:**\n"
        "```json\n{\"name\": \"search_database\",\n \"description\": \"Search the product database. Returns matching products with price and availability. Use for product lookup queries.\",\n"
        " \"parameters\": {...}}\n```\n\n"
        "**2. Specific parameter descriptions:**\n"
        "```json\n{\"city\": {\"type\": \"string\", \"description\": \"Full city name, e.g., 'San Francisco' not 'SF'\"}}\n```\n\n"
        "**3. Use enums for constrained values:**\n"
        "```json\n{\"units\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}}\n```\n\n"
        "**4. Limit parameter count:** 3-5 parameters max per tool\n\n"
        "**5. Explicit required fields:** mark required vs optional\n\n"
        "**6. Give usage guidance in description:**\n"
        "- When to use this tool vs alternatives\n"
        "- What inputs produce best results\n"
        "- Expected output format\n\n"
        "**7. Avoid ambiguous tools:** if two tools overlap, the model struggles to choose"
    ))
    pairs.append(msg(
        "How do I implement a tool execution loop for autonomous agents?",
        "Standard agentic tool loop:\n\n"
        "```python\nasync def agent_loop(query, tools, model, max_iterations=10):\n"
        "    messages = [{'role': 'user', 'content': query}]\n\n"
        "    for i in range(max_iterations):\n"
        "        response = await model.chat(messages=messages, tools=tools)\n\n"
        "        # Check if model wants to use tools\n"
        "        if not response.tool_calls:\n"
        "            return response.content  # Final answer\n\n"
        "        # Execute tool calls\n"
        "        messages.append(response.message)  # Assistant message with tool_calls\n\n"
        "        for tc in response.tool_calls:\n"
        "            try:\n"
        "                result = await execute_tool(tc.name, tc.arguments)\n"
        "                messages.append({\n"
        "                    'role': 'tool',\n"
        "                    'tool_call_id': tc.id,\n"
        "                    'content': json.dumps(result)\n"
        "                })\n"
        "            except Exception as e:\n"
        "                messages.append({\n"
        "                    'role': 'tool',\n"
        "                    'tool_call_id': tc.id,\n"
        "                    'content': f'Error: {e}'\n"
        "                })\n\n"
        "    return 'Max iterations reached.'\n```\n\n"
        "Important: always set max_iterations to prevent infinite loops. "
        "Send errors back as tool results (not exceptions) so the model can recover."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 7. MCP Protocol
# ---------------------------------------------------------------------------
def gen_mcp() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What is MCP (Model Context Protocol) and what problem does it solve?",
        "MCP (Model Context Protocol) is an open standard for connecting AI models to external data and tools.\n\n"
        "**Problem**: Every AI application needs to connect to different data sources (databases, APIs, files) "
        "and tools. Without a standard, each integration is custom — N models × M tools = N×M integrations.\n\n"
        "**Solution**: MCP provides a universal protocol:\n"
        "- **MCP Server**: exposes tools, resources, and prompts via a standard interface\n"
        "- **MCP Client**: AI application that connects to servers\n"
        "- **MCP Host**: the application managing client connections (e.g., Claude Desktop)\n\n"
        "**Three primitives:**\n"
        "1. **Tools**: functions the model can call (e.g., `search_database`, `run_query`)\n"
        "2. **Resources**: data the model can read (e.g., file contents, database schemas)\n"
        "3. **Prompts**: reusable prompt templates with parameters\n\n"
        "MCP reduces N×M to N+M integrations. Supported by: Claude Desktop, Claude Code, "
        "Cursor, Zed, Cody, and many others."
    ))
    pairs.append(msg(
        "How do I build an MCP server in Python?",
        "```python\nfrom mcp.server.fastmcp import FastMCP\n\nmcp = FastMCP('My Server')\n\n"
        "@mcp.tool()\ndef search_documents(query: str, limit: int = 5) -> str:\n"
        "    \"\"\"Search the document database.\n\n    Args:\n        query: Search query string\n"
        "        limit: Maximum number of results (default 5)\n    \"\"\"\n"
        "    results = db.search(query, limit=limit)\n    return json.dumps(results)\n\n"
        "@mcp.resource('docs://schema')\ndef get_schema() -> str:\n"
        "    \"\"\"Get the database schema.\"\"\"\n    return json.dumps(db.get_schema())\n\n"
        "@mcp.prompt()\ndef analyze_data(dataset: str) -> str:\n"
        "    \"\"\"Create a prompt for data analysis.\"\"\"\n"
        "    return f'Analyze the following dataset and provide insights:\\n\\n{dataset}'\n\n"
        "if __name__ == '__main__':\n    mcp.run(transport='stdio')  # or 'sse' for HTTP\n```\n\n"
        "Install: `pip install mcp[cli]`\n"
        "Test: `mcp dev server.py`\n"
        "The server communicates via stdio (default) or SSE (HTTP)."
    ))
    pairs.append(msg(
        "How do I build an MCP server in TypeScript?",
        "```typescript\nimport { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';\n"
        "import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';\n"
        "import { z } from 'zod';\n\nconst server = new McpServer({\n  name: 'My Server',\n  version: '1.0.0'\n});\n\n"
        "// Add a tool\nserver.tool(\n  'search_documents',\n  'Search the document database',\n"
        "  { query: z.string(), limit: z.number().default(5) },\n"
        "  async ({ query, limit }) => {\n    const results = await db.search(query, limit);\n"
        "    return { content: [{ type: 'text', text: JSON.stringify(results) }] };\n  }\n);\n\n"
        "// Add a resource\nserver.resource(\n  'docs://schema',\n  'Database schema',\n"
        "  async () => ({ contents: [{ uri: 'docs://schema', text: JSON.stringify(schema) }] })\n);\n\n"
        "// Start server\nconst transport = new StdioServerTransport();\nawait server.connect(transport);\n```\n\n"
        "Install: `npm install @modelcontextprotocol/sdk`"
    ))
    pairs.append(msg(
        "How do I configure MCP servers in Claude Desktop?",
        "Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:\n\n"
        "```json\n{\n  \"mcpServers\": {\n"
        "    \"filesystem\": {\n      \"command\": \"npx\",\n"
        "      \"args\": [\"-y\", \"@modelcontextprotocol/server-filesystem\", \"/Users/me/docs\"]\n"
        "    },\n"
        "    \"my-custom-server\": {\n      \"command\": \"python\",\n"
        "      \"args\": [\"path/to/server.py\"],\n"
        "      \"env\": {\"API_KEY\": \"sk-...\"}\n"
        "    },\n"
        "    \"remote-server\": {\n      \"command\": \"npx\",\n"
        "      \"args\": [\"-y\", \"@anthropic/mcp-proxy\", \"https://api.example.com/mcp\"]\n"
        "    }\n  }\n}\n```\n\n"
        "After editing, restart Claude Desktop. Each server entry specifies:\n"
        "- `command`: executable to run\n"
        "- `args`: command-line arguments\n"
        "- `env`: (optional) environment variables\n\n"
        "MCP servers run as child processes communicating via stdio."
    ))
    pairs.append(msg(
        "What MCP transports are available?",
        "MCP supports two transport mechanisms:\n\n"
        "**1. stdio (Standard I/O):**\n"
        "- Server runs as a subprocess of the client\n"
        "- Communication via stdin/stdout\n"
        "- Best for: local integrations, desktop apps\n"
        "- Default transport for Claude Desktop\n"
        "```python\nmcp.run(transport='stdio')\n```\n\n"
        "**2. SSE (Server-Sent Events over HTTP):**\n"
        "- Server runs as an HTTP service\n"
        "- Client connects via HTTP\n"
        "- Best for: remote servers, shared services\n"
        "```python\nmcp.run(transport='sse', port=3001)\n```\n\n"
        "**3. Streamable HTTP (new, 2025+):**\n"
        "- Bidirectional streaming over HTTP\n"
        "- Replaces SSE for new implementations\n"
        "- Better support for stateless deployments\n\n"
        "Security considerations:\n"
        "- stdio: inherits process permissions\n"
        "- SSE/HTTP: need authentication, CORS, TLS\n"
        "- Always validate tool inputs on the server side"
    ))
    pairs.append(msg(
        "What are MCP resources and how do they differ from tools?",
        "**Resources** = data the model can read (passive)\n"
        "**Tools** = functions the model can execute (active)\n\n"
        "**Resources:**\n"
        "- Identified by URI (e.g., `file:///path`, `db://table/schema`)\n"
        "- Read-only data retrieval\n"
        "- Client can list and read resources\n"
        "- Used for: file contents, database schemas, config files, documentation\n"
        "```python\n@mcp.resource('config://settings')\ndef get_settings() -> str:\n"
        "    return json.dumps(load_settings())\n```\n\n"
        "**Tools:**\n"
        "- Named functions with typed parameters\n"
        "- Can have side effects (write, send, execute)\n"
        "- Model decides when to call them\n"
        "- Used for: search, API calls, code execution, data mutation\n"
        "```python\n@mcp.tool()\ndef send_email(to: str, subject: str, body: str) -> str:\n"
        "    return send(to, subject, body)\n```\n\n"
        "Rule of thumb: if it's a noun (data), make it a resource. If it's a verb (action), make it a tool."
    ))
    pairs.append(msg(
        "How do I test and debug MCP servers?",
        "Testing tools:\n\n"
        "**1. MCP Inspector (interactive testing):**\n"
        "```bash\nnpx @modelcontextprotocol/inspector python server.py\n```\n"
        "Opens a web UI to list tools/resources, call them, and inspect responses.\n\n"
        "**2. MCP CLI:**\n"
        "```bash\nmcp dev server.py  # Start with hot reload\nmcp list tools     # List available tools\n"
        "mcp call search_documents --query 'test'  # Call a tool\n```\n\n"
        "**3. Unit testing:**\n"
        "```python\nimport pytest\nfrom mcp.server.fastmcp import FastMCP\n\n"
        "def test_search_tool():\n    mcp = create_server()\n"
        "    result = mcp.call_tool('search_documents', {'query': 'test'})\n"
        "    assert 'results' in result\n```\n\n"
        "**4. Debugging tips:**\n"
        "- Logs go to stderr (stdout is the MCP protocol)\n"
        "- Use `logging.getLogger('mcp')` for protocol-level logs\n"
        "- Add `--verbose` flag to your server for debug output\n"
        "- Check JSON-RPC message format if communication fails"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------------
def gen_evaluation() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do I use an LLM as a judge for evaluation?",
        "LLM-as-a-judge evaluates model outputs using another LLM:\n\n"
        "```python\ndef llm_judge(question, answer, reference=None, model='gpt-4o'):\n"
        "    prompt = f'''Rate the following answer on a scale of 1-5.\n\n"
        "Question: {question}\nAnswer: {answer}\n'''\n"
        "    if reference:\n        prompt += f'Reference answer: {reference}\\n'\n"
        "    prompt += '''\\nCriteria:\n- Accuracy (correct information)\n"
        "- Completeness (addresses all aspects)\n- Clarity (well-explained)\n\n"
        "Provide your rating as JSON: {{\"score\": N, \"reasoning\": \"...\"}}\n'''\n\n"
        "    response = client.chat.completions.create(\n"
        "        model=model,\n        messages=[{'role': 'user', 'content': prompt}],\n"
        "        response_format={'type': 'json_object'}\n    )\n"
        "    return json.loads(response.choices[0].message.content)\n```\n\n"
        "Best practices:\n"
        "- Use a stronger model as judge (GPT-4o judging smaller models)\n"
        "- Include rubric/criteria in the prompt\n"
        "- Use pairwise comparison for relative ranking\n"
        "- Randomize order to avoid position bias\n"
        "- Validate against human judgments (calibrate)"
    ))
    pairs.append(msg(
        "What is RAGAS and how do I use it for RAG evaluation?",
        "RAGAS (Retrieval-Augmented Generation Assessment) evaluates RAG pipelines:\n\n"
        "```python\nfrom ragas import evaluate\nfrom ragas.metrics import (\n"
        "    answer_relevancy,  # Does the answer address the question?\n"
        "    faithfulness,       # Is the answer grounded in the context?\n"
        "    context_recall,     # Were all relevant documents retrieved?\n"
        "    context_precision,  # Are retrieved documents relevant?\n"
        ")\nfrom datasets import Dataset\n\n"
        "eval_data = Dataset.from_dict({\n"
        "    'question': ['What is LoRA?'],\n"
        "    'answer': ['LoRA (Low-Rank Adaptation)...'],\n"
        "    'contexts': [['LoRA paper excerpt...']],\n"
        "    'ground_truth': ['Low-Rank Adaptation reduces...'],\n})\n\n"
        "results = evaluate(eval_data, metrics=[\n"
        "    answer_relevancy, faithfulness, context_recall, context_precision\n])\nprint(results)\n```\n\n"
        "Metrics explained:\n"
        "- **Faithfulness** (0-1): no hallucination, every claim supported by context\n"
        "- **Answer Relevancy** (0-1): answer is on-topic\n"
        "- **Context Precision** (0-1): retrieved docs are relevant (precision@k)\n"
        "- **Context Recall** (0-1): all needed info was retrieved (needs ground truth)\n\n"
        "Aim for >0.8 on all metrics for production."
    ))
    pairs.append(msg(
        "What are the main LLM evaluation benchmarks?",
        "Key benchmarks by category:\n\n"
        "**General Knowledge:**\n"
        "- **MMLU** (Massive Multitask Language Understanding): 57 subjects, multiple choice\n"
        "- **MMLU-Pro**: harder version with 10 choices + reasoning\n"
        "- **ARC** (AI2 Reasoning Challenge): grade-school science questions\n\n"
        "**Reasoning:**\n"
        "- **GSM8K**: grade-school math word problems\n"
        "- **MATH**: competition-level mathematics\n"
        "- **BBH** (Big-Bench Hard): 23 challenging tasks\n\n"
        "**Coding:**\n"
        "- **HumanEval**: Python function completion (164 problems)\n"
        "- **MBPP**: Python programming (974 problems)\n"
        "- **SWE-bench**: real GitHub issues\n"
        "- **LiveCodeBench**: continuously updated coding problems\n\n"
        "**Chat/Instruction:**\n"
        "- **MT-Bench**: multi-turn conversation quality (GPT-4 judge)\n"
        "- **AlpacaEval**: instruction following (GPT-4 judge)\n"
        "- **Chatbot Arena**: human preference ranking (ELO)\n\n"
        "**Safety:**\n"
        "- **TruthfulQA**: resistance to common misconceptions\n"
        "- **BBQ**: bias detection"
    ))
    pairs.append(msg(
        "How do I evaluate my fine-tuned model against the base model?",
        "Comprehensive evaluation strategy:\n\n"
        "**1. Perplexity comparison:**\n"
        "```bash\n# Base model\nllama-perplexity -m base.gguf -f eval.txt\n# Fine-tuned\n"
        "llama-perplexity -m finetuned.gguf -f eval.txt\n```\n\n"
        "**2. Task-specific accuracy:**\n"
        "```python\ndef evaluate_task(model, test_set):\n"
        "    correct = 0\n    for item in test_set:\n"
        "        response = model.generate(item['question'])\n"
        "        if extract_answer(response) == item['answer']:\n"
        "            correct += 1\n"
        "    return correct / len(test_set)\n```\n\n"
        "**3. Forgetting check (critical!):**\n"
        "- Run base model benchmarks on fine-tuned model\n"
        "- If general capability drops >5%, you have catastrophic forgetting\n"
        "- Test on: MMLU subset, coding, common sense reasoning\n\n"
        "**4. Win-rate comparison:**\n"
        "```python\n# Generate from both models, use judge\nfor q in test_questions:\n"
        "    base_answer = base_model.generate(q)\n"
        "    ft_answer = finetuned_model.generate(q)\n"
        "    winner = llm_judge.compare(q, base_answer, ft_answer)\n```\n\n"
        "**5. Human evaluation:** for subjective quality (tone, style, helpfulness)"
    ))
    pairs.append(msg(
        "What is pass@k and how is it calculated?",
        "**pass@k** measures the probability that at least one of K generated samples passes the test:\n\n"
        "```\npass@k = 1 - C(n-c, k) / C(n, k)\n```\n"
        "Where n = total samples generated, c = number that pass, k = samples considered.\n\n"
        "Unbiased estimator:\n"
        "```python\nimport numpy as np\nfrom math import comb\n\n"
        "def pass_at_k(n, c, k):\n    \"\"\"n: total samples, c: correct samples, k: k value\"\"\"\n"
        "    if n - c < k:\n        return 1.0\n    return 1.0 - comb(n - c, k) / comb(n, k)\n\n"
        "# Example: generated 20 samples, 5 passed, what's pass@1?\n"
        "print(pass_at_k(20, 5, 1))   # 0.25\nprint(pass_at_k(20, 5, 5))   # 0.717\n"
        "print(pass_at_k(20, 5, 10))  # 0.957\n```\n\n"
        "Usage:\n"
        "- **pass@1**: how often the first try works (most relevant for production)\n"
        "- **pass@10**: probability of finding a solution in 10 tries\n"
        "- Generate n>>k samples with temperature >0, then compute pass@k\n"
        "- Standard for code benchmarks (HumanEval, MBPP)"
    ))
    pairs.append(msg(
        "How do I set up automated evaluation for model comparison?",
        "```python\nimport json\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\n"
        "@dataclass\nclass EvalResult:\n    model: str\n    benchmark: str\n    score: float\n    metadata: dict\n\n"
        "def run_eval_suite(model_path, benchmarks=['mmlu', 'humaneval', 'gsm8k']):\n"
        "    results = []\n    for bench in benchmarks:\n"
        "        if bench == 'mmlu':\n            score = run_mmlu(model_path, subjects=['stem', 'humanities'])\n"
        "        elif bench == 'humaneval':\n            score = run_humaneval(model_path, n_samples=20)\n"
        "        elif bench == 'gsm8k':\n            score = run_gsm8k(model_path)\n"
        "        results.append(EvalResult(model=model_path, benchmark=bench, score=score, metadata={}))\n\n"
        "    # Save results\n    Path('eval_results').mkdir(exist_ok=True)\n"
        "    with open(f'eval_results/{Path(model_path).stem}.json', 'w') as f:\n"
        "        json.dump([r.__dict__ for r in results], f, indent=2)\n\n"
        "    return results\n\n"
        "# Compare models\nbase_results = run_eval_suite('base-model.gguf')\nft_results = run_eval_suite('finetuned-model.gguf')\n\n"
        "for b, f in zip(base_results, ft_results):\n"
        "    delta = f.score - b.score\n    print(f'{b.benchmark}: base={b.score:.3f} ft={f.score:.3f} delta={delta:+.3f}')\n```"
    ))
    pairs.append(msg(
        "What is ELO rating in the context of LLM evaluation?",
        "ELO rating ranks models based on pairwise human preferences (Chatbot Arena):\n\n"
        "**How it works:**\n"
        "1. User asks a question\n2. Two anonymous models respond\n3. User picks the better response\n"
        "4. ELO ratings update based on the outcome\n\n"
        "**Formula:**\n"
        "```\nExpected(A) = 1 / (1 + 10^((Rating_B - Rating_A) / 400))\n"
        "New_Rating_A = Rating_A + K * (Actual - Expected)\n```\n\n"
        "**Current top ELO scores (approximate, 2025):**\n"
        "| Model | ELO |\n"
        "|-------|-----|\n"
        "| GPT-4o | ~1280 |\n"
        "| Claude Opus 4 | ~1275 |\n"
        "| Gemini Ultra | ~1260 |\n"
        "| Llama 3.1 405B | ~1220 |\n"
        "| Qwen 2.5 72B | ~1200 |\n\n"
        "ELO advantages:\n"
        "- Captures subjective quality (not just accuracy)\n"
        "- Relative ranking (not absolute scores)\n"
        "- Continuously updated with new comparisons\n\n"
        "Limitations: position bias, verbosity bias, style preferences vary by user."
    ))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
GENERATORS = {
    "langchain": ("LangChain patterns", gen_langchain),
    "llamaindex": ("LlamaIndex patterns", gen_llamaindex),
    "rag": ("RAG patterns", gen_rag),
    "multiagent": ("Multi-agent systems", gen_multiagent),
    "prompting": ("Prompt engineering", gen_prompting),
    "function_calling": ("Function calling / tool use", gen_function_calling),
    "mcp": ("MCP protocol", gen_mcp),
    "evaluation": ("Evaluation methods", gen_evaluation),
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate LLM Orchestration training dataset")
    parser.add_argument("--stats", action="store_true", help="Print stats only")
    parser.add_argument("--categories", default=",".join(GENERATORS.keys()))
    args = parser.parse_args()

    requested = [c.strip() for c in args.categories.split(",")]
    total = 0
    stats: dict[str, int] = {}

    for cat in requested:
        if cat not in GENERATORS:
            logger.error("Unknown: %s", cat)
            sys.exit(1)
        label, gen_fn = GENERATORS[cat]
        pairs = gen_fn()
        stats[label] = len(pairs)
        total += len(pairs)
        if not args.stats:
            for pair in pairs:
                print(json.dumps(pair, ensure_ascii=False))

    logger.info("=== LLM Orchestration Generation Statistics ===")
    for label, count in stats.items():
        logger.info("  %-35s %5d pairs", label, count)
    logger.info("  %-35s %5d pairs", "TOTAL", total)


if __name__ == "__main__":
    main()
