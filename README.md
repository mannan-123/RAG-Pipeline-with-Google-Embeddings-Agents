# 🧠 LangChain-Pinecone RAG Pipeline with Google Embeddings & Agents

This repository demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline using [LangChain](https://www.langchain.com/), [Pinecone](https://www.pinecone.io/), Google Generative AI embeddings, and intelligent agents. The use case involves question-answering over documents, specifically a sample PDF from UET Lahore, with conversational memory and agentic decision-making via custom tools.

---

## 🚀 Features

- Load and process PDFs using LangChain's `PyMuPDFLoader`
- Generate semantic document embeddings via Google Generative AI
- Store and retrieve vectors from Pinecone with namespaces
- Calculate cosine similarity manually or through LangChain
- Perform similarity search with and without scores
- Build a custom RAG pipeline using agents and LangChain tools
- Integrate conversational memory for context-aware interactions
- Use `TavilySearch` as a fallback tool if no information is found in Pinecone

---

## 🛠️ Installation

Install all required dependencies:

```bash
pip install langchain python-dotenv tiktoken langchain-pinecone openai langchain-openai langchainhub ipykernel pandas langchain-groq
```

## 🔐 Environment Setup

Set up your `.env` or export the following environment variables:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langchain_key
GOOGLE_API_KEY=your_google_api_key
COHERE_API_KEY=your_cohere_api_key
PINECONE_API_KEY=your_pinecone_key
TAVILY_API_KEY=your_tavily_key
```

---

## 📄 Document Loading & Preprocessing

We use `PyMuPDFLoader` to load the `UET_Lahore.pdf` and LangChain's `RecursiveCharacterTextSplitter` to split the content into chunks for embedding.

---

## 🧠 Embedding & Vector Store

- Embeddings: `GoogleGenerativeAIEmbeddings`
- Vector DB: Pinecone (cosine similarity, AWS region)

```python
# Example of embedding and storing
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = PineconeVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    index_name="langchain-pinecone-malik-uet-lahore",
    namespace="main"
)
```

---

## 🔍 Similarity Search

Perform document similarity search using:

```python
vectorstore.similarity_search(query, k=4)
vectorstore.similarity_search_with_score(query, k=4)
```

---

## 🧩 Agent Setup with Tools

Agents are created using LangChain's ReAct agent and multiple tools:

- **Pinecone Document Store** – Primary retrieval tool
- **Tavily Search** – Web-based fallback tool

A custom prompt template ensures the agent always queries Pinecone first.

```python
tools = [
    Tool(name="Pinecone Document Store", func=qa_db.run, description="..."),
    Tool(name="Tavily", func=tavily.run, description="...")
]
```

---

## 💬 Conversational Memory & RetrievalQA

Maintain recent context using `ConversationBufferWindowMemory`:

```python
conversational_memory = ConversationBufferWindowMemory(k=5, return_messages=True)
qa_db = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
```

---

## 🧪 Run Agent Query

Once the agent is initialized:

```python
response = agent_executor.invoke({
    "input": "What is the role of PROF. DR. HAB. ...?"
})
```

## 📌 Notes

- You can switch embedding models (e.g., OpenAI, Cohere) as needed
- Make sure your Pinecone index is initialized with the correct dimension (e.g., 768 for Google)
- Use `hub.pull("hwchase17/react")` for prompt templates

---

## 📚 Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [Pinecone Docs](https://docs.pinecone.io/)
- [Google Generative AI](https://ai.google.dev/)
- [Tavily Search](https://www.tavily.com/)

---
