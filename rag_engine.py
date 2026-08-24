import os
import re
from typing import List, Tuple, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

class RAGEngine:
    """Manages Vector Store Indexing, Context Retrieval, and High-Intelligence Generation."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Set it in .env or pass it explicitly.")
        
        # Fast local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Gemini LLM with balanced creativity and facts
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=self.api_key,
            temperature=0.3
        )
        self.vector_store = None

    def build_vector_store(self, chunks: List[Document]) -> int:
        """Converts text chunks to embeddings and indexes them inside FAISS."""
        if not chunks:
            raise ValueError("No text chunks provided for vector indexing.")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return len(chunks)

    def query(self, question: str, top_k: int = 6) -> Dict[str, any]:
        """Retrieves context and synthesizes comprehensive, human-level responses."""
        if not self.vector_store:
            raise RuntimeError("Vector store is not initialized. Upload a document first.")

        # Retrieve expanded context window to cover full system workflows
        retrieved_docs = self.vector_store.similarity_search(question, k=top_k)

        context_str = ""
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", 0) + 1
            context_str += f"\n--- Document Chunk {i} (Source: {source}, Page: {page}) ---\n{doc.page_content}\n"

        system_prompt = PromptTemplate(
            template="""
            You are NexusRAG, an enterprise AI Knowledge Assistant operating with human-level intelligence, domain expertise, and natural technical synthesis.

            HUMAN-LEVEL INTELLIGENCE & ADAPTABILITY:
            1. **Format Precision**: Adapt your output style strictly to the user's prompt:
               - If requested in **paragraphs**, write detailed, comprehensive multi-paragraph explanations that thoroughly elaborate on mechanisms, causes, and impacts.
               - If requested in **tables**, organize components into a clean Markdown table.
               - If requested in **bullet points**, provide well-structured itemized breakdowns.
            2. **Exhaustive Synthesis**: Do NOT give shallow 2-sentence summaries when deep technical context exists. Connect the dots between client-side components, server-side APIs, database schemas, security toggles, and workflows.
            3. **Typo & Intent Tolerance**: Handle user typos, informal phrasing, or grammatical mistakes gracefully by accurately inferring technical intent.
            4. **Visual Highlights**: Use **bold text** for key concepts, code blocks (` ``` `) for payloads/code, and clear spacing for scannability.

            GROUNDEDNESS RULE:
            - Synthesize deeply using the facts provided in the document context.
            - If the context does not contain sufficient information, state clearly: "⚠️ The provided document context does not contain enough detail to answer this specific question."

            MUST INCLUDE AT THE VERY END:
            Conclude your entire response with a single line formatted EXACTLY like this:
            SUGGESTED_QUESTION: [Insert one relevant, insightful follow-up question here]

            Document Context:
            {context}

            User Query: {question}

            Detailed Response:
            """,
            input_variables=["context", "question"]
        )

        formatted_prompt = system_prompt.format(context=context_str, question=question)
        response = self.llm.invoke(formatted_prompt)
        
        # Extract clean text string
        raw_content = response.content
        if isinstance(raw_content, list):
            text_blocks = [item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in raw_content]
            clean_text = "\n".join(text_blocks)
        else:
            clean_text = str(raw_content)

        # Parse out the suggested follow-up question for UI callout rendering
        follow_up = None
        if "SUGGESTED_QUESTION:" in clean_text:
            parts = clean_text.split("SUGGESTED_QUESTION:")
            main_answer = parts[0].strip()
            follow_up = parts[1].strip()
        else:
            main_answer = clean_text.strip()

        return {
            "answer": main_answer,
            "follow_up": follow_up,
            "sources": retrieved_docs
        }