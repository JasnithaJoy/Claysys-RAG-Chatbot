from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vector_store import load_vector_store
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PROMPT_TEMPLATE = """
You are a helpful assistant. Use the context below to answer the question accurately.
If the answer is not in the context, say "I don't have enough information from this website."

Context:
{context}

Question: {question}

Answer:
"""

def get_answer(query: str) -> tuple[str, list[str]]:
    try:
        vectordb = load_vector_store()
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(query)
        docs = retriever.invoke(query)
        sources = [doc.metadata.get("source", "") for doc in docs]
        return answer, sources

    except Exception as e:
        return f"Error: {str(e)}", []