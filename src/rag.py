from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from transformers import pipeline

from load_docs import load_documents


def create_rag():
    docs = load_documents()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    vectorstore = FAISS.from_documents(docs, embeddings)

    prompt = PromptTemplate(
        template="""
Você é um assistente acadêmico da Universidade Federal da Bahia (UFBA).

Responda usando APENAS as informações do contexto abaixo.
Se a resposta não estiver presente, diga:
"A informação não consta nos documentos analisados."

Contexto:
{context}

Pergunta:
{question}

Resposta:
""",
        input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa
