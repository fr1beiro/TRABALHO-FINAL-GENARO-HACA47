from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents():
    files = [
        "data/regimento_geral_ufba.pdf",
        "data/regulamento_graduacao.pdf",
        "data/resolucao_ac_bi_2025.pdf",
        "data/guia_procedimentos_ihac.pdf",
        "data/manual_calouro.pdf"
    ]

    documents = []
    for file in files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)
