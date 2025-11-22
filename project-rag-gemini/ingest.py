# 1- PIPELINE DE DADOS

import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Carregar variáveis de ambiente
load_dotenv(override=True)

def create_vector_db():
    print("--- Iniciando Processo de Ingestão ---")
    
    # 1. Carregar o PDF
    if not os.path.exists("attention.pdf"):
        print("Erro: O arquivo 'attention.pdf' não foi encontrado.")
        return

    print("Carregando PDF...")
    loader = PyPDFLoader("attention.pdf")
    pages = loader.load()
    print(f"PDF carregado: {len(pages)} páginas.")

    # 2. Split (Dividir em chunks)
    print("Dividindo texto...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(pages)
    print(f"Chunks criados: {len(docs)}")

    # 3. Embeddings e Vector Store
    print("Gerando Embeddings e criando Vector Store...")
    my_api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=my_api_key 
    )
    batch_size = 30  # Processar 30 chunks por vez
    vector_store = None

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        print(f"Processando lote {i} a {i + batch_size}...")
        
        if vector_store is None:
            # Cria o banco com o primeiro lote
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            # Adiciona os próximos lotes ao banco existente
            vector_store.add_documents(batch)
        
        time.sleep(5)
        #Fiz esta adaptação para processar os lotes de maneira a se adaptar ao meu plano de API gratuito
    
    # Salvar localmente
    vector_store.save_local("faiss_index_attention")
    print("--- SUCESSO TOTAL: Banco de vetores criado e salvo! ---")

if __name__ == "__main__":
    create_vector_db()