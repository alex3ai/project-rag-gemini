import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Carregar .env forçando atualização
load_dotenv(override=True)

# 2. LIMPEZA DA CHAVE (Igual fizemos no ingest.py)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERRO: Chave não encontrada.")
    exit()
api_key = api_key.strip() # Remove espaços e lixo

class RAGChatbot:
    def __init__(self):
        self.vector_store_path = "faiss_index_attention"
        self.model_name = "gemini-2.5-flash"
        self.setup_chain()

    def format_docs(self, docs):
        """Formata os documentos recuperados em uma única string de texto."""
        return "\n\n".join(doc.page_content for doc in docs)

    def setup_chain(self):
        # Verificar se o banco existe
        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError("Banco de vetores não encontrado. Execute ingest.py primeiro.")
        
        print("--- Carregando Banco de Vetores... ---")
        
        # 3. IMPORTANTE: Usar o MESMO modelo do ingest.py e passar a chave
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",  # <--- TEM QUE SER IGUAL AO INGEST
            google_api_key=api_key              # <--- Passando a chave limpa
        )
        
        try:
            vector_store = FAISS.load_local(
                self.vector_store_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Erro ao carregar FAISS: {e}")
            print("Dica: Tente apagar a pasta 'faiss_index_attention' e rodar o ingest.py novamente.")
            exit()
        
        # Configurar o Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Configurar o LLM (Passando a chave aqui também)
        llm = ChatGoogleGenerativeAI(
            model=self.model_name, 
            temperature=0.3,
            google_api_key=api_key  # <--- Passando a chave limpa
        )

        # Prompt
        template = """
        Você é um assistente especializado em responder perguntas sobre o artigo "Attention Is All You Need".
        Use os trechos de contexto recuperados abaixo para responder à pergunta do usuário.
        Se a resposta não estiver no contexto, diga: "Não encontrei essa informação no documento."
        
        Contexto Recuperado:
        {context}

        Pergunta do Usuário:
        {question}

        Resposta:
        """
        prompt = PromptTemplate.from_template(template)

        # Criar a Chain
        self.chain = (
            {
                "context": retriever | self.format_docs, # Formata os docs antes de passar
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, query):
        try:
            response = self.chain.invoke(query)
            return response
        except Exception as e:
            return f"Ocorreu um erro ao processar sua pergunta: {e}"

# Bloco de execução principal
if __name__ == "__main__":
    bot = RAGChatbot()
    
    print("\n🤖 Chatbot RAG Iniciado (Digite 'sair' para encerrar)")
    print("-" * 50)
    
    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando...")
            break
        
        print("Bot: Pensando...")
        answer = bot.ask(user_input)
        print(f"Bot: {answer}")