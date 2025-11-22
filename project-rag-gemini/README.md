# 🤖 Chatbot RAG: Geração Aumentada com Google Gemini e LangChain

Este projeto implementa um sistema de **RAG (Retrieval-Augmented Generation)**, a arquitetura moderna para construir chatbots que respondem a perguntas baseadas em documentos específicos, **mitigando o problema central de alucinação em LLMs**.

O sistema utiliza o artigo técnico *"Attention Is All You Need"* (o paper original dos Transformers) como base de conhecimento.

---

## 🛠 Tecnologias Chave (GenAI Engineering)

* **Python 3.9+**
* **LangChain:** Framework de orquestração de IA para construir a cadeia RAG.
* **Google Gemini 1.5 Flash:** O Large Language Model (LLM) de alta performance responsável pela **Geração Aumentada** das respostas.
* **FAISS (Meta):** Banco de dados de vetores ultrarrápido para **persitência e busca** (`Retrieval`) dos embeddings.
* **Google Generative AI Embeddings:** Modelo de embedding para **vetorização** do texto.
* **Boas Práticas:** Uso de `python-dotenv`, `venv` e `.gitignore` para segurança e reprodutibilidade.

---

## 🚀 Arquitetura (Como Funciona)

A lógica RAG garante que a resposta seja factualmente precisa, seguindo três etapas orquestradas:

1.  **Ingestão/Pipeline de Dados:** O script `ingest.py` carrega o PDF, divide o texto em *chunks* e cria *embeddings* vetoriais armazenados localmente via FAISS.
2.  **Recuperação (Retrieval):** Ao receber uma pergunta, o sistema busca os 4 trechos (*top-k*) mais relevantes no **Vector Store** (FAISS).
3.  **Geração Aumentada (Generation):** O contexto recuperado + a pergunta são enviados ao Gemini, instruído via **Prompt Engineering** a responder estritamente baseado nos dados fornecidos.

---

## ⚙️ Instalação e Uso

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU_USUARIO/Portfolio-RAG-Chatbot-Gemini.git](https://github.com/SEU_USUARIO/Portfolio-RAG-Chatbot-Gemini.git)
    cd Portfolio-RAG-Chatbot-Gemini
    ```
2.  **Configuração de Segurança:** Crie um arquivo `.env` na raiz com sua chave da API do Google:
    ```text
    GOOGLE_API_KEY="SUA_CHAVE_AQUI"
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Execute a Ingestão de Dados (Criação do Vector Store):**
    ```bash
    python ingest.py
    ```
5.  **Inicie o Chatbot:**
    ```bash
    python rag_chat.py
    ```

---

## 🧪 Demonstração de Mitigação de Alucinação

| Pergunta | Resultado Esperado | Valor Demonstrado |
| :--- | :--- | :--- |
| "O que é Scaled Dot-Product Attention?" | Explicação técnica baseada no artigo. | ✅ Precisão e Uso do Contexto |
| "Quais são os autores?" | Lista dos autores do paper. | ✅ Extração de Fato |
| **"Qual a capital da França?"** | **"Não encontrei a informação no documento."** | 🛑 **Prova de Mitigação de Alucinações** |

---
*Projeto desenvolvido para fins de estudo em Engenharia de IA e LLMs.*