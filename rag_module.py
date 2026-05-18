from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI 

prompt_template = """
You are a financial assistant. Answer the user's question based on the provided transaction history.

Context:
{context}

Question:
{question}

Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

def get_insight(transactions, question):
    text_data = ""
    for txn in transactions:
        text_data += f"{txn.get('title')}, {txn.get('amount')}, {txn.get('category')}, {txn.get('date')}\n"

    documents = [Document(page_content=text_data)]
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)

    llm = LlamaCpp(
        model_path="./models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        n_ctx=2048,
        verbose=True,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        ) 
    return qa.run(question)
