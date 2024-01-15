from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

DB_FAISS_PATH="vectorstores/db_faiss"

custom_prompt_template="""Use following pieces of information to answer the user's question.
If you don't know the answer, please just say that "I don't know", don't try to make up answer.

Context : {context}
Question : {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt Template for QA Retrieval for each vector stores
    """

    prompt=PromptTemplate(template=custom_prompt_template,input_variables=['context','question'])

    return prompt

def load_llm():
    llm=CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type='llama',
        max_new_tokens=512,
        temprature=0.5
    )
    return llm

def retrieval_qa_chain(llm,prompt,db):
    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt':prompt}
    )

    return qa_chain

def qa_bot():
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db=FAISS.load_local(DB_FAISS_PATH,embeddings)
    llm=load_llm()
    qa_prompt=set_custom_prompt()
    qa=retrieval_qa_chain(llm,qa_prompt,db)
    return qa

def chat(query):
    qa_result=qa_bot()
    response=qa_result({'query':query})
    return response


if __name__=="__main__":
    result=chat("Hi, What is LLM?")
    print(result)