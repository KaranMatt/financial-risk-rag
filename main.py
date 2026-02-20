import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import os
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

class Questionclass(BaseModel):
    question:str


class ResponseClass(BaseModel):
    quesiton:str
    response:str

vector_db=None
pipe=None
embeddings=None
rerank=None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global pipe,embeddings,vector_db,rerank

    print('Models Loading....')
    
    embeddings=HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    print('Embeddings Loaded')
    vector_db=FAISS.load_local('RAG Vector DB',embeddings=embeddings,allow_dangerous_deserialization=True)
    print('Vector DB Loaded')
    MODEL='Qwen/Qwen2.5-1.5B-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,device_map='auto',dtype=torch.bfloat16,low_cpu_mem_usage=True)
    pipe=pipeline(task='text-generation',temperature=0.3,do_sample=True,tokenizer=tokenizer,model=model,max_new_tokens=512,repetition_penalty=1.1,
              no_repeat_ngram_size=3)
    rerank=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
    print('Reranker Model Loaded')
    print('Models Loaded!!!')

    yield
    print('Shutdown Initiated')
    vector_db=None
    pipe=None
    rerank=None
    pipe=None

app=FastAPI(title='MultiDoc Financial Risk RAG API',lifespan=lifespan)

@app.get('/root')
def root():
    return {'message':'Welcome to the RAG API'}

@app.get('/health')
def health():
    if vector_db and pipe:
        return {'status':'Active','Models Loaded':True}
    else:
        return {'status':'Not yet Active','Models Loaded':False}
    
@app.post('/ask',response_model=ResponseClass)
def predict(request:Questionclass):
    initial_search=vector_db.similarity_search(request.question,k=20)
    pairs=[[request.question,doc.page_content] for doc in initial_search]
    scores=rerank.predict(pairs)
    score_results=sorted(zip(scores,initial_search),key=lambda x:x[0],reverse=True)
    final_results=[doc for score,doc in score_results[:3]]
    context_list=[]
    for doc in final_results:
        file_path=doc.metadata.get('source','unknown')
        filename=os.path.basename(file_path)
        page_num=doc.metadata.get('page',0)+1
        header=f'[Doc : {filename} | Page : {page_num}]'
        context_list.append(f'{header}\n{doc.page_content}')
    context='\n\n-\n\n'.join(context_list)
    prompt=f'''You are a financial analyst assistant. Answer the question using ONLY the provided context.

IMPORTANT RULES:
1. Be concise - maximum 500 words
2. Always cite sources: [Doc: filename | Page: X]
3. If context is insufficient, state: "Based on available documents, I cannot fully answer this."
4. No speculation beyond the documents
5. For financial metrics, copy exact numbers from source

Question: {request.question}

Context:
{context}

Answer (concise, cited):'''

    response=pipe(prompt,return_full_text=False)
    answer=response[0]['generated_text']

    return ResponseClass(quesiton=request.question,response=answer)