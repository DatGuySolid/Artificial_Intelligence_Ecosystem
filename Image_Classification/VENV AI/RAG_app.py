import logging
from transformers import logging as hf_logging
import warnings

logging.getLogger("langchain_text_splitters").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Warning: No OpenAI key found—check .env file")

chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 20

cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_m = 8

with open("Selected_Document.txt", "r", encoding="utf-8") as f:
    text = f.read()
print(f"Document loaded ({len(text)} characters)")

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
chunks = text_splitter.split_text(text)
print(f"Split into {len(chunks)} chunks")

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)
print("FAISS index built")

def retrieve_chunks(question, k=top_k):
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_arr = np.array(q_vec).astype('float32')
    D, I = faiss_index.search(q_arr, k)
    retrieved = [chunks[i] for i in I[0] if i != -1]
    return retrieved

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(cross_encoder_name)
print("Re-ranker loaded")

def dedupe_preserve_order(items):
    seen = set()
    unique = []
    for item in items:
        norm = ' '.join(item.split())
        if norm not in seen:
            seen.add(norm)
            unique.append(item)
    return unique

def rerank_chunks(question, candidate_chunks, m=top_m):
    pairs = [[question, chunk] for chunk in candidate_chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidate_chunks), reverse=True)
    top_chunks = [chunk for score, chunk in ranked[:m]]
    return dedupe_preserve_order(top_chunks)

def answer_question(question):
    candidates = retrieve_chunks(question)
    relevant_chunks = rerank_chunks(question, candidates)
    context = "\n\n".join(relevant_chunks)
    system_prompt = "You are a knowledgeable assistant that answers questions based on the provided context. If the answer is not in the context, say you don’t know."
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {e}"

if __name__ == "__main__":
    print("RAG System Ready! Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if question:
            print("Thinking...")
            answer = answer_question(question)
            print("\nAnswer:", answer)