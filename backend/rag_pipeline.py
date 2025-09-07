
import os
from typing import List
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from .config import OPENAI_API_KEY, VECTOR_PERSIST_DIR, TOP_K, MIN_CONTEXT_TOKENS, EMBEDDING_BACKEND

# Simple rerank using CrossEncoder if available
def rerank(question: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [(question, d.page_content[:512]) for d in docs]
        scores = ce.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:top_k]]
    except Exception:
        return docs[:top_k]

SYSTEM_PROMPT = '''당신은 한국 법률 문서를 요약하고 출처를 정확히 표기하는 법률 도우미입니다.
규칙:
1) 당신은 주어진 문맥(context) 안의 정보만 사용하여 답변합니다.
2) 문맥에 없으면 추측하지 말고 '문맥에 근거한 답변을 찾을 수 없습니다'라고 답하세요.
3) 모든 주요 주장 끝에 출처를 대괄호 형태로 표기하세요.
4) 이 서비스는 법률 자문이 아니며 참고용입니다.
'''

def build_prompt():
    return PromptTemplate(
        input_variables=['context', 'question'],
        template=SYSTEM_PROMPT + "\n[문맥]\n{context}\n[질문]\n{question}\n\n답변 양식:\n1) 요약\n2) 근거(한두 문장)\n3) 출처 목록"
    )

class RAGService:
    def __init__(self):
        # load vectorstore
        self.vs = None
        if os.path.exists(VECTOR_PERSIST_DIR):
            try:
                # embeddings object is not needed to load FAISS local; but LangChain FAISS API requires embeddings param
                # We'll use a lightweight dummy embeddings when loading; actual retrieval uses FAISS's internal index
                from langchain_community.embeddings import SentenceTransformerEmbeddings
                emb = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                self.vs = FAISS.load_local(VECTOR_PERSIST_DIR, emb, allow_dangerous_deserialization=True)
            except Exception as e:
                print('Failed to load vectorstore:', e)
                self.vs = None
        self.prompt = build_prompt()
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, openai_api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def retrieve(self, query: str, top_k: int = TOP_K):
        docs = []
        if self.vs:
            docs = self.vs.similarity_search(query, k=top_k)
        # basic dedup
        seen = set()
        uniq = []
        for d in docs:
            key = d.page_content[:200]
            if key not in seen:
                seen.add(key); uniq.append(d)
        return uniq

    def answer(self, query: str):
        # retrieve
        candidates = self.retrieve(query, top_k=TOP_K*3)
        if not candidates:
            return {'answer': "문맥에 근거한 답변을 찾을 수 없습니다.", 'sources': []}
        # rerank
        top_docs = rerank(query, candidates, top_k=TOP_K)
        total_tokens = sum(len(d.page_content.split()) for d in top_docs)
        if total_tokens < MIN_CONTEXT_TOKENS:
            return {'answer': "문맥에 근거한 답변을 찾을 수 없습니다.", 'sources': []}
        context = '\n---\n'.join([d.page_content for d in top_docs])
        prompt = self.prompt.format(context=context, question=query)
        # call LLM (use chain for convenience)
        if self.llm:
            chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type='stuff', retriever=None, chain_type_kwargs={'prompt': self.prompt})
            result_text = chain.run({'context': context, 'question': query})
        else:
            # no OpenAI key: fallback simple summarize using top_docs content
            result_text = " (모의 응답) " + context[:1000]
        sources = [{'title': getattr(d, 'metadata', {}).get('title', 'unknown'), 'snippet': d.page_content[:300]} for d in top_docs]
        return {'answer': result_text, 'sources': sources}
