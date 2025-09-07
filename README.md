
# 법률상담 챗봇 (Full Project)

## 구성
- backend: Flask API + RAG service (backend/app.py, backend/rag_pipeline.py, backend/data_loader.py)
- frontend: Streamlit chat UI (frontend/app.py)
- data: raw_texts -> faiss_index

## 빠른 시작
1) 가상환경 생성 및 패키지 설치
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2) (옵션) .env 설정
   cp .env.example .env
   # 환경변수 수정

3) 벡터 DB 생성 (data_loader)
   cd backend
   python data_loader.py

4) 백엔드 실행
   python app.py

5) 프론트엔드 실행 (다른 터미널)
   cd frontend
   streamlit run app.py

## 주의사항
- 본 서비스는 참고용입니다. 법률 자문으로 사용할 수 없습니다.
- OpenAI 키를 넣지 않으면 LLM 호출 대신 문서요약 기반 모의응답이 동작합니다.
