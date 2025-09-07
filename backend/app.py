
from flask import Flask, request, jsonify
from flask_cors import CORS
from .rag_pipeline import RAGService

app = Flask(__name__)
CORS(app)
rag = RAGService()

@app.post('/chat')
def chat():
    data = request.get_json() or {}
    q = data.get('query', '').strip()
    if not q:
        return jsonify({'error': 'empty query'}), 400
    res = rag.answer(q)
    return jsonify(res)

@app.get('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
