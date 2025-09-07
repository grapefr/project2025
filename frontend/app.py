
import streamlit as st
import requests, os, time

st.set_page_config(page_title='법률상담 챗봇', layout='wide')
st.title('⚖️ 법률상담 챗봇 (RAG 기반)')

DISCLAIMER = '본 서비스는 참고용 법률정보를 제공합니다. 법률 자문으로 사용할 수 없습니다.'

with st.sidebar:
    st.markdown('### 설정')
    api_url = st.text_input('Backend URL', value=os.environ.get('BACKEND_URL', 'http://localhost:8000'))
    top_k = st.slider('Top-K (프론트에서 표시용)', 1, 10, 5)
    st.markdown('---')
    st.markdown(f'**면책:** {DISCLAIMER}')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.chat_message('user').markdown(msg['content'])
    else:
        st.chat_message('assistant').markdown(msg['content'])

user_input = st.chat_input('질문을 입력하세요...')
if user_input:
    st.session_state.messages.append({'role':'user','content':user_input})
    # call backend
    try:
        resp = requests.post(f'{api_url}/chat', json={'query': user_input}, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            answer = data.get('answer') or data.get('result') or '응답이 없습니다.'
            sources = data.get('sources', [])
            display = f"{answer}\n\n---\n출처:\n" + '\n'.join([f"- {s.get('title','')}: {s.get('snippet','')[:200]}" for s in sources])
        else:
            display = f'Error: {resp.status_code} {resp.text}'
    except Exception as e:
        display = f'Exception calling backend: {e}'
    st.session_state.messages.append({'role':'assistant','content': display})
    st.experimental_rerun()
