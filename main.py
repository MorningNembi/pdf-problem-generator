import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.prompts import loading
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_teddynote import logging
from dotenv import load_dotenv
import glob
import yaml
import os

# API KEY 정보로드
load_dotenv()

logging.langsmith("[Project] PDF RAG")

# 캐시 디렉토리 생성
os.makedirs(".cache/files", exist_ok=True)
os.makedirs(".cache/embeddings", exist_ok=True)

st.title("PDF 기반 문제 생성기 QA💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "chain_mc" not in st.session_state:
    st.session_state["chain_mc"] = None
if "chain_desc" not in st.session_state:
    st.session_state["chain_desc"] = None

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    # 객관식, 서술형 프롬프트 YAML 파일 선택
    mc_prompt_files = glob.glob("prompts/multiple_choice_*.yaml")
    desc_prompt_files = glob.glob("prompts/descriptive_*.yaml")
    selected_prompt_mc = st.selectbox("객관식 프롬프트 선택", mc_prompt_files, index=0)
    selected_prompt_desc = st.selectbox(
        "서술형 프롬프트 선택", desc_prompt_files, index=0
    )
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )
    generate_btn = st.button("문제 생성")


# 이전 대화 기록 출력 함수
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가 함수
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# PDF 파일 임베딩 생성 (캐싱)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 문서 로드
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 임베딩 및 벡터 스토어 생성
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


# 체인 생성 함수 (프롬프트 YAML 파일을 받아 체인 생성)
def create_chain(retriever, prompt_path, model_name="gpt-4o"):
    with open(prompt_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    prompt = loading.load_prompt_from_config(config)
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 파일 업로드 시 retriever 및 체인 생성
if uploaded_file:
    retriever = embed_file(uploaded_file)
    st.session_state["retriever"] = retriever
    st.session_state["chain_mc"] = create_chain(
        retriever, selected_prompt_mc, model_name=selected_model
    )
    st.session_state["chain_desc"] = create_chain(
        retriever, selected_prompt_desc, model_name=selected_model
    )

# 초기화 버튼
if clear_btn:
    st.session_state["messages"] = []

print_messages()

# "문제 생성" 버튼 클릭 시, 객관식 3문제와 서술형 2문제 생성
# "문제 생성" 버튼 클릭 시, 객관식 3문제와 서술형 2문제를 생성하도록 수정
if generate_btn:
    if st.session_state["retriever"] is None:
        st.error("먼저 PDF 파일을 업로드 해주세요.")
    else:
        questions_output = []
        # 객관식 문제 3개 생성
        for i in range(3):
            user_query = f"객관식 문제 {i+1} 생성"
            # .invoke()를 사용하여 체인 실행 (또는 chain(user_query)도 가능)
            response = st.session_state["chain_mc"].invoke(user_query)
            questions_output.append(f"**객관식 문제 {i+1}:**\n{response}")
        # 서술형 문제 2개 생성
        for i in range(2):
            user_query = f"서술형 문제 {i+1} 생성"
            response = st.session_state["chain_desc"].invoke(user_query)
            questions_output.append(f"**서술형 문제 {i+1}:**\n{response}")

        st.write("### 생성된 문제")
        for q in questions_output:
            st.write(q)


# 기존 채팅 입력 (추가 질문용)
user_input = st.chat_input("궁금한 내용을 물어보세요!")
if user_input:
    st.chat_message("user").write(user_input)
    # 기본적으로 객관식 체인을 사용해 답변(필요에 따라 조정 가능)
    response = st.session_state["chain_mc"].stream(user_input)
    ai_answer = ""
    container = st.empty()
    for token in response:
        ai_answer += token
        container.markdown(ai_answer)
    add_message("user", user_input)
    add_message("assistant", ai_answer)
