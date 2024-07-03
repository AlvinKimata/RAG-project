from rag_engine import *
import streamlit as st

st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")


def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        memory=ConversationBufferMemory()
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

#Include feature sliders.
st.header('Document features')
col1, col2 = st.columns(2)

with col1:
    st.text('Number of documents.')
    num_docs = st.slider('Number of RAG documents', 1, 3, 1)

with col2:
    st.text('Context window')
    ctx_win = st.slider('Number of tokens for model to generate', 100, 2000, 100)



def process_documents():
    pass

def boot():
    #Code for performing RAG and returning LLM output.

    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    boot()