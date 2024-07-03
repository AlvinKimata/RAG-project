import streamlit as st
from rag_engine import *
from api import query

# Streamlit configuration
st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")
st.markdown('This is a web app that performs retrieval augmented generation<br> on arXiv articles on Software Engineering and Programming languages.', unsafe_allow_html=True)


# Document features
st.header('Document features')

st.text('Number of documents to retrieve.')
num_docs = st.slider('Number of RAG documents', 1, 3, 1)

st.text('Token size')
ctx_win = st.slider('Select the number of tokens the model should generate', 100, 500, 100)

def query_llm(retriever, query_text, num_docs, ctx_win):
    """Function to query the LLM and get the response."""
    context = retriever(query_text, num_docs)
    prompt = generate_prompt(query_text, context)
    response = query({
        "inputs": prompt,
        "parameters": {
            "top_k": 10,
            "top_p": 0.95,
            "temperature": 0.1,
            "max_new_tokens": ctx_win,
            "do_sample": True,
            "return_text": True,
            "return_full_text": True,
            "return_tensors": False,
            "clean_up_tokenization_spaces": True
        }
    })
    return response['text']

def boot():
    """Main function to perform RAG and display the chat interface."""
    query_text = st.text_input("Enter your query:")
    
    if st.button('Submit'):
        if query_text:
            with st.spinner('Fetching response...'):
                try:
                    response = query_llm(query_text, num_docs, ctx_win)
                    st.write("### Response")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter the query.")

if __name__ == '__main__':
    boot()
