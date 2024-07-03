import streamlit as st
from rag_engine import *
from api import query

# Streamlit configuration
st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")
st.markdown('This is a web app that performs retrieval augmented generation on arXiv articles on <i>Software Engineering</i> and <i>Programming language</i> topics. <br><br> Example: <i>"Provide examples of compiler optimization techniques."</i>', unsafe_allow_html=True)


# Document features
st.header('Document features')

st.text('Number of documents to retrieve.')
NUM_DOCS = st.slider('Number of RAG documents', 1, 3, 1)


def query_llm(query_text, num_docs):
    """Function to query the LLM and get the response."""
    prompt = generate_prompt(query_text, num_docs)
    response = query({
        "inputs": prompt,
        "parameters": {
            "top_k": 1,
            "top_p": 0.95,
            "temperature": 0.2,
            "max_new_tokens": 200,
            "do_sample": True,
            "return_text": True,
            "return_full_text": False,
            "return_tensors": False,
            "clean_up_tokenization_spaces": True
        }
    })
    return response

def main():
    """Main function to perform RAG and display the chat interface."""
    query_text = st.text_input("Enter your query:")
    
    if st.button('Submit'):
        if query_text:
            with st.spinner('Fetching response...'):
                try:
                    response = query_llm(query_text, num_docs = NUM_DOCS)
                    st.write("### Response")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter the query.")

if __name__ == '__main__':
    main()
