# Retrieval Augmented Generation (RAG) Engine

## Overview

This project implements a Retrieval Augmented Generation (RAG) Engine using Streamlit. The application performs retrieval augmented generation on arXiv articles focusing on Software Engineering and Programming Language topics.

## Features

- Web-based interface built with Streamlit
- Retrieval of relevant documents from a PostgreSQL database
- Generation of responses using a language model
- Customizable number of documents to retrieve
- Adjustable token size for generated responses

## Requirements

- Python 3.10
- Streamlit
- langchain
- langchain_postgres
- sentence_transformers
- psycopg2

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/AlvinKimata/RAG-project
   cd RAG-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the PostgreSQL database with the arXiv documents.

## Configuration

Ensure you have the correct PostgreSQL connection details in the `connection` variable:

```python
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. Use the sliders to adjust the number of documents to retrieve and the token size for generation.

4. Enter your query in the text input field and click 'Submit'.

5. The app will retrieve relevant documents and generate a response based on your query.

## Project Structure

- `app.py`: Main Streamlit application file
- `rag_engine.py`: Contains functions for document retrieval and prompt generation
- `api.py`: Handles the interaction with the language model API

## Functions

- `query_llm()`: Queries the language model with the generated prompt
- `similarity_search()`: Performs similarity search to retrieve relevant documents
- `document_template()`: Formats the retrieved documents
- `rag_function()`: Performs the RAG process
- `generate_prompt()`: Generates the prompt for the language model

## Note

This application uses a pre-trained language model and a pre-populated database of arXiv articles. Ensure you have the necessary API access and database set up correctly for the application to function properly.
