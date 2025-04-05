import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from anthropic import Anthropic
from openai import OpenAI
import numpy as np
from typing import List, Dict
import requests
import os
from dotenv import load_dotenv
import tempfile
import chromadb
from chromadb.config import Settings

# Load from specific file
load_dotenv('keys.env')

# Get API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Validate API keys
if not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY and ANTHROPIC_API_KEY in your environment variables")



class DocumentProcessor:
    def __init__(self, openai_api_key):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create a directory for Chroma
        os.makedirs("chroma_db", exist_ok=True)
        
        # Initialize Chroma with persistent directory
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")

    def process_pdf(self, file_path):
        """Process a PDF file and return chunks of text."""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Split text into chunks
            chunks = self.text_splitter.split_documents(pages)

            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection("pdf_collection")
            except:
                pass

            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="chroma_db",
                collection_name="pdf_collection"
            )
            
            return vectorstore

        except Exception as e:
            raise Exception(f"PDF processing error: {str(e)}")

    def process_text(self, text_content):
        """Process raw text and return chunks."""
        try:
            chunks = self.text_splitter.split_text(text_content)

            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection("text_collection")
            except:
                pass

            # Create vector store
            texts = [{"text": chunk} for chunk in chunks]
            vectorstore = Chroma.from_texts(
                texts=[doc["text"] for doc in texts],
                embedding=self.embeddings,
                persist_directory="chroma_db",
                collection_name="text_collection"
            )

            return vectorstore
            
        except Exception as e:
            raise Exception(f"Text processing error: {str(e)}")

    def get_relevant_chunks(self, vectorstore, query, k=3):
        """Retrieve the most relevant chunks for a given query."""
        try:
            results = vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            raise Exception(f"Error retrieving chunks: {str(e)}")

class StudyAssistant:
    def __init__(self, anthropic_api_key, openai_api_key):
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        self.document_processor = DocumentProcessor(openai_api_key)
        self.vectorstore = None

    def load_document(self, file_path):
        """Load and process a document."""

        if file_path.endswith('.pdf'):
            try:
                self.vectorstore = self.document_processor.process_pdf(file_path)
            except Exception as e:
                raise Exception(f"Error processing PDF: {str(e)}")
        else:
            try:
                # First trying UTF-8
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
            except UnicodeDecodeError:
                try:
                    # Then cp1251 (Windows-1251) encoding
                    with open(file_path, 'r', encoding='cp1251') as file:
                        text_content = file.read()
                except UnicodeDecodeError:
                    # If both fail, trying binary mode and decode with errors ignored
                    with open(file_path, 'rb') as file:
                        text_content = file.read().decode('utf-8', errors='ignore')
            
            self.vectorstore = self.document_processor.process_text(text_content)
    

    def generate_response(self, query):
        """Generate a response using RAG with Claude."""
        if not self.vectorstore:
            return "Please load a document first."

        # Get relevant chunks
        relevant_chunks = self.document_processor.get_relevant_chunks(
            self.vectorstore,
            query
        )

        # Construct the prompt
        context = "\n\n".join(relevant_chunks)
        prompt = f"""Based on the following excerpts from the document:

{context}

Question: {query}

Please provide a clear and comprehensive response. If the provided context doesn't 
contain enough information to fully answer the question, please indicate that."""

        # Generate response using Claude
        message = self.anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return message.content, relevant_chunks

class MyRAG:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.documents = []
        self.embeddings = []
    
    def search_papers(self, query: str, limit: int = 5) -> List[Dict]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,abstract,year,authors,citationCount'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get('data', [])
        except Exception as e:
            st.error(f"Error searching papers: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def process_papers(self, papers: List[Dict]):
        self.documents = []
        self.embeddings = []
        
        for paper in papers:
            doc_text = f"Title: {paper.get('title', '')}\n"
            if paper.get('abstract'):
                doc_text += f"Abstract: {paper.get('abstract')}\n"
            
            self.documents.append({
                'text': doc_text,
                'metadata': {
                    'title': paper.get('title'),
                    'year': paper.get('year'),
                    'citations': paper.get('citationCount', 0)
                }
            })
            self.embeddings.append(self.get_embedding(doc_text))

    def find_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.get_embedding(query)
        
        similarities = []
        for doc_embedding in self.embeddings:
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(similarity)
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i]['text'] for i in top_indices]
    
    def get_summary(self, query: str, relevant_docs: List[str]) -> str:
        prompt = (
            f"Based on these research paper excerpts, provide a comprehensive summary addressing: {query}\n\n"
            "Papers:\n"
            "----------------------------------------\n"
            f"{chr(10).join(relevant_docs)}\n"
            "----------------------------------------\n\n"
            "Please provide:\n"
            "1. Main findings and key points\n"
            "2. Notable methodologies mentioned\n"
            "3. Key implications\n"
            "4. Current research direction in this area\n\n"
            "Format as a clear academic summary."
        )

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # Access the content correctly from the response
        return response.choices[0].message.content

    
    

def main():
    st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")
    
    st.title("Research Assistant")
    st.write("Upload a document or search academic papers for analysis")

    # Create tabs
    tab1, tab2 = st.tabs(["Document Analysis", "Research Paper Search"])
    
    with tab1:
        st.header("Document Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a document (PDF or TXT)",
            type=["pdf", "txt"]
        )
        
        if uploaded_file:
            # Save the uploaded file temporarily with proper handling
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name[uploaded_file.name.rfind('.'):]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Initialize StudyAssistant with environment variables
                assistant = StudyAssistant(ANTHROPIC_API_KEY, OPENAI_API_KEY)
                
                with st.spinner("Processing document..."):
                    assistant.load_document(tmp_file_path)
                
                # Query input
                query = st.text_input("Ask a question about your document:")
                
                if query:
                    with st.spinner("Generating response..."):
                        response, relevant_chunks = assistant.generate_response(query)
                        
                        # Display results
                        st.subheader("Answer:")
                        st.write(response)
                        
                        with st.expander("View relevant document excerpts"):
                            for i, chunk in enumerate(relevant_chunks, 1):
                                st.markdown(f"**Excerpt {i}:**")
                                st.write(chunk)
                                st.markdown("---")
            
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            finally:
                # Clean up
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
    
    with tab2:
        st.header("Research Paper Search")
        
        # Initialize MyRag with environment variable
        rag = MyRAG(OPENAI_API_KEY)
        
        # Search query input
        search_query = st.text_input("Enter your research topic:")
        
        if search_query:
            with st.spinner("Searching for papers..."):
                papers = rag.search_papers(search_query)
                
                if papers:
                    # Process papers
                    rag.process_papers(papers)
                    
                    # Display found papers
                    st.subheader("Found Papers:")
                    for i, paper in enumerate(papers, 1):
                        st.markdown(f"**{i}. {paper.get('title')}**")
                        st.write(f"Year: {paper.get('year')}")
                        st.write(f"Citations: {paper.get('citationCount', 0)}")
                        if paper.get('abstract'):
                            with st.expander("View Abstract"):
                                st.write(paper.get('abstract'))
                        st.markdown("---")
                    
                    # Generate summary
                    with st.spinner("Generating summary..."):
                        relevant_docs = rag.find_relevant_docs(search_query)
                        summary = rag.get_summary(search_query, relevant_docs)
                        
                        st.subheader("Summary:")
                        st.write(summary)
                else:
                    st.error("No papers found for the given query.")

if __name__ == "__main__":
    main()
