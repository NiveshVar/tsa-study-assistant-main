import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
import shutil

class PDFNotesRAG:
    def __init__(self, notes_directory, persist_directory="./chroma_db"):
        self.notes_directory = notes_directory
        self.persist_directory = persist_directory
        self.documents = []
        self.chunks = []
        self.vector_store = None
        self.embeddings = None
        self.genai_model = None
        
    def load_pdfs(self):
        """Load all PDF files from the data directory"""
        print("Loading PDF notes...")
        
        for filename in os.listdir(self.notes_directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.notes_directory, filename)
                print(f"Loading: {filename}")
                
                try:
                    loader = PyPDFLoader(file_path)
                    pdf_documents = loader.load()
                    
                    # Add metadata to identify which unit it came from
                    for doc in pdf_documents:
                        doc.metadata['unit'] = filename.replace('.pdf', '')
                        doc.metadata['source'] = filename
                    
                    self.documents.extend(pdf_documents)
                    print(f"✓ Successfully loaded {filename} ({len(pdf_documents)} pages)")
                    
                except Exception as e:
                    print(f"✗ Error loading {filename}: {e}")
        
        print(f"\nTotal documents loaded: {len(self.documents)}")
        return self.documents

    def chunk_documents(self, chunk_size=1000, chunk_overlap=200):
        """Split documents into smaller chunks for better retrieval"""
        print("\nChunking documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"✓ Created {len(self.chunks)} chunks from {len(self.documents)} pages")
        
        return self.chunks

    def setup_vector_store(self):
        """Create embeddings and vector store"""
        print("\nSetting up vector store...")
        
        # Initialize embedding model (local & free)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✓ Embedding model loaded")
        
        # Create or load vector store
        if os.path.exists(self.persist_directory):
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store...")
            self.vector_store = Chroma.from_documents(
                documents=self.chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("✓ Vector store created and persisted")
        
        return self.vector_store

    def setup_gemini_llm(self, api_key):
        """Setup Google Gemini LLM with correct model names"""
        try:
            # Configure the API directly
            genai.configure(api_key=api_key)
            
            # Use the latest model name - gemini-2.0-flash is fast and free
            self.genai_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Test the connection with a simple prompt
            test_response = self.genai_model.generate_content("Say 'TEST OK' in one word.")
            
            if test_response and test_response.text:
                print("✓ Google Gemini AI loaded successfully!")
                print(f"✓ Using model: gemini-2.0-flash")
                return True
            else:
                print("✗ Test response was empty")
                return False
                
        except Exception as e:
            print(f"✗ Error setting up Gemini: {e}")
            # Try fallback model
            try:
                print("Trying fallback model: gemini-2.0-flash-lite")
                self.genai_model = genai.GenerativeModel('gemini-2.0-flash-lite')
                test_response = self.genai_model.generate_content("Test")
                if test_response.text:
                    print("✓ Fallback model loaded successfully!")
                    return True
            except Exception as e2:
                print(f"✗ Fallback also failed: {e2}")
            
            return False

    def ask_question(self, question, k=5):
        """Ask a question and get a proper AI-generated answer"""
        if self.vector_store is None:
            return {"error": "Please setup vector store first!"}
        
        # Retrieve relevant chunks
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(question)
        
        # Combine context from all relevant chunks
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        if self.genai_model and context.strip():
            # Create prompt for Gemini
            prompt = f"""You are a helpful study assistant. Answer the question based ONLY on the provided study notes.

STUDY NOTES CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer using ONLY the information from the study notes above
2. If the information isn't in the notes, say "I cannot find this information in my study notes"
3. Write in clear, proper sentences with good paragraph structure
4. Be comprehensive but concise
5. Do not add any external knowledge or information not in the notes
6. Format your answer with clear paragraphs and bullet points if helpful

ANSWER:"""
            
            try:
                # Generate answer with Gemini
                response = self.genai_model.generate_content(prompt)
                if response and response.text:
                    coherent_answer = response.text
                else:
                    coherent_answer = "I couldn't generate an answer. Here's the relevant context from your notes:\n\n" + context
            except Exception as e:
                coherent_answer = f"Error generating AI answer: {str(e)}\n\nRelevant context from notes:\n{context}"
        else:
            # Fallback: simple context combination
            coherent_answer = f"Relevant information from your notes:\n\n{context}"
        
        return {
            "question": question,
            "answer": coherent_answer,
            "sources": relevant_docs
        }

# Test the AI connection
if __name__ == "__main__":
    # Initialize the system
    rag_system = PDFNotesRAG("./data")
    
    # Step 1: Load all PDFs
    documents = rag_system.load_pdfs()
    
    # Step 2: Chunk the documents
    chunks = rag_system.chunk_documents()
    
    # Step 3: Create vector store
    vector_store = rag_system.setup_vector_store()
    
    # Step 4: Setup Gemini AI with your API key
    print("\n" + "="*70)
    print("SETTING UP GOOGLE GEMINI AI")
    print("="*70)
    
    api_key = "AIzaSyBpCOIHt6VO-OVj9pN8_PZC6oKtvlE14FI"
    if rag_system.setup_gemini_llm(api_key):
        print("🎉 AI-powered Q&A ready!")
        
        # Test with AI answers
        test_questions = [
            "What is natural language processing?",
            "Explain RNN in simple terms",
        ]
        
        for question in test_questions:
            print(f"\n🧠 Question: {question}")
            result = rag_system.ask_question(question)
            if result and "error" not in result:
                print(f"✅ AI Answer: {result['answer']}")
                print(f"📚 Sources: Units {list(set([doc.metadata['unit'] for doc in result['sources']]))}")
                print("="*70)
    else:
        print("❌ AI setup failed. Using retrieval-only mode.")