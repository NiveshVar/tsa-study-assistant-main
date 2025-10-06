import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from src.rag_system import PDFNotesRAG

class StudyAssistantApp:
    def __init__(self, root):
        self.root = root
        self.rag_system = None
        self.setup_complete = False
        
        self.setup_ui()
        self.setup_system_in_background()
    
    def setup_ui(self):
        """Setup the user interface"""
        self.root.title("🎓 Study Notes Assistant")
        self.root.geometry("800x700")
        self.root.configure(bg='white')
        
        # Header
        header_frame = tk.Frame(self.root, bg='white', pady=10)
        header_frame.pack(fill='x')
        
        title_label = tk.Label(
            header_frame, 
            text="🎓 Study Notes Assistant", 
            font=('Arial', 20, 'bold'),
            bg='white',
            fg='#2E86AB'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Ask questions about your TSA study notes",
            font=('Arial', 12),
            bg='white',
            fg='#666666'
        )
        subtitle_label.pack()
        
        # Separator
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=20, pady=10)
        
        # Question Input
        input_frame = tk.Frame(self.root, bg='white', pady=10)
        input_frame.pack(fill='x', padx=20)
        
        tk.Label(
            input_frame,
            text="💬 Ask a question:",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#333333'
        ).pack(anchor='w')
        
        self.question_entry = tk.Entry(
            input_frame,
            font=('Arial', 12),
            width=70
        )
        self.question_entry.pack(fill='x', pady=5)
        self.question_entry.bind('<Return>', lambda e: self.get_answer())
        
        # Buttons
        button_frame = tk.Frame(self.root, bg='white', pady=10)
        button_frame.pack(fill='x', padx=20)
        
        self.answer_button = tk.Button(
            button_frame,
            text="🚀 Get Answer",
            font=('Arial', 12, 'bold'),
            bg='#2E86AB',
            fg='white',
            command=self.get_answer,
            width=15,
            state='disabled'
        )
        self.answer_button.pack(side='left', padx=(0, 10))
        
        tk.Button(
            button_frame,
            text="🗑️ Clear",
            font=('Arial', 12),
            bg='#666666',
            fg='white',
            command=self.clear_chat,
            width=10
        ).pack(side='left')
        
        # Status
        self.status_label = tk.Label(
            self.root,
            text="🔄 Loading your study notes...",
            font=('Arial', 10),
            bg='white',
            fg='#666666'
        )
        self.status_label.pack(pady=5)
        
        # Separator
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=20, pady=10)
        
        # Answer Display
        answer_frame = tk.Frame(self.root, bg='white')
        answer_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        tk.Label(
            answer_frame,
            text="Answer:",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#333333'
        ).pack(anchor='w')
        
        self.answer_text = scrolledtext.ScrolledText(
            answer_frame,
            font=('Arial', 11),
            wrap=tk.WORD,
            width=80,
            height=15,
            bg='#FFFFFF',
            fg='#000000'
        )
        self.answer_text.pack(fill='both', expand=True, pady=5)
        self.answer_text.config(state='disabled')
        
        # Sources Display
        tk.Label(
            answer_frame,
            text="📚 Sources:",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#333333'
        ).pack(anchor='w', pady=(10, 5))
        
        self.sources_text = scrolledtext.ScrolledText(
            answer_frame,
            font=('Arial', 10),
            wrap=tk.WORD,
            width=80,
            height=6,
            bg='#F8F9FA',
            fg='#000000'
        )
        self.sources_text.pack(fill='x', pady=5)
        self.sources_text.config(state='disabled')
    
    def setup_system_in_background(self):
        """Setup the RAG system in background thread"""
        def setup():
            try:
                self.rag_system = PDFNotesRAG("./data")
                
                # Load PDFs
                self.root.after(0, lambda: self.status_label.config(text="📥 Loading PDFs..."))
                documents = self.rag_system.load_pdfs()
                
                # Chunk documents
                self.root.after(0, lambda: self.status_label.config(text="✂️ Processing text..."))
                chunks = self.rag_system.chunk_documents()
                
                # Setup vector store
                self.root.after(0, lambda: self.status_label.config(text="🔢 Creating search index..."))
                vector_store = self.rag_system.setup_vector_store()
                
                # Setup Gemini AI
                self.root.after(0, lambda: self.status_label.config(text="🤖 Connecting to AI..."))
                api_key = "AIzaSyBpCOIHt6VO-OVj9pN8_PZC6oKtvlE14FI"
                ai_ready = self.rag_system.setup_gemini_llm(api_key)
                
                self.setup_complete = True
                
                # Update UI
                self.root.after(0, self.setup_complete_ui)
                
            except Exception as e:
                self.root.after(0, lambda: self.setup_failed_ui(str(e)))
        
        # Start setup in background thread
        thread = threading.Thread(target=setup)
        thread.daemon = True
        thread.start()
    
    def setup_complete_ui(self):
        """Update UI when setup is complete"""
        self.status_label.config(text="✅ System ready! Ask your question.", fg='green')
        self.answer_button.config(state='normal', bg='#28a745')
        self.question_entry.focus()
    
    def setup_failed_ui(self, error):
        """Update UI when setup fails"""
        self.status_label.config(text=f"❌ Setup failed: {error}", fg='red')
    
    def get_answer(self):
        """Get answer for the question"""
        if not self.setup_complete:
            messagebox.showwarning("System Not Ready", "Please wait for the system to finish loading.")
            return
        
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("Empty Question", "Please enter a question.")
            return
        
        # Disable button during processing
        self.answer_button.config(state='disabled', bg='#6c757d')
        self.status_label.config(text="🔍 Searching through your notes...", fg='blue')
        
        def process_question():
            try:
                result = self.rag_system.ask_question(question)
                
                # Update UI in main thread
                self.root.after(0, lambda: self.display_answer(question, result))
                
            except Exception as e:
                self.root.after(0, lambda: self.display_error(str(e)))
            finally:
                self.root.after(0, self.enable_answer_button)
        
        # Process in background thread
        thread = threading.Thread(target=process_question)
        thread.daemon = True
        thread.start()
    
    def display_answer(self, question, result):
        """Display the answer in the UI"""
        # Clear previous content
        self.answer_text.config(state='normal')
        self.answer_text.delete(1.0, tk.END)
        self.sources_text.config(state='normal')
        self.sources_text.delete(1.0, tk.END)
        
        # Display question
        self.answer_text.insert(tk.END, f"❓ Question: {question}\n\n")
        self.answer_text.insert(tk.END, "🎯 Answer:\n")
        
        # Display answer
        if result and "error" not in result:
            self.answer_text.insert(tk.END, f"{result['answer']}\n\n")
            
            # Display sources
            self.sources_text.insert(tk.END, "Sources:\n")
            for i, doc in enumerate(result["sources"], 1):
                unit_name = doc.metadata.get('unit', 'Unknown')
                page_num = doc.metadata.get('page', 0) + 1
                self.sources_text.insert(tk.END, f"• Source {i}: {unit_name} - Page {page_num}\n")
        else:
            self.answer_text.insert(tk.END, "Sorry, I couldn't find an answer to that question.\n")
        
        # Make text read-only
        self.answer_text.config(state='disabled')
        self.sources_text.config(state='disabled')
        
        self.status_label.config(text="✅ Answer ready!", fg='green')
    
    def display_error(self, error):
        """Display error message"""
        self.answer_text.config(state='normal')
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, f"❌ Error: {error}")
        self.answer_text.config(state='disabled')
        
        self.sources_text.config(state='normal')
        self.sources_text.delete(1.0, tk.END)
        self.sources_text.config(state='disabled')
        
        self.status_label.config(text="❌ Error occurred", fg='red')
    
    def enable_answer_button(self):
        """Re-enable the answer button"""
        self.answer_button.config(state='normal', bg='#2E86AB')
    
    def clear_chat(self):
        """Clear the chat interface"""
        self.question_entry.delete(0, tk.END)
        
        self.answer_text.config(state='normal')
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.config(state='disabled')
        
        self.sources_text.config(state='normal')
        self.sources_text.delete(1.0, tk.END)
        self.sources_text.config(state='disabled')
        
        self.status_label.config(text="✅ System ready! Ask your question.", fg='green')
        self.question_entry.focus()

def main():
    # Create and run the application
    root = tk.Tk()
    app = StudyAssistantApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()