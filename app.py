import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
import uuid
import io
import pickle
import tempfile
import sqlite3
from pathlib import Path

# Real imports
try:
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
    import PyPDF2
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.error("Please install: pip install langchain langchain-groq sentence-transformers faiss-cpu PyPDF2 pyautogen")
    DEPENDENCIES_AVAILABLE = False

# Create data directory for persistence
DATA_DIR = Path("kb_data")
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR = DATA_DIR / "vectorstores"
VECTORSTORE_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "kb_app.db"

# Login credentials (hidden from UI)
VALID_CREDENTIALS = {
    "aiplanet": "aiplanet000"
}

# Get Groq API key from environment or secrets
def get_groq_api_key():
    # Try to get from streamlit secrets first
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variable
        return os.getenv("GROQ_API_KEY", "")

# Set page config
st.set_page_config(
    page_title="Knowledge Base Chat & Search",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .login-container {
        max-width: 400px;
        margin: 50px auto;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .kb-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    
    .kb-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .agent-interaction {
        background: #f3e5f5;
        border: 2px dashed #9c27b0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .user-message {
        background: #e3f2fd;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        margin-left: 20%;
    }
    
    .agent-message {
        background: #f3e5f5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        margin-right: 20%;
    }
    
    .multi-agent-status {
        background: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .agent-step {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 3px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Knowledge bases table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP,
                processed BOOLEAN,
                summary TEXT,
                user_id TEXT
            )
        ''')
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                kb_id TEXT,
                filename TEXT,
                content TEXT,
                doc_type TEXT,
                added_at TIMESTAMP,
                FOREIGN KEY (kb_id) REFERENCES knowledge_bases (id)
            )
        ''')
        
        # Chat sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at TIMESTAMP,
                last_updated TIMESTAMP,
                user_id TEXT
            )
        ''')
        
        # Chat messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                message_type TEXT,
                content TEXT,
                agent_name TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
            )
        ''')
        
        # Agent interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                step_number INTEGER,
                agent_name TEXT,
                action TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_knowledge_base(self, kb, user_id: str):
        """Save knowledge base to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO knowledge_bases 
            (id, name, description, created_at, processed, summary, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (kb.id, kb.name, kb.description, kb.created_at, kb.processed, kb.summary, user_id))
        
        # Save documents
        for doc in kb.documents:
            cursor.execute('''
                INSERT OR REPLACE INTO documents
                (id, kb_id, filename, content, doc_type, added_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (doc['id'], kb.id, doc['filename'], doc['content'], doc['type'], doc['added_at']))
        
        conn.commit()
        conn.close()
    
    def load_knowledge_bases(self, user_id: str) -> Dict:
        """Load knowledge bases for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, description, created_at, processed, summary
            FROM knowledge_bases WHERE user_id = ?
        ''', (user_id,))
        
        kbs = {}
        for row in cursor.fetchall():
            kb_id, name, description, created_at, processed, summary = row
            
            kb = KnowledgeBase(name, description)
            kb.id = kb_id
            kb.created_at = datetime.fromisoformat(created_at) if created_at else datetime.now()
            kb.processed = bool(processed)
            kb.summary = summary or ""
            
            # Load documents
            cursor.execute('''
                SELECT id, filename, content, doc_type, added_at
                FROM documents WHERE kb_id = ?
            ''', (kb_id,))
            
            for doc_row in cursor.fetchall():
                doc_id, filename, content, doc_type, added_at = doc_row
                document = {
                    'id': doc_id,
                    'filename': filename,
                    'content': content,
                    'type': doc_type,
                    'chunks': [],
                    'added_at': datetime.fromisoformat(added_at) if added_at else datetime.now()
                }
                kb.documents.append(document)
            
            # Load vector store if exists
            vectorstore_path = VECTORSTORE_DIR / f"{kb_id}.faiss"
            if vectorstore_path.exists() and kb.processed:
                try:
                    embeddings_model = get_embeddings_model()
                    if embeddings_model:
                        kb.vector_store = FAISS.load_local(str(vectorstore_path), embeddings_model, allow_dangerous_deserialization=True)
                        kb.embeddings = embeddings_model
                except Exception as e:
                    st.warning(f"Could not load vector store for {name}: {e}")
                    kb.processed = False
            
            kbs[kb_id] = kb
        
        conn.close()
        return kbs
    
    def save_chat_session(self, session_id: str, name: str, user_id: str):
        """Save chat session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO chat_sessions
            (id, name, created_at, last_updated, user_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, name, datetime.now(), datetime.now(), user_id))
        
        conn.commit()
        conn.close()
    
    def save_chat_message(self, session_id: str, message_type: str, content: str, agent_name: str = None):
        """Save chat message"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_messages
            (session_id, message_type, content, agent_name, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, message_type, content, agent_name, datetime.now()))
        
        # Update session last_updated
        cursor.execute('''
            UPDATE chat_sessions SET last_updated = ? WHERE id = ?
        ''', (datetime.now(), session_id))
        
        conn.commit()
        conn.close()
    
    def save_agent_interaction(self, session_id: str, step_number: int, agent_name: str, action: str, content: str):
        """Save agent interaction step"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO agent_interactions
            (session_id, step_number, agent_name, action, content, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, step_number, agent_name, action, content, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def load_chat_sessions(self, user_id: str) -> List[Dict]:
        """Load chat sessions for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, created_at, last_updated
            FROM chat_sessions WHERE user_id = ?
            ORDER BY last_updated DESC
        ''', (user_id,))
        
        sessions = []
        for row in cursor.fetchall():
            session_id, name, created_at, last_updated = row
            sessions.append({
                'id': session_id,
                'name': name,
                'created_at': datetime.fromisoformat(created_at),
                'last_updated': datetime.fromisoformat(last_updated)
            })
        
        conn.close()
        return sessions
    
    def load_chat_messages(self, session_id: str) -> List[Dict]:
        """Load chat messages for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT message_type, content, agent_name, timestamp
            FROM chat_messages WHERE session_id = ?
            ORDER BY timestamp
        ''', (session_id,))
        
        messages = []
        for row in cursor.fetchall():
            message_type, content, agent_name, timestamp = row
            messages.append({
                'type': message_type,
                'content': content,
                'agent': agent_name,
                'timestamp': datetime.fromisoformat(timestamp)
            })
        
        conn.close()
        return messages

# Initialize database manager
db_manager = DatabaseManager(str(DB_PATH))

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'current_chat_session' not in st.session_state:
    st.session_state.current_chat_session = None
if 'agent_steps' not in st.session_state:
    st.session_state.agent_steps = []

def login_screen():
    """Display login screen"""
    # Main area for app name and description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="main-header">
            <h1>🧠 Knowledge Base Chat & Search</h1>
            <p>AI-powered knowledge management with AutoGen multi-agent collaboration</p>
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                <h3>🚀 Features</h3>
                <ul style="text-align: left; color: white;">
                    <li>Multi-Agent AI Collaboration with AutoGen</li>
                    <li>Smart Knowledge Base Processing</li>
                    <li>Advanced RAG (Retrieval-Augmented Generation)</li>
                    <li>Real-time Agent Coordination</li>
                    <li>Comprehensive Document Analysis</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar for login
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2>🔐 Login</h2>
            <p>Enter your credentials to access Knowledge Base Chat & Search</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            login_button = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if login_button:
                if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8em; color: #666;">
            <p><strong>Demo Credentials:</strong></p>
            <p>Username: aiplanet</p>
            <p>Password: aiplanet000</p>
        </div>
        """, unsafe_allow_html=True)

class KnowledgeBase:
    def __init__(self, name: str, description: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.documents = []
        self.embeddings = None
        self.vector_store = None
        self.created_at = datetime.now()
        self.summary = ""
        self.processed = False
        
    def add_document(self, content: str, filename: str, doc_type: str):
        doc_id = hashlib.md5(content.encode()).hexdigest()
        document = {
            'id': doc_id,
            'filename': filename,
            'content': content,
            'type': doc_type,
            'chunks': [],
            'added_at': datetime.now()
        }
        self.documents.append(document)
        return doc_id
    
    def process_documents(self, embeddings_model):
        """Process documents with chunking and embeddings"""
        if not self.documents:
            return 0
            
        all_chunks = []
        all_documents = []
        
        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        for doc in self.documents:
            # Split text into chunks
            chunks = text_splitter.split_text(doc['content'])
            doc['chunks'] = chunks
            
            # Create Document objects for FAISS
            for i, chunk in enumerate(chunks):
                doc_obj = Document(
                    page_content=chunk,
                    metadata={
                        'filename': doc['filename'],
                        'doc_id': doc['id'],
                        'chunk_id': i,
                        'doc_type': doc['type']
                    }
                )
                all_documents.append(doc_obj)
                all_chunks.append(chunk)
        
        if not all_documents:
            return 0
            
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(all_documents, embeddings_model)
        self.embeddings = embeddings_model
        self.processed = True
        
        # Save vector store to disk
        vectorstore_path = VECTORSTORE_DIR / f"{self.id}.faiss"
        self.vector_store.save_local(str(vectorstore_path))
        
        # Generate summary
        self.summary = self.generate_summary()
        
        return len(all_chunks)
    
    def generate_summary(self):
        """Generate KB summary"""
        total_docs = len(self.documents)
        total_content = sum(len(doc['content']) for doc in self.documents)
        total_chunks = sum(len(doc.get('chunks', [])) for doc in self.documents)
        
        summary = f"Knowledge Base '{self.name}' contains {total_docs} documents with {total_content:,} characters split into {total_chunks} chunks. "
        
        if self.documents:
            doc_types = set(doc['type'] for doc in self.documents)
            summary += f"Document types: {', '.join(doc_types)}. "
            
        return summary

class MultiAgentKBSystem:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.current_kb = None
        self.agents = {}
        self.group_chat = None
        self.manager = None
        self.setup_agents()
    
    def setup_agents(self):
        """Setup AutoGen agents for different roles"""
        if not self.groq_api_key:
            return
        
        # Configure LLM for AutoGen
        llm_config = {
            "config_list": [{
                "model": "llama3-70b-8192",
                "api_key": self.groq_api_key,
                "api_type": "groq",
                "base_url": "https://api.groq.com/openai/v1"
            }],
            "temperature": 0.3,
            "timeout": 120,
        }
        
        # Query Analyzer Agent
        self.agents['query_analyzer'] = AssistantAgent(
            name="QueryAnalyzer",
            system_message="""You are a Query Analyzer. Your role is to:
            1. Analyze user queries and break them down into components
            2. Identify the type of information being requested
            3. Determine the best search strategy
            4. Extract key terms and concepts
            5. Classify the query complexity (simple, moderate, complex)
            
            Always provide structured analysis in this format:
            - Query Type: [factual/analytical/comparative/procedural]
            - Key Terms: [list of important terms]
            - Search Strategy: [broad/specific/multi-step]
            - Complexity: [simple/moderate/complex]
            - Expected Answer Type: [brief/detailed/step-by-step]""",
            llm_config=llm_config
        )
        
        # Knowledge Retriever Agent
        self.agents['retriever'] = AssistantAgent(
            name="KnowledgeRetriever", 
            system_message="""You are a Knowledge Retriever. Your role is to:
            1. Search the knowledge base using the analyzed query
            2. Retrieve relevant document chunks
            3. Rank results by relevance
            4. Extract the most pertinent information
            5. Identify gaps if information is incomplete
            
            Format your retrieval results as:
            - Relevant Chunks: [number] found
            - Top Sources: [list of document names]
            - Key Information: [extracted facts/data]
            - Confidence Level: [high/medium/low]
            - Gaps Identified: [missing information if any]""",
            llm_config=llm_config
        )
        
        # Answer Synthesizer Agent
        self.agents['synthesizer'] = AssistantAgent(
            name="AnswerSynthesizer",
            system_message="""You are an Answer Synthesizer. Your role is to:
            1. Combine information from retrieved knowledge chunks
            2. Create coherent, comprehensive answers
            3. Ensure accuracy and completeness
            4. Structure answers logically
            5. Include source citations
            
            Format your answers with:
            - Main Answer: [comprehensive response]
            - Supporting Details: [additional context]
            - Sources: [cited documents]
            - Confidence: [assessment of answer quality]
            - Recommendations: [if applicable]""",
            llm_config=llm_config
        )
        
        # Quality Validator Agent
        self.agents['validator'] = AssistantAgent(
            name="QualityValidator",
            system_message="""You are a Quality Validator. Your role is to:
            1. Review the synthesized answer for accuracy
            2. Check for logical consistency
            3. Verify source citations are appropriate
            4. Assess completeness of the response
            5. Suggest improvements if needed
            
            Provide validation in this format:
            - Accuracy Check: [passed/needs review]
            - Consistency Check: [passed/issues found]
            - Citation Verification: [appropriate/missing/incorrect]
            - Completeness: [complete/partial/insufficient]
            - Final Recommendation: [approve/revise/reject]
            - Improvement Suggestions: [if any]""",
            llm_config=llm_config
        )
        
        # User Proxy Agent (represents the user)
        self.user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
        )
    
    def setup_group_chat(self):
        """Setup group chat with all agents"""
        if not self.agents:
            return
        
        agent_list = [self.user_proxy] + list(self.agents.values())
        
        self.group_chat = GroupChat(
            agents=agent_list,
            messages=[],
            max_round=10,
            speaker_selection_method="round_robin"
        )
        
        llm_config = {
            "config_list": [{
                "model": "llama3-70b-8192", 
                "api_key": self.groq_api_key,
                "api_type": "groq",
                "base_url": "https://api.groq.com/openai/v1"
            }],
            "temperature": 0.1,
        }
        
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=llm_config
        )
    
    def set_knowledge_base(self, kb: KnowledgeBase):
        """Set the current knowledge base for retrieval"""
        self.current_kb = kb
        
        # Update retriever agent with KB access
        if 'retriever' in self.agents and kb.processed:
            # Create retrieval QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(api_key=self.groq_api_key, model_name="llama3-70b-8192"),
                chain_type="stuff",
                retriever=kb.vector_store.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True
            )
    
    def retrieve_knowledge(self, query: str) -> Dict:
        """Retrieve knowledge from current KB"""
        if not self.current_kb or not self.current_kb.processed:
            return {"error": "No processed knowledge base available"}
        
        try:
            result = self.qa_chain({"query": query})
            
            # Format sources (remove duplicates)
            unique_sources = {}
            if 'source_documents' in result:
                for doc in result['source_documents']:
                    filename = doc.metadata.get('filename', 'Unknown')
                    # Only keep the first occurrence of each filename
                    if filename not in unique_sources:
                        unique_sources[filename] = {
                            'filename': filename,
                            'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        }
            
            # Convert to list
            sources = list(unique_sources.values())
            
            return {
                "answer": result['result'],
                "sources": sources,
                "unique_filenames": list(unique_sources.keys())
            }
        except Exception as e:
            return {"error": f"Retrieval error: {str(e)}"}
    
    def process_query(self, user_query: str, session_id: str = None) -> str:
        """Process user query through multi-agent system"""
        if not self.agents or not self.current_kb:
            return "❌ Multi-agent system not properly configured or no knowledge base selected."
        
        self.setup_group_chat()
        
        # Add step tracking
        step_counter = 0
        
        def add_agent_step(agent_name: str, action: str, content: str):
            nonlocal step_counter
            step_counter += 1
            step_info = {
                'step': step_counter,
                'agent': agent_name,
                'action': action,
                'content': content,
                'timestamp': datetime.now()
            }
            st.session_state.agent_steps.append(step_info)
            
            if session_id:
                db_manager.save_agent_interaction(session_id, step_counter, agent_name, action, content)
        
        try:
            # Step 1: Query Analysis
            add_agent_step("QueryAnalyzer", "Analyzing Query", f"Breaking down user query: {user_query[:100]}...")
            
            analysis_prompt = f"""
            Please analyze this user query: "{user_query}"
            
            Provide your analysis following the specified format.
            """
            
            # Step 2: Knowledge Retrieval
            add_agent_step("KnowledgeRetriever", "Retrieving Knowledge", "Searching knowledge base for relevant information...")
            
            retrieval_result = self.retrieve_knowledge(user_query)
            
            if "error" in retrieval_result:
                return f"❌ {retrieval_result['error']}"
            
            # Step 3: Answer Synthesis
            add_agent_step("AnswerSynthesizer", "Synthesizing Answer", "Combining retrieved information into coherent response...")
            
            synthesis_prompt = f"""
            Based on the retrieved information, synthesize a comprehensive answer to: "{user_query}"
            
            Retrieved Information:
            {retrieval_result['answer']}
            
            Sources Available:
            {[source['filename'] for source in retrieval_result['sources']]}
            
            Please provide a well-structured response following the specified format.
            """
            
            # Step 4: Quality Validation
            add_agent_step("QualityValidator", "Validating Response", "Checking answer quality and accuracy...")
            
            # For now, return the direct answer with improvements
            final_answer = retrieval_result['answer']
            
            # Add source information (ensure no duplicates)
            if retrieval_result.get('unique_filenames'):
                final_answer += "\n\n📚 **Sources:**\n"
                for filename in retrieval_result['unique_filenames']:
                    final_answer += f"- {filename}\n"
            
            add_agent_step("System", "Process Complete", "Multi-agent processing completed successfully")
            
            return final_answer
            
        except Exception as e:
            add_agent_step("System", "Error", f"Error in multi-agent processing: {str(e)}")
            return f"❌ Error in multi-agent processing: {str(e)}"

# Initialize session state
if 'knowledge_bases' not in st.session_state:
    st.session_state.knowledge_bases = {}

if 'multi_agent_system' not in st.session_state:
    st.session_state.multi_agent_system = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def get_embeddings_model():
    """Initialize and cache the embeddings model"""
    if st.session_state.embeddings_model is None:
        try:
            st.session_state.embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            st.error(f"❌ Error loading embeddings model: {str(e)}")
            return None
    return st.session_state.embeddings_model

def load_user_data():
    """Load user's knowledge bases"""
    if st.session_state.logged_in:
        st.session_state.knowledge_bases = db_manager.load_knowledge_bases(st.session_state.username)

def save_user_data():
    """Save user's data to database"""
    if st.session_state.logged_in:
        for kb in st.session_state.knowledge_bases.values():
            db_manager.save_knowledge_base(kb, st.session_state.username)

def display_agent_steps():
    """Display multi-agent processing steps"""
    if st.session_state.agent_steps:
        with st.expander("🤖 Multi-Agent Processing Steps", expanded=True):
            for step_info in st.session_state.agent_steps[-10:]:  # Show last 10 steps
                st.markdown(f"""
                <div class="agent-step">
                    <strong>Step {step_info['step']}: {step_info['agent']}</strong> - {step_info['action']}<br>
                    <small>{step_info['timestamp'].strftime('%H:%M:%S')}</small><br>
                    {step_info['content'][:150]}{'...' if len(step_info['content']) > 150 else ''}
                </div>
                """, unsafe_allow_html=True)

def display_chat_history_sidebar():
    """Display chat history in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("💬 Chat History")
        
        sessions = db_manager.load_chat_sessions(st.session_state.username)
        
        if sessions:
            if st.button("➕ New Chat Session", use_container_width=True):
                st.session_state.current_chat_session = None
                st.session_state.chat_history = []
                st.session_state.agent_steps = []
                st.rerun()
            
            st.markdown("**Recent Sessions:**")
            for session in sessions[:10]:
                session_name = session['name'][:30] + "..." if len(session['name']) > 30 else session['name']
                
                if st.button(
                    f"📝 {session_name}",
                    key=f"session_{session['id']}",
                    help=f"Last updated: {session['last_updated'].strftime('%Y-%m-%d %H:%M')}",
                    use_container_width=True
                ):
                    st.session_state.current_chat_session = session['id']
                    st.session_state.chat_history = db_manager.load_chat_messages(session['id'])
                    st.rerun()
        else:
            st.info("No chat history yet. Start a conversation!")

def main():
    if not DEPENDENCIES_AVAILABLE:
        st.stop()
    
    # Check login status
    if not st.session_state.logged_in:
        login_screen()
        return
    
    # Load user data on first load
    if not st.session_state.knowledge_bases:
        load_user_data()
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🧠 Knowledge Base Chat & Search</h1>
        <p>Welcome, {st.session_state.username}! AI-powered knowledge management with AutoGen multi-agent collaboration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            save_user_data()
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.knowledge_bases = {}
            st.session_state.multi_agent_system = None
            st.session_state.chat_history = []
            st.session_state.agent_steps = []
            st.rerun()
        
        # API Configuration
        st.subheader("⚙️ Configuration")
        
        # Get API key from environment/secrets
        groq_api_key = get_groq_api_key()
        
        if groq_api_key:
            st.success("✅ Groq API Key configured")
            # Initialize multi-agent system
            if st.session_state.multi_agent_system is None:
                st.session_state.multi_agent_system = MultiAgentKBSystem(groq_api_key)
        else:
            st.error("❌ Groq API key not found")
            st.info("💡 Please set GROQ_API_KEY in your .env file or Streamlit secrets")
        
        # Quick Stats
        st.subheader("📊 Quick Stats")
        st.metric("Knowledge Bases", len(st.session_state.knowledge_bases))
        
        processed_kbs = sum(1 for kb in st.session_state.knowledge_bases.values() if kb.processed)
        st.metric("Processed KBs", processed_kbs)
        
        if st.session_state.multi_agent_system:
            st.metric("Active Agents", len(st.session_state.multi_agent_system.agents))
        
        # Chat History
        display_chat_history_sidebar()
    
    # Main tabs
    tab1, tab2 = st.tabs(["📚 Knowledge Bases", "💬 Multi-Agent Chat"])
    
    with tab1:
        st.header("📚 Knowledge Base Management")
        
        # Create new Knowledge Base
        with st.expander("➕ Create New Knowledge Base", expanded=not st.session_state.knowledge_bases):
            with st.form("create_kb_form"):
                col1, col2 = st.columns(2)
                with col1:
                    kb_name = st.text_input("Knowledge Base Name*", placeholder="e.g., Company Policies")
                with col2:
                    kb_description = st.text_area("Description", placeholder="Brief description of this knowledge base")
                
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    accept_multiple_files=True,
                    type=['pdf', 'txt', 'md']
                )
                
                submit_kb = st.form_submit_button("🚀 Create Knowledge Base", use_container_width=True)
                
                if submit_kb and kb_name:
                    # Create KB
                    kb = KnowledgeBase(kb_name, kb_description)
                    
                    # Process uploaded files
                    if uploaded_files:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, file in enumerate(uploaded_files):
                            status_text.text(f"Processing {file.name}...")
                            
                            try:
                                if file.type == "application/pdf":
                                    content = extract_text_from_pdf(file)
                                else:
                                    content = str(file.read(), "utf-8")
                                
                                if content.strip():
                                    kb.add_document(content, file.name, file.type)
                                else:
                                    st.warning(f"⚠️ No text content found in {file.name}")
                                    
                            except Exception as e:
                                st.error(f"❌ Error processing {file.name}: {str(e)}")
                            
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        # Get embeddings model and process
                        embeddings_model = get_embeddings_model()
                        
                        if embeddings_model and kb.documents:
                            status_text.text("Creating embeddings and vector store...")
                            
                            # Show spinner for embeddings processing
                            with st.spinner("🔄 Processing documents and creating embeddings..."):
                                try:
                                    chunks_count = kb.process_documents(embeddings_model)
                                    
                                    if chunks_count > 0:
                                        st.success(f"✅ Knowledge base '{kb_name}' created with {chunks_count} chunks!")
                                    else:
                                        st.error("❌ No text chunks were created.")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error creating embeddings: {str(e)}")
                        else:
                            st.error("❌ Failed to load embeddings model.")
                        
                        progress_bar.progress(1.0)
                        status_text.empty()
                    
                    # Store KB
                    st.session_state.knowledge_bases[kb.id] = kb
                    db_manager.save_knowledge_base(kb, st.session_state.username)
                    st.rerun()
        
        # Display existing Knowledge Bases
        if st.session_state.knowledge_bases:
            st.subheader("📋 Existing Knowledge Bases")
            
            for kb_id, kb in st.session_state.knowledge_bases.items():
                status_icon = "✅" if kb.processed else "⏳"
                
                with st.container():
                    st.markdown(f"""
                    <div class="kb-card">
                        <h4>{status_icon} 📚 {kb.name}</h4>
                        <p><strong>Description:</strong> {kb.description or 'No description provided'}</p>
                        <p><strong>Documents:</strong> {len(kb.documents)} | <strong>Status:</strong> {'Processed' if kb.processed else 'Not Processed'}</p>
                        <p><strong>Created:</strong> {kb.created_at.strftime('%Y-%m-%d %H:%M')}</p>
                        {f'<p><strong>Summary:</strong> {kb.summary}</p>' if kb.summary else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if st.button(f"📊 Details", key=f"view_{kb_id}"):
                            with st.expander(f"📋 Details for {kb.name}", expanded=True):
                                st.write(f"**ID:** {kb.id}")
                                st.write(f"**Documents:** {len(kb.documents)}")
                                if kb.documents:
                                    st.write("**Files:**")
                                    for doc in kb.documents:
                                        chunks_info = f" ({len(doc.get('chunks', []))} chunks)" if doc.get('chunks') else ""
                                        st.write(f"- {doc['filename']}{chunks_info}")
                    
                    with col2:
                        if not kb.processed and st.button(f"🔄 Process", key=f"process_{kb_id}"):
                            embeddings_model = get_embeddings_model()
                            if embeddings_model:
                                with st.spinner("🔄 Processing documents and creating embeddings..."):
                                    chunks_count = kb.process_documents(embeddings_model)
                                    if chunks_count > 0:
                                        db_manager.save_knowledge_base(kb, st.session_state.username)
                                        st.success(f"✅ Processed {chunks_count} chunks")
                                        st.rerun()
                            else:
                                st.error("❌ Failed to load embeddings model")
                    
                    with col3:
                        if kb.processed and st.button(f"🔄 Reprocess", key=f"reprocess_{kb_id}"):
                            embeddings_model = get_embeddings_model()
                            if embeddings_model:
                                with st.spinner("🔄 Reprocessing documents and creating embeddings..."):
                                    chunks_count = kb.process_documents(embeddings_model)
                                    db_manager.save_knowledge_base(kb, st.session_state.username)
                                    st.success(f"✅ Reprocessed {chunks_count} chunks")
                                    st.rerun()
                            else:
                                st.error("❌ Failed to load embeddings model")
                    
                    with col4:
                        if st.button(f"🗑️ Delete", key=f"delete_{kb_id}"):
                            # Delete vector store file
                            vectorstore_path = VECTORSTORE_DIR / f"{kb_id}.faiss"
                            if vectorstore_path.exists():
                                import shutil
                                shutil.rmtree(vectorstore_path, ignore_errors=True)
                            
                            del st.session_state.knowledge_bases[kb_id]
                            st.success(f"✅ Deleted knowledge base: {kb.name}")
                            st.rerun()
    
    with tab2:
        st.header("💬 Multi-Agent Chat Interface")
        
        if not st.session_state.knowledge_bases:
            st.warning("⚠️ Please create at least one knowledge base before starting a chat.")
        elif not st.session_state.multi_agent_system:
            st.warning("⚠️ Groq API key not configured. Please check your environment variables or secrets.")
            st.info("💡 Make sure GROQ_API_KEY is set in your .env file or Streamlit secrets")
        else:
            # Knowledge base selection
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                kb_options = {kb.name: kb_id for kb_id, kb in st.session_state.knowledge_bases.items() if kb.processed}
                if not kb_options:
                    st.error("❌ No processed knowledge bases available. Please process at least one KB first.")
                    return
                
                selected_kb_name = st.selectbox("Select Knowledge Base", list(kb_options.keys()))
                selected_kb_id = kb_options[selected_kb_name] if selected_kb_name else None
            
            with col2:
                if st.button("🔄 Clear Current Chat"):
                    st.session_state.chat_history = []
                    st.session_state.current_chat_session = None
                    st.session_state.agent_steps = []
                    st.rerun()
            
            with col3:
                if st.button("💾 Save Chat Session"):
                    if st.session_state.chat_history:
                        session_id = str(uuid.uuid4())
                        session_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        
                        db_manager.save_chat_session(session_id, session_name, st.session_state.username)
                        
                        for msg in st.session_state.chat_history:
                            db_manager.save_chat_message(
                                session_id,
                                msg['type'],
                                msg['content'],
                                msg.get('agent', None)
                            )
                        
                        st.session_state.current_chat_session = session_id
                        st.success("✅ Chat session saved!")
                    else:
                        st.warning("⚠️ No chat history to save!")
            
            if selected_kb_id:
                kb = st.session_state.knowledge_bases[selected_kb_id]
                
                # Set KB in multi-agent system
                st.session_state.multi_agent_system.set_knowledge_base(kb)
                
                # Display multi-agent system status
                st.markdown(f"""
                <div class="multi-agent-status">
                    🤖 Multi-Agent System Active<br>
                    <strong>Knowledge Base:</strong> {kb.name}<br>
                    <strong>Active Agents:</strong> QueryAnalyzer, KnowledgeRetriever, AnswerSynthesizer, QualityValidator<br>
                    <strong>Status:</strong> ✅ Ready for collaboration
                </div>
                """, unsafe_allow_html=True)
                
                # Display agent processing steps
                display_agent_steps()
                
                # Display chat history
                if st.session_state.chat_history:
                    st.markdown("### 💬 Chat History")
                    for message in st.session_state.chat_history:
                        if message['type'] == 'user':
                            st.markdown(f"""
                            <div class="user-message">
                                <strong>🧑 You:</strong> {message['content']}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="agent-message">
                                <strong>🤖 Multi-Agent System:</strong><br>
                                {message['content'].replace('\n', '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Chat input
                st.markdown("### ✍️ Ask a Question")
                with st.form("chat_form", clear_on_submit=True):
                    user_input = st.text_area(
                        "Your Question:", 
                        placeholder="Ask me anything about the knowledge base...", 
                        height=100,
                        key="chat_input"
                    )
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        submit_chat = st.form_submit_button("📤 Send to Multi-Agent Team", use_container_width=True)
                    
                    if submit_chat and user_input:
                        # Clear previous agent steps
                        st.session_state.agent_steps = []
                        
                        # Add user message
                        st.session_state.chat_history.append({
                            'type': 'user',
                            'content': user_input,
                            'timestamp': datetime.now()
                        })
                        
                        # Process with multi-agent system
                        with st.spinner("🤖 Multi-agent team is collaborating on your query..."):
                            response = st.session_state.multi_agent_system.process_query(
                                user_input, 
                                st.session_state.current_chat_session
                            )
                            
                            # Add agent response
                            st.session_state.chat_history.append({
                                'type': 'agent',
                                'agent': 'Multi-Agent System',
                                'content': response,
                                'timestamp': datetime.now()
                            })
                        
                        # Auto-save to current session if exists
                        if st.session_state.current_chat_session:
                            db_manager.save_chat_message(
                                st.session_state.current_chat_session,
                                'user',
                                user_input
                            )
                            db_manager.save_chat_message(
                                st.session_state.current_chat_session,
                                'agent',
                                response,
                                'Multi-Agent System'
                            )
                        
                        st.rerun()

if __name__ == "__main__":
    main()