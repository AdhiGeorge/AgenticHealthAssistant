"""SQLite-based memory and agent state management utility."""
import sqlite3
import os
import json
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'results', 'analysis_memory.db')

# Ensure the results directory exists before any DB operation
results_dir = os.path.dirname(DB_PATH)
os.makedirs(results_dir, exist_ok=True)

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def init_db():
    """Initialize the database tables for memory and agent state."""
    with get_conn() as conn:
        c = conn.cursor()
        
        # Analysis table
        c.execute('''CREATE TABLE IF NOT EXISTS analysis (
            session_id TEXT PRIMARY KEY,
            original TEXT,
            analysis TEXT,
            status TEXT,
            score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Agent state table
        c.execute('''CREATE TABLE IF NOT EXISTS agent_state (
            session_id TEXT,
            agent TEXT,
            state TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, agent)
        )''')
        
        # Conversations table
        c.execute('''CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            session_id TEXT,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES analysis (session_id)
        )''')
        
        # Chat messages table
        c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT,
            role TEXT,  -- 'user' or 'assistant'
            content TEXT,
            agent_used TEXT,  -- which agent was used to answer
            confidence_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
        )''')
        
        # Context embeddings table (for vector search if needed)
        c.execute('''CREATE TABLE IF NOT EXISTS context_embeddings (
            embedding_id TEXT PRIMARY KEY,
            session_id TEXT,
            content_type TEXT,  -- 'original', 'analysis', 'message'
            content_hash TEXT,
            embedding_data TEXT,  -- JSON encoded embedding
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES analysis (session_id)
        )''')
        
        # Create indexes for better performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat_messages(conversation_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_context_embeddings_session_id ON context_embeddings(session_id)')

def save_analysis(session_id, original, analysis, status, score):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute('''REPLACE INTO analysis (session_id, original, analysis, status, score)
                     VALUES (?, ?, ?, ?, ?)''', (session_id, original, analysis, status, score))

def get_analysis(session_id):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute('SELECT original, analysis, status, score FROM analysis WHERE session_id=?', (session_id,))
        return c.fetchone()

def save_agent_state(session_id, agent, state):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute('''REPLACE INTO agent_state (session_id, agent, state)
                     VALUES (?, ?, ?)''', (session_id, agent, state))

def get_agent_state(session_id, agent):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute('SELECT state FROM agent_state WHERE session_id=? AND agent=?', (session_id, agent))
        row = c.fetchone()
        return row[0] if row else None

# Conversation management functions
def create_conversation(session_id, title=None):
    """Create a new conversation for a session."""
    conversation_id = f"conv_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not title:
        title = f"Conversation for session {session_id[:8]}"
    
    with get_conn() as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO conversations (conversation_id, session_id, title)
                     VALUES (?, ?, ?)''', (conversation_id, session_id, title))
    
    return conversation_id

def get_conversation(conversation_id):
    """Get conversation details."""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute('''SELECT conversation_id, session_id, title, created_at, last_updated 
                     FROM conversations WHERE conversation_id=?''', (conversation_id,))
        return c.fetchone()

def get_conversations_for_session(session_id):
    """Get all conversations for a session."""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute('''SELECT conversation_id, title, created_at, last_updated 
                     FROM conversations WHERE session_id=? ORDER BY created_at DESC''', (session_id,))
        return c.fetchall()

def save_chat_message(conversation_id, role, content, agent_used=None, confidence_score=None):
    """Save a chat message."""
    message_id = f"msg_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    with get_conn() as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO chat_messages (message_id, conversation_id, role, content, agent_used, confidence_score)
                     VALUES (?, ?, ?, ?, ?, ?)''', (message_id, conversation_id, role, content, agent_used, confidence_score))
        
        # Update conversation last_updated
        c.execute('''UPDATE conversations SET last_updated = CURRENT_TIMESTAMP 
                     WHERE conversation_id = ?''', (conversation_id,))
    
    return message_id

def get_chat_history(conversation_id, limit=50):
    """Get chat history for a conversation."""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute('''SELECT message_id, role, content, agent_used, confidence_score, created_at 
                     FROM chat_messages WHERE conversation_id=? 
                     ORDER BY created_at ASC LIMIT ?''', (conversation_id, limit))
        return c.fetchall()

def get_conversation_context(conversation_id):
    """Get full context for a conversation including original analysis."""
    with get_conn() as conn:
        c = conn.cursor()
        
        # Get conversation and session info
        c.execute('''SELECT c.session_id, c.title, a.original, a.analysis 
                     FROM conversations c 
                     JOIN analysis a ON c.session_id = a.session_id 
                     WHERE c.conversation_id = ?''', (conversation_id,))
        conv_data = c.fetchone()
        
        if not conv_data:
            return None
        
        session_id, title, original, analysis = conv_data
        
        # Get recent chat history
        chat_history = get_chat_history(conversation_id, limit=20)
        
        return {
            'session_id': session_id,
            'title': title,
            'original': original,
            'analysis': analysis,
            'chat_history': chat_history
        }

def save_context_embedding(session_id, content_type, content_hash, embedding_data):
    """Save context embedding for vector search."""
    embedding_id = f"emb_{session_id}_{content_type}_{content_hash}"
    
    with get_conn() as conn:
        c = conn.cursor()
        c.execute('''REPLACE INTO context_embeddings (embedding_id, session_id, content_type, content_hash, embedding_data)
                     VALUES (?, ?, ?, ?, ?)''', (embedding_id, session_id, content_type, content_hash, json.dumps(embedding_data)))

def get_context_embeddings(session_id):
    """Get all context embeddings for a session."""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute('''SELECT content_type, content_hash, embedding_data 
                     FROM context_embeddings WHERE session_id=?''', (session_id,))
        return c.fetchall() 