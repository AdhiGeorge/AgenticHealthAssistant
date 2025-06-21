"""SQLite-based memory and agent state management utility."""
import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'results', 'analysis_memory.db')

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
        c.execute('''CREATE TABLE IF NOT EXISTS analysis (
            session_id TEXT PRIMARY KEY,
            original TEXT,
            analysis TEXT,
            status TEXT,
            score REAL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS agent_state (
            session_id TEXT,
            agent TEXT,
            state TEXT,
            PRIMARY KEY (session_id, agent)
        )''')


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