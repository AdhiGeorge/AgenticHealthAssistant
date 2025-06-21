# from langgraph.graph import StateGraph
from .cardiology import CardiologyAgent
from .neurology import NeurologyAgent
from .pulmonology import PulmonologyAgent
from .endocrinology import EndocrinologyAgent
from .gastroenterology import GastroenterologyAgent
from .hematology import HematologyAgent
from .nephrology import NephrologyAgent
from .rheumatology import RheumatologyAgent
from .infectious_disease import InfectiousDiseaseAgent
from .oncology import OncologyAgent
from .dermatology import DermatologyAgent
from .ophthalmology import OphthalmologyAgent
from .orthopedics import OrthopedicsAgent
from .psychiatry import PsychiatryAgent
from .pediatrics import PediatricsAgent
from .geriatrics import GeriatricsAgent
from medical_analysis.utils.text_utils import chunk_text, get_default_tokenizer
from medical_analysis.utils.config import get_config
from medical_analysis.utils.logger import get_logger
from medical_analysis.utils.prompt_utils import load_prompts
from medical_analysis.utils.review_agent import ReviewAgent
import os
import yaml
from pathlib import Path
from medical_analysis.utils.document_extractor import extract_text_from_document
from medical_analysis.extractors.pdf_extractor import extract_text_from_pdf
from medical_analysis.extractors.image_extractor import extract_text_from_image
from medical_analysis.extractors.docx_extractor import extract_text_from_docx
from medical_analysis.extractors.txt_extractor import extract_text_from_txt
from medical_analysis.extractors.csv_extractor import extract_text_from_csv
from medical_analysis.extractors.excel_extractor import extract_text_from_excel
from dotenv import load_dotenv
import requests
from medical_analysis.utils.db import save_analysis, get_analysis, save_agent_state, get_agent_state
import uuid
from typing import TypedDict
from datetime import datetime
from medical_analysis.utils.llm_api import summarize_chunk, process_agent_prompt
from medical_analysis.utils.fallback_analysis import generate_fallback_analysis, generate_fallback_comprehensive_summary
from medical_analysis.utils.report_formatter import format_final_report, format_fallback_report

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = None

# Load environment variables first
load_dotenv()

config = get_config()
logger = get_logger(__name__)

# Load prompts
prompts = load_prompts()

# API keys - prioritize environment variables over config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = config.get('api_keys', {}).get('gemini', '')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = config.get('api_keys', {}).get('groq', '')

# Debug logging for API keys
logger.info(f"GEMINI_API_KEY loaded: {'Yes' if GEMINI_API_KEY else 'No'} (length: {len(GEMINI_API_KEY) if GEMINI_API_KEY else 0})")
logger.info(f"GROQ_API_KEY loaded: {'Yes' if GROQ_API_KEY else 'No'} (length: {len(GROQ_API_KEY) if GROQ_API_KEY else 0})")

# Test API key validity
def test_gemini_api_key():
    """Test if the Gemini API key is valid."""
    if not GEMINI_API_KEY:
        logger.warning("No Gemini API key found")
        return False
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello")
        logger.info("Gemini API key test successful")
        return True
    except Exception as e:
        logger.error(f"Gemini API key test failed: {e}")
        return False

# Test the API key on startup
GEMINI_API_VALID = test_gemini_api_key()
if not GEMINI_API_VALID:
    logger.warning("Gemini API key is invalid or not working. Will use fallback analysis.")

PRIMARY_MODEL = config['models']['primary']
FALLBACK_MODEL = config['models']['fallback']
GEMINI_MODEL = config['models']['gemini_model']
GROQ_MODEL = config['models']['groq_model']
MAX_TOKENS = config['models']['max_tokens']
CHUNK_OVERLAP = config['models']['chunk_overlap']

# Gemini setup
try:
    import google.generativeai as genai
    if GEMINI_API_KEY and GEMINI_API_VALID:
        genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = GEMINI_API_VALID
except ImportError:
    GEMINI_AVAILABLE = False

GROQ_AVAILABLE = bool(GROQ_API_KEY)

class OrchestratorState(TypedDict):
    text: str
    attempt: int
    agent_results: dict
    report: str
    status: str
    score: float

class OrchestratorAgent:
    """
    Orchestrates multi-agent analysis using LangGraph's StateGraph.
    The workflow:
    1. Analyze with all relevant agents.
    2. Aggregate results and format the report.
    3. Review the report with the ReviewAgent.
    4. If review fails, repeat analysis; if approved, finalize.
    """
    def __init__(self, model: str = GEMINI_MODEL, use_gpu: bool = True):
        self.agents = {
            'cardiology': CardiologyAgent(model, use_gpu),
            'neurology': NeurologyAgent(model, use_gpu),
            'pulmonology': PulmonologyAgent(model, use_gpu),
            'endocrinology': EndocrinologyAgent(model, use_gpu),
            'gastroenterology': GastroenterologyAgent(model, use_gpu),
            'hematology': HematologyAgent(model, use_gpu),
            'nephrology': NephrologyAgent(model, use_gpu),
            'rheumatology': RheumatologyAgent(model, use_gpu),
            'infectious_disease': InfectiousDiseaseAgent(model, use_gpu),
            'oncology': OncologyAgent(model, use_gpu),
            'dermatology': DermatologyAgent(model, use_gpu),
            'ophthalmology': OphthalmologyAgent(model, use_gpu),
            'orthopedics': OrthopedicsAgent(model, use_gpu),
            'psychiatry': PsychiatryAgent(model, use_gpu),
            'pediatrics': PediatricsAgent(model, use_gpu),
            'geriatrics': GeriatricsAgent(model, use_gpu)
        }
        self.specialty_keywords = self._build_specialty_keywords()
        self.tokenizer = get_default_tokenizer()
        self.review_agent = ReviewAgent(threshold=config.get('review_threshold', 0.5))
        logger.info("OrchestratorAgent (LangGraph) initialized.")

    def _build_specialty_keywords(self):
        return {
            'cardiology': ['heart', 'cardiac', 'cardiovascular', 'chest pain', 'ecg', 'echo'],
            'neurology': ['brain', 'nerve', 'neurological', 'headache', 'seizure', 'stroke'],
            'pulmonology': ['lung', 'respiratory', 'breathing', 'asthma', 'copd', 'pneumonia'],
            'endocrinology': ['hormone', 'diabetes', 'thyroid', 'endocrine', 'insulin'],
            'gastroenterology': ['stomach', 'intestine', 'liver', 'digestive', 'gi'],
            'hematology': ['blood', 'anemia', 'leukemia', 'platelet', 'coagulation'],
            'nephrology': ['kidney', 'renal', 'dialysis', 'creatinine', 'glomerular'],
            'rheumatology': ['joint', 'arthritis', 'autoimmune', 'lupus', 'rheumatoid'],
            'infectious_disease': ['infection', 'bacterial', 'viral', 'fever', 'antibiotic'],
            'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'radiation'],
            'dermatology': ['skin', 'rash', 'dermatitis', 'psoriasis', 'eczema'],
            'ophthalmology': ['eye', 'vision', 'retina', 'glaucoma', 'cataract'],
            'orthopedics': ['bone', 'joint', 'fracture', 'spine', 'musculoskeletal'],
            'psychiatry': ['mental', 'depression', 'anxiety', 'psychiatric', 'behavior'],
            'pediatrics': ['child', 'pediatric', 'growth', 'development', 'vaccination'],
            'geriatrics': ['elderly', 'aging', 'geriatric', 'senior', 'dementia']
        }

    def _identify_relevant_agents(self, text: str):
        text_lower = text.lower()
        relevant = []
        for specialty, keywords in self.specialty_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                relevant.append(specialty)
        logger.info(f"Identified relevant agents: {relevant}")
        return relevant if relevant else list(self.agents.keys())

    def summarize_chunk(self, chunk: str) -> str:
        return summarize_chunk(chunk)

    def analyze_with_agents(self, content: str):
        relevant_agents = self._identify_relevant_agents(content)
        results = {}
        for specialty in relevant_agents:
            agent = self.agents[specialty]
            agent_result = agent.analyze(content)
            if isinstance(agent_result, dict) and 'prompt' in agent_result:
                analysis_text = process_agent_prompt(agent_result['prompt'], content)
                if not analysis_text:
                    analysis_text = generate_fallback_analysis(agent_result['prompt'], content)
                results[specialty] = {
                    'findings': analysis_text,
                    'recommendations': f"Consult {specialty.title()} specialist for detailed evaluation.",
                    'diagnosis': f"Requires {specialty.title()} assessment.",
                    'summary': analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text
                }
            else:
                results[specialty] = {
                    'findings': str(agent_result),
                    'recommendations': f"Consult {specialty.title()} specialist.",
                    'diagnosis': f"Requires {specialty.title()} assessment.",
                    'summary': str(agent_result)[:200] + "..." if len(str(agent_result)) > 200 else str(agent_result)
                }
        logger.info(f"Analysis completed by agents: {list(results.keys())}")
        return results

    def _process_agent_prompt(self, prompt: str, content: str) -> str:
        # Deprecated: use analyze_with_agents instead
        pass

    def _generate_fallback_analysis(self, prompt: str, content: str) -> str:
        return generate_fallback_analysis(prompt, content)

    def format_final_report(self, patient_info, summary, agent_results):
        return format_final_report(patient_info, summary, agent_results)

    def _generate_comprehensive_summary(self, text: str) -> dict:
        # Only fallback logic here
        return generate_fallback_comprehensive_summary(text)

    def orchestrate(self, text: str, max_attempts: int = 3) -> str:
        session_id = str(uuid.uuid4())
        attempt = 0
        patient_info = text[:300]
        summary = {'presenting_complaint': text, 'summary': text}
        last_report = None
        last_status = None
        last_score = None
        if StateGraph:
            # Build the workflow graph using the builder pattern
            builder = StateGraph(OrchestratorState)
            
            def agent_step(state):
                logger.info(f"[LangGraph] Agent analysis step (attempt {state['attempt']})")
                agent_results = self.analyze_with_agents(state['text'])
                return {**state, 'agent_results': agent_results}
            
            def review_step(state):
                logger.info(f"[LangGraph] Review step (attempt {state['attempt']})")
                report = self.format_final_report(patient_info, summary, state['agent_results'])
                status, score = self.review_agent.review(state['text'], report)
                save_analysis(session_id, state['text'], report, status, score)
                return {**state, 'report': report, 'status': status, 'score': score}
            
            builder.add_node('analyze', agent_step)
            builder.add_node('review', review_step)
            builder.add_edge('analyze', 'review')
            
            # Fix the conditional edge logic
            def should_continue(state):
                if state['status'] == 'approved':
                    return END
                if state['attempt'] >= max_attempts:
                    return END
                return 'analyze'
            
            builder.add_conditional_edges('review', should_continue)
            builder.set_entry_point('analyze')
            
            # Compile with increased recursion limit
            graph = builder.compile(checkpointer=None)
            
            # Run the graph with proper state management
            state = {'text': text, 'attempt': 1, 'agent_results': {}, 'report': '', 'status': '', 'score': 0.0}
            
            try:
                # Use invoke with proper error handling
                final_state = graph.invoke(state)
                logger.info("Final analysis report generated and approved via LangGraph.")
                return final_state['report']
            except Exception as e:
                logger.error(f"LangGraph execution failed: {e}")
                # Fallback to simple loop
                logger.info("Falling back to simple loop due to LangGraph error")
                return self._simple_orchestrate(text, max_attempts, session_id, patient_info, summary)
        else:
            # Fallback: simple loop
            attempt = 0
            last_report = None
            last_status = None
            last_score = None
            while attempt < max_attempts:
                attempt += 1
                logger.info(f"Analysis attempt {attempt}")
                agent_results = self.analyze_with_agents(text)
                report = self.format_final_report(patient_info, summary, agent_results)
                status, score = self.review_agent.review(text, report)
                save_analysis(session_id, text, report, status, score)
                last_report = report
                last_status = status
                last_score = score
                logger.info(f"Review status: {status}, score: {score:.2f}")
                if status == "approved":
                    logger.info("Final analysis report generated and approved.")
                    return report
                if attempt >= max_attempts:
                    logger.warning(f"Max attempts ({max_attempts}) reached. Returning best available report.")
                    return f"[WARNING] Max attempts reached. The report may be incomplete or less relevant.\n\n{last_report}"
                logger.warning("Analysis did not meet relevancy threshold. Retrying...")
            return last_report or "Failed to generate report"

    def _simple_orchestrate(self, text: str, max_attempts: int, session_id: str, patient_info: str, summary: dict) -> str:
        """Simple orchestration loop as fallback."""
        attempt = 0
        last_report = None
        last_status = None
        last_score = None
        
        while attempt < max_attempts:
            attempt += 1
            logger.info(f"Analysis attempt {attempt}")
            agent_results = self.analyze_with_agents(text)
            report = self.format_final_report(patient_info, summary, agent_results)
            status, score = self.review_agent.review(text, report)
            save_analysis(session_id, text, report, status, score)
            last_report = report
            last_status = status
            last_score = score
            logger.info(f"Review status: {status}, score: {score:.2f}")
            
            if status == "approved":
                logger.info("Final analysis report generated and approved.")
                return report
            
            if attempt >= max_attempts:
                logger.warning(f"Max attempts ({max_attempts}) reached. Returning best available report.")
                return f"[WARNING] Max attempts reached. The report may be incomplete or less relevant.\n\n{last_report}"
            
            logger.warning("Analysis did not meet relevancy threshold. Retrying...")
        
        return last_report or "Failed to generate report"

    def answer_query(self, query: str, context: dict):
        relevant_agents = self._identify_relevant_agents(query)
        answers = {}
        for specialty in relevant_agents:
            agent = self.agents[specialty]
            answers[specialty] = agent.answer_query(query, context)
        logger.info(f"Query answered by agents: {list(answers.keys())}")
        return answers 