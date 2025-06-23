"""Chat manager for handling medical conversations with context and agent routing."""

import uuid
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from medical_analysis.agents.orchestrator import OrchestratorAgent
from medical_analysis.utils.db import (
    create_conversation, get_conversation_context, save_chat_message,
    get_chat_history, get_analysis
)
from medical_analysis.utils.vector_store import MedicalVectorStore
from medical_analysis.utils.logger import get_logger
from medical_analysis.utils.config import get_config
from medical_analysis.utils.text_utils import get_default_tokenizer

class ChatManager:
    """Manages medical conversations with context preservation and agent routing."""
    
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.tokenizer = get_default_tokenizer()
        
        # Initialize vector store
        self.vector_store = MedicalVectorStore()
        
        # Load prompts
        try:
            import yaml
            import os
            prompts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'prompts.yaml')
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.prompts = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load prompts: {e}")
            self.prompts = {}
    
    def start_conversation(self, session_id: str, title: Optional[str] = None) -> str:
        """Start a new conversation for a session."""
        try:
            conversation_id = create_conversation(session_id, title)
            self.logger.info(f"Started conversation {conversation_id} for session {session_id}")
            return conversation_id
        except Exception as e:
            self.logger.error(f"Failed to start conversation: {e}")
            raise
    
    def store_session_context(self, session_id: str, original_data: str, analysis_report: str):
        """Store session context in vector database for enhanced search."""
        try:
            # Store original data
            if original_data:
                self.vector_store.store_medical_context(
                    session_id=session_id,
                    content=original_data,
                    content_type="original_data",
                    metadata={"source": "user_input"}
                )
            
            # Store analysis report
            if analysis_report:
                self.vector_store.store_medical_context(
                    session_id=session_id,
                    content=analysis_report,
                    content_type="analysis_report",
                    metadata={"source": "ai_analysis"}
                )
                
            self.logger.info(f"Stored session context in vector database for {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store session context: {e}")
    
    def get_conversation_info(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation information and context."""
        try:
            context = get_conversation_context(conversation_id)
            if not context:
                self.logger.warning(f"Conversation {conversation_id} not found")
                return None
            
            # Add vector search results if available
            if self.vector_store.client:
                session_id = context.get('session_id')
                if session_id:
                    vector_context = self.vector_store.get_session_context(session_id)
                    context['vector_context'] = vector_context
            
            return context
        except Exception as e:
            self.logger.error(f"Failed to get conversation info: {e}")
            return None
    
    def _search_relevant_context(self, question: str, session_id: str) -> List[Dict]:
        """Search for relevant context using vector similarity."""
        try:
            if not self.vector_store.client:
                return []
            
            # Search for similar context
            similar_context = self.vector_store.search_similar_context(
                query=question,
                session_id=session_id,
                limit=3,
                threshold=0.6
            )
            
            return similar_context
            
        except Exception as e:
            self.logger.error(f"Failed to search relevant context: {e}")
            return []
    
    def _determine_agent_for_question(self, question: str) -> str:
        """Determine which agent should handle the question."""
        try:
            # Use agent routing prompt from prompts.yaml
            routing_prompt = self.prompts.get('chat_agent_routing_prompt', '')
            if routing_prompt:
                prompt = routing_prompt.format(user_question=question)
            else:
                # Fallback routing logic
                question_lower = question.lower()
                if any(term in question_lower for term in ['heart', 'cardiac', 'chest pain', 'ecg']):
                    return 'cardiology'
                elif any(term in question_lower for term in ['brain', 'nerve', 'headache', 'seizure']):
                    return 'neurology'
                elif any(term in question_lower for term in ['lung', 'breathing', 'asthma']):
                    return 'pulmonology'
                elif any(term in question_lower for term in ['hormone', 'diabetes', 'thyroid']):
                    return 'endocrinology'
                elif any(term in question_lower for term in ['stomach', 'intestine', 'liver']):
                    return 'gastroenterology'
                elif any(term in question_lower for term in ['blood', 'anemia', 'leukemia']):
                    return 'hematology'
                elif any(term in question_lower for term in ['kidney', 'renal', 'dialysis']):
                    return 'nephrology'
                elif any(term in question_lower for term in ['joint', 'arthritis', 'lupus']):
                    return 'rheumatology'
                elif any(term in question_lower for term in ['infection', 'bacterial', 'viral']):
                    return 'infectious_disease'
                elif any(term in question_lower for term in ['cancer', 'tumor', 'malignant']):
                    return 'oncology'
                elif any(term in question_lower for term in ['skin', 'rash', 'dermatitis']):
                    return 'dermatology'
                elif any(term in question_lower for term in ['eye', 'vision', 'retina']):
                    return 'ophthalmology'
                elif any(term in question_lower for term in ['bone', 'joint', 'fracture']):
                    return 'orthopedics'
                elif any(term in question_lower for term in ['mental', 'depression', 'anxiety']):
                    return 'psychiatry'
                elif any(term in question_lower for term in ['child', 'pediatric', 'growth']):
                    return 'pediatrics'
                elif any(term in question_lower for term in ['elderly', 'aging', 'dementia']):
                    return 'geriatrics'
                else:
                    return 'general'
            
            # Try to use LLM for routing if available
            try:
                from medical_analysis.agents.orchestrator import GEMINI_API_VALID, GEMINI_AVAILABLE, GEMINI_API_KEY
                if GEMINI_API_VALID and GEMINI_AVAILABLE and GEMINI_API_KEY:
                    import google.generativeai as genai
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(prompt)
                    specialty = response.text.strip().lower()
                    
                    # Validate the response
                    valid_specialties = [
                        'cardiology', 'neurology', 'pulmonology', 'endocrinology', 
                        'gastroenterology', 'hematology', 'nephrology', 'rheumatology',
                        'infectious_disease', 'oncology', 'dermatology', 'ophthalmology',
                        'orthopedics', 'psychiatry', 'pediatrics', 'geriatrics', 'general'
                    ]
                    
                    if specialty in valid_specialties:
                        return specialty
            except Exception as e:
                self.logger.warning(f"LLM routing failed, using fallback: {e}")
            
            return 'general'
            
        except Exception as e:
            self.logger.error(f"Failed to determine agent for question: {e}")
            return 'general'
    
    def _assess_confidence(self, question: str, context: Dict) -> float:
        """Assess confidence in answering the question."""
        try:
            confidence_prompt = self.prompts.get('chat_confidence_assessment_prompt', '')
            if confidence_prompt:
                prompt = confidence_prompt.format(
                    user_question=question,
                    has_original_data=bool(context.get('original')),
                    has_analysis_report=bool(context.get('analysis')),
                    has_chat_history=bool(context.get('chat_history'))
                )
                
                # Try to use LLM for confidence assessment
                try:
                    from medical_analysis.agents.orchestrator import GEMINI_API_VALID, GEMINI_AVAILABLE, GEMINI_API_KEY
                    if GEMINI_API_VALID and GEMINI_AVAILABLE and GEMINI_API_KEY:
                        import google.generativeai as genai
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(prompt)
                        try:
                            confidence = float(response.text.strip())
                            return max(0.0, min(1.0, confidence))
                        except ValueError:
                            pass
                except Exception as e:
                    self.logger.warning(f"LLM confidence assessment failed: {e}")
            
            # Fallback confidence assessment
            has_original = bool(context.get('original'))
            has_analysis = bool(context.get('analysis'))
            has_history = bool(context.get('chat_history'))
            has_vector_context = bool(context.get('vector_context'))
            
            if has_original and has_analysis and has_vector_context:
                return 0.9
            elif has_original and has_analysis:
                return 0.8
            elif has_analysis:
                return 0.6
            elif has_original:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            self.logger.error(f"Failed to assess confidence: {e}")
            return 0.5
    
    def _format_chat_history(self, chat_history: List) -> str:
        """Format chat history for context."""
        if not chat_history:
            return "No previous conversation."
        
        formatted = []
        for msg in chat_history[-10:]:  # Last 10 messages
            role, content = msg[1], msg[2]
            formatted.append(f"{role.title()}: {content}")
        
        return "\n".join(formatted)
    
    def _format_vector_context(self, vector_context: List[Dict]) -> str:
        """Format vector search results for context."""
        if not vector_context:
            return ""
        
        formatted = []
        for item in vector_context[:3]:  # Top 3 results
            content_type = item.get('content_type', 'unknown')
            preview = item.get('content_preview', '')
            score = item.get('score', 0)
            formatted.append(f"[{content_type.upper()}] (Score: {score:.2f}): {preview}")
        
        return "\n".join(formatted)
    
    def _generate_response_with_context(self, question: str, context: Dict, agent_used: str) -> str:
        """Generate response using context and appropriate agent."""
        try:
            # Format context
            original_data = context.get('original', 'Not available')
            analysis_report = context.get('analysis', 'Not available')
            chat_history = self._format_chat_history(context.get('chat_history', []))
            vector_context = self._format_vector_context(context.get('vector_context', []))
            
            # Use context prompt from prompts.yaml
            context_prompt = self.prompts.get('chat_context_prompt', '')
            if context_prompt:
                prompt = context_prompt.format(
                    original_data=original_data[:2000],  # Limit length
                    analysis_report=analysis_report[:3000],  # Limit length
                    chat_history=chat_history,
                    user_question=question
                )
                
                # Add vector context if available
                if vector_context:
                    prompt += f"\n\nRELEVANT CONTEXT FROM VECTOR SEARCH:\n{vector_context}"
            else:
                # Fallback prompt
                prompt = f"""
                Context:
                Original Data: {original_data[:1000]}
                Analysis Report: {analysis_report[:2000]}
                Chat History: {chat_history}
                
                {f"Relevant Context: {vector_context}" if vector_context else ""}
                
                Question: {question}
                
                Please provide a helpful response based on the available information.
                """
            
            # Try to use LLM for response generation
            try:
                from medical_analysis.agents.orchestrator import GEMINI_API_VALID, GEMINI_AVAILABLE, GEMINI_API_KEY
                if GEMINI_API_VALID and GEMINI_AVAILABLE and GEMINI_API_KEY:
                    import google.generativeai as genai
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(prompt)
                    return response.text.strip()
            except Exception as e:
                self.logger.warning(f"LLM response generation failed: {e}")
            
            # Fallback response
            return self._generate_fallback_response(question, context, agent_used)
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            return "I apologize, but I'm unable to process your question at the moment. Please try again or consult with a healthcare professional for specific medical advice."
    
    def _generate_fallback_response(self, question: str, context: Dict, agent_used: str) -> str:
        """Generate fallback response when LLM is unavailable."""
        try:
            question_lower = question.lower()
            
            # Check if question is about the specific report
            if any(term in question_lower for term in ['report', 'analysis', 'findings', 'diagnosis']):
                if context.get('analysis'):
                    return f"Based on the medical analysis report, I can provide information about the findings. However, for specific medical advice, please consult with a healthcare professional. The report contains detailed clinical assessments and recommendations."
                else:
                    return "I don't have access to the specific medical analysis report you're referring to. Please ensure the report has been generated and try again."
            
            # Check if it's a general medical question
            elif any(term in question_lower for term in ['what is', 'define', 'explain', 'how does']):
                return f"This appears to be a general medical question about {agent_used}. While I can provide general information, for specific medical advice, please consult with a healthcare professional. I recommend speaking with a {agent_used} specialist for detailed evaluation."
            
            # Default response
            else:
                return "I understand your question, but I need more specific information to provide a helpful response. Could you please clarify your question or provide more context? For medical advice, please consult with a healthcare professional."
                
        except Exception as e:
            self.logger.error(f"Failed to generate fallback response: {e}")
            return "I apologize, but I'm unable to process your question at the moment. Please consult with a healthcare professional for medical advice."
    
    def _is_question_relevant(self, question: str, context: dict) -> bool:
        """Determine if the question is relevant to the medical report, analysis, or medical field."""
        question_lower = question.lower()
        # Keywords for medical relevance
        medical_keywords = [
            'report', 'analysis', 'finding', 'diagnosis', 'treatment', 'symptom', 'disease', 'condition',
            'doctor', 'specialist', 'recommendation', 'follow-up', 'prescription', 'test', 'scan', 'lab',
            'blood', 'heart', 'lung', 'brain', 'pain', 'anxiety', 'attack', 'disorder', 'medicine',
            'therapy', 'procedure', 'risk', 'prognosis', 'prognostic', 'clinical', 'consult', 'referral',
            'medical', 'health', 'hospital', 'clinic', 'care', 'cardiac', 'psychiatry', 'pulmonary',
            'endocrine', 'hematology', 'oncology', 'gastro', 'nephro', 'neuro', 'derm', 'ophthal', 'ortho',
            'pediatric', 'geriatrics', 'infectious', 'imaging', 'scan', 'x-ray', 'ct', 'mri', 'ultrasound',
            'prescribe', 'dose', 'side effect', 'contraindication', 'allergy', 'emergency', 'urgent',
        ]
        # Check if any medical keyword is present
        if any(kw in question_lower for kw in medical_keywords):
            return True
        # Check if question references the report or analysis
        if context.get('analysis') and any(kw in question_lower for kw in ['report', 'analysis', 'finding', 'diagnosis']):
            return True
        # NEW: Check if the question's main words/phrases are present in the report or analysis text
        report_text = (context.get('original', '') + ' ' + context.get('analysis', '')).lower()
        # Split question into words and check if any are present in the report/analysis
        question_words = [w for w in question_lower.split() if len(w) > 2]
        if any(word in report_text for word in question_words):
            return True
        return False

    def ask_question(self, conversation_id: str, question: str) -> Dict:
        """Ask a question in a conversation and get a response."""
        try:
            # Get conversation context
            context = self.get_conversation_info(conversation_id)
            if not context:
                return {
                    'success': False,
                    'error': 'Conversation not found',
                    'response': None
                }

            # Search for relevant context using vector similarity
            session_id = context.get('session_id')
            if session_id:
                relevant_context = self._search_relevant_context(question, session_id)
                if relevant_context:
                    context['vector_context'] = relevant_context
            
            # Determine appropriate agent
            agent_used = self._determine_agent_for_question(question)
            
            # Assess confidence
            confidence = self._assess_confidence(question, context)
            
            # Generate response
            response = self._generate_response_with_context(question, context, agent_used)
            
            # Store the question and response in vector database
            if session_id:
                self.vector_store.store_medical_context(
                    session_id=session_id,
                    content=question,
                    content_type="user_question",
                    metadata={"agent_used": agent_used, "confidence": confidence}
                )
                self.vector_store.store_medical_context(
                    session_id=session_id,
                    content=response,
                    content_type="ai_response",
                    metadata={"agent_used": agent_used, "confidence": confidence}
                )
            
            # Save the exchange
            save_chat_message(conversation_id, 'user', question)
            save_chat_message(conversation_id, 'assistant', response, agent_used, confidence)
            
            return {
                'success': True,
                'response': response,
                'agent_used': agent_used,
                'confidence': confidence,
                'conversation_id': conversation_id,
                'vector_context_used': bool(relevant_context) if 'relevant_context' in locals() else False
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process question: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "I apologize, but I encountered an error processing your question. Please try again."
            }
    
    def get_chat_history(self, conversation_id: str, limit: int = 50) -> List:
        """Get chat history for a conversation."""
        try:
            return get_chat_history(conversation_id, limit)
        except Exception as e:
            self.logger.error(f"Failed to get chat history: {e}")
            return []
    
    def list_conversations(self, session_id: str) -> List:
        """List all conversations for a session."""
        try:
            from medical_analysis.utils.db import get_conversations_for_session
            return get_conversations_for_session(session_id)
        except Exception as e:
            self.logger.error(f"Failed to list conversations: {e}")
            return []
    
    def get_vector_stats(self) -> Dict:
        """Get vector database statistics."""
        return self.vector_store.get_collection_stats()
