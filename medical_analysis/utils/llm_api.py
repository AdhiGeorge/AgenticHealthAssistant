import os
import requests
from medical_analysis.utils.config import get_config
from medical_analysis.utils.logger import get_logger
from medical_analysis.utils.prompt_utils import load_prompts

config = get_config()
logger = get_logger(__name__)
prompts = load_prompts()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or config.get('api_keys', {}).get('gemini', '')
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or config.get('api_keys', {}).get('groq', '')
GEMINI_MODEL = config['models']['gemini_model']
GROQ_MODEL = config['models']['groq_model']

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

GROQ_AVAILABLE = bool(GROQ_API_KEY)

def summarize_chunk(chunk: str) -> str:
    """Summarize or analyze a chunk of text using Gemini or Groq, with fallback."""
    # Try Gemini
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            summarize_prompt_template = prompts.get('summarize_chunk', '')
            if summarize_prompt_template:
                prompt = summarize_prompt_template.format(chunk=chunk)
            else:
                prompt = f"Analyze: {chunk}"
            response = model.generate_content(prompt)
            logger.info("Chunk summarized using Gemini.")
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini summarization failed: {e}")
    # Try Groq
    if GROQ_AVAILABLE and GROQ_API_KEY:
        try:
            url = "https://api.groq.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            summarize_prompt_template = prompts.get('summarize_chunk', '')
            if summarize_prompt_template:
                prompt = summarize_prompt_template.format(chunk=chunk)
            else:
                prompt = chunk
            system_prompt = prompts.get('system_medical_expert', 'You are a medical expert.')
            data = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.3
            }
            resp = requests.post(url, headers=headers, json=data, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                logger.info("Chunk summarized using Groq.")
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"Groq API error: {resp.status_code} {resp.text}")
                return f"[GROQ ERROR] {resp.status_code}: {resp.text}"
        except Exception as e:
            logger.error(f"Groq summarization failed: {e}")
            return f"[GROQ FALLBACK ERROR] {str(e)}"
    # Fallback
    logger.warning("Both Gemini and Groq summarization failed. Returning fallback summary.")
    return f"[FALLBACK SUMMARY]\n{chunk[:500]}..."

def process_agent_prompt(prompt: str, content: str) -> str:
    """Process an agent prompt to get detailed, professional medical analysis."""
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            comprehensive_prompt_template = prompts.get('comprehensive_analysis_prompt', '')
            if comprehensive_prompt_template:
                enhanced_prompt = comprehensive_prompt_template.format(content=content, prompt=prompt)
            else:
                enhanced_prompt = f"Analyze: {content}\nTask: {prompt}"
            response = model.generate_content(enhanced_prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}")
    if GROQ_AVAILABLE and GROQ_API_KEY:
        try:
            url = "https://api.groq.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            detailed_prompt_template = prompts.get('detailed_analysis', '')
            if detailed_prompt_template:
                enhanced_prompt = detailed_prompt_template.format(content=content, prompt=prompt)
            else:
                enhanced_prompt = f"Analyze: {content}\nTask: {prompt}"
            system_prompt = prompts.get('system_senior_specialist', 'Medical specialist.')
            data = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": enhanced_prompt}
                ],
                "max_tokens": 2048,
                "temperature": 0.3
            }
            resp = requests.post(url, headers=headers, json=data, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.warning(f"Groq API error: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}")
    logger.info("Using fallback analysis due to API issues")
    return None  # Fallback will be handled by orchestrator 