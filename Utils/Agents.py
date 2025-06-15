import os
import yaml
import time
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv
from Utils.logger import setup_logger, log_execution_time, log_model_usage

# Load environment variables
load_dotenv()

# Configure API keys
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Setup logger
logger = setup_logger(__name__)

# Model configurations
GROQ_MODEL = "llama-3.3-70b-versatile"  # Production model with large context window
GEMINI_MODELS = {
    'primary': "gemini-1.5-pro",        # Best for complex medical reasoning
    'fallback': "gemini-1.5-flash",     # Faster alternative if primary fails
    'emergency': "gemini-1.5-flash-8b"  # Last resort for high volume tasks
}
MAX_RETRIES = 3
INITIAL_WAIT = 30  # Reduced initial wait time

class Agent:
    def __init__(self, role):
        """Initialize the agent with a specific role."""
        logger.info(f"Initializing agent with role: {role}")
        self.role = role
        self.primary_model = None
        self.fallback_model = None
        self.emergency_model = None
        self.prompts = self._load_prompts()
        self.initialize_models()
        
    def _load_prompts(self):
        """Load prompts from prompts.yaml file."""
        try:
            with open('prompts.yaml', 'r', encoding='utf-8') as file:
                prompts = yaml.safe_load(file)
                if self.role not in prompts:
                    raise ValueError(f"No prompts found for role: {self.role}")
                logger.info(f"Successfully loaded prompts for role: {self.role}")
                return prompts[self.role]
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise

    @log_execution_time
    def initialize_models(self):
        """Initialize all available models."""
        try:
            # Initialize Gemini models
            self.primary_model = genai.GenerativeModel(GEMINI_MODELS['primary'])
            self.fallback_model = genai.GenerativeModel(GEMINI_MODELS['fallback'])
            self.emergency_model = genai.GenerativeModel(GEMINI_MODELS['emergency'])
            logger.info("Successfully initialized Gemini models")
            
            # Initialize Groq model
            self.groq_model = groq_client.chat.completions
            logger.info("Successfully initialized Groq model")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    @log_execution_time
    def set_primary_model(self, model_name):
        """Set which model to use as primary."""
        if model_name.lower() == 'gemini':
            self.primary_model = genai.GenerativeModel(GEMINI_MODELS['primary'])
            self.fallback_model = self.groq_model
            logger.info("Set Gemini as primary model with Groq as fallback")
        elif model_name.lower() == 'groq':
            self.primary_model = self.groq_model
            self.fallback_model = genai.GenerativeModel(GEMINI_MODELS['primary'])
            logger.info("Set Groq as primary model with Gemini as fallback")
        else:
            raise ValueError("Model must be either 'gemini' or 'groq'")

    @log_execution_time
    def create_prompt_template(self, data):
        """Create a prompt template based on the agent's role and loaded prompts."""
        try:
            # Get the role description and steps from prompts
            role_desc = self.prompts['role']
            steps = self.prompts['steps']
            format_template = self.prompts['format']
            
            # Create the prompt
            prompt = f"{role_desc}\n\n"
            prompt += "Follow these steps:\n"
            for step in steps:
                prompt += f"- {step}\n"
            
            prompt += "\nUse this format for your response:\n"
            prompt += format_template
            
            prompt += "\n\nPatient Data:\n"
            prompt += f"{data}"
            
            logger.info("Successfully created prompt template")
            return prompt
            
        except Exception as e:
            logger.error(f"Error creating prompt template: {str(e)}")
            raise

    def _handle_gemini_rate_limit(self, model, prompt, max_retries=MAX_RETRIES):
        """Handle Gemini rate limits with optimized backoff."""
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = INITIAL_WAIT * (2 ** attempt)  # Exponential backoff starting at 30s
                    logger.warning(f"Gemini rate limit hit, waiting {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise

    def _call_groq_model(self, prompt):
        """Call Groq model with optimized parameters."""
        try:
            response = self.groq_model.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=32768,  # Maximum completion tokens for llama-3.3-70b-versatile
                top_p=0.95,
                stream=False,
                presence_penalty=0.1,  # Slight penalty for repetition
                frequency_penalty=0.1   # Slight penalty for frequent tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq model error: {str(e)}")
            raise

    @log_execution_time
    @log_model_usage
    def run(self, data):
        """Run the agent with the given data."""
        try:
            prompt = self.create_prompt_template(data)
            logger.info("Generated prompt for model")
            
            # Try primary model first
            try:
                if isinstance(self.primary_model, genai.GenerativeModel):
                    result = self._handle_gemini_rate_limit(self.primary_model, prompt)
                    logger.info("Successfully generated response using primary model (Gemini Pro)")
                else:
                    result = self._call_groq_model(prompt)
                    logger.info("Successfully generated response using primary model (Groq)")
                    
            except Exception as primary_error:
                logger.warning(f"Primary model failed: {str(primary_error)}")
                logger.info("Attempting fallback model...")
                
                # Try fallback model
                try:
                    if isinstance(self.fallback_model, genai.GenerativeModel):
                        # Try Gemini Flash first
                        try:
                            result = self._handle_gemini_rate_limit(self.fallback_model, prompt)
                            logger.info("Successfully generated response using fallback model (Gemini Flash)")
                        except Exception as flash_error:
                            logger.warning(f"Gemini Flash failed: {str(flash_error)}")
                            # Try emergency model as last resort
                            result = self._handle_gemini_rate_limit(self.emergency_model, prompt)
                            logger.info("Successfully generated response using emergency model (Gemini Flash-8B)")
                    else:
                        result = self._call_groq_model(prompt)
                        logger.info("Successfully generated response using fallback model (Groq)")
                        
                except Exception as fallback_error:
                    logger.error(f"All models failed: {str(fallback_error)}")
                    raise Exception("All available models failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in run method: {str(e)}")
            raise

class Cardiologist(Agent):
    def __init__(self, data):
        super().__init__('cardiology')
        self.data = data

class Psychologist(Agent):
    def __init__(self, data):
        super().__init__('psychology')
        self.data = data

class Pulmonologist(Agent):
    def __init__(self, data):
        super().__init__('pulmonology')
        self.data = data

class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        """Initialize the multidisciplinary team with reports from all specialists."""
        # Combine all reports into a single data structure
        data = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__('multidisciplinary')
        self.data = data

    def create_prompt_template(self, data):
        """Override create_prompt_template to format multiple reports."""
        try:
            # Get the role description and steps from prompts
            role_desc = self.prompts['role']
            steps = self.prompts['steps']
            format_template = self.prompts['format']
            
            # Create the prompt
            prompt = f"{role_desc}\n\n"
            prompt += "Follow these steps:\n"
            for step in steps:
                prompt += f"- {step}\n"
            
            prompt += "\nUse this format for your response:\n"
            prompt += format_template
            
            prompt += "\n\nSpecialist Reports:\n"
            prompt += f"Cardiology Report:\n{data['cardiologist_report']}\n\n"
            prompt += f"Psychology Report:\n{data['psychologist_report']}\n\n"
            prompt += f"Pulmonology Report:\n{data['pulmonologist_report']}\n"
            
            logger.info("Successfully created multidisciplinary prompt template")
            return prompt
            
        except Exception as e:
            logger.error(f"Error creating prompt template: {str(e)}")
            raise
