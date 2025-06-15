# AI Agents for Medical Diagnostics

An advanced medical diagnostic system that uses multiple AI models to analyze medical reports and provide comprehensive specialist insights.

## Features

- Multi-model AI analysis using Gemini and Groq
- Specialized medical report analysis
- Multi-disciplinary team approach
- Comprehensive logging and error handling
- Configurable model selection and fallback strategies

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- API keys for:
  - Google Gemini
  - Groq

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Agents-for-Medical-Diagnostics.git
cd AI-Agents-for-Medical-Diagnostics
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage

1. Place your medical report in the appropriate directory
2. Run the main script:
```bash
python main.py
```

3. The system will:
   - Analyze the medical report
   - Generate specialist insights
   - Provide a comprehensive diagnosis
   - Save the results to the output directory

## Project Structure

```
AI-Agents-for-Medical-Diagnostics/
├── Utils/
│   ├── Agents.py      # AI agent implementations
│   └── logger.py      # Logging utilities
├── main.py            # Main application entry
├── prompts.yaml       # AI prompt templates
├── requirements.txt   # Project dependencies
├── .env              # Environment variables
└── README.md         # This file
```

## Configuration

The system can be configured through:
- `prompts.yaml`: Customize AI prompts
- Environment variables: API keys and model settings
- Logging configuration in `Utils/logger.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API
- Groq API
- Medical diagnostic community
