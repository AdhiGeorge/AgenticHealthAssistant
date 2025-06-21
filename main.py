"""Enhanced Medical Diagnostics CLI with Rich console and Qdrant integration."""

import os
import sys
from pathlib import Path
from typing import Optional
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich import box

from medical_analysis.agents.orchestrator import OrchestratorAgent
from medical_analysis.agents.chat_manager import ChatManager
from medical_analysis.utils.db import init_db, get_analysis, save_analysis
from medical_analysis.utils.logger import get_logger
from medical_analysis.utils.config import get_config
from medical_analysis.utils.text_utils import chunk_text, get_default_tokenizer
from medical_analysis.utils.document_extractor import extract_text_from_document

# Initialize Rich console
console = Console()

class ReviewAgent:
    """Reflection agent to compare analysis to original data and score relevancy."""
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.logger = get_logger("ReviewAgent")

    def score_relevancy(self, original, analysis):
        # Simple scoring: ratio of overlapping words (can be replaced with embedding similarity)
        original_words = set(original.lower().split())
        analysis_words = set(analysis.lower().split())
        overlap = len(original_words & analysis_words)
        score = overlap / max(len(original_words), 1)
        self.logger.info(f"Relevancy score: {score:.2f}")
        return score

    def review(self, original, analysis):
        score = self.score_relevancy(original, analysis)
        status = "approved" if score >= self.threshold else "retry"
        return status, score

def display_welcome():
    """Display welcome screen."""
    welcome_text = Text()
    welcome_text.append("🏥 ", style="bold blue")
    welcome_text.append("MEDICAL DIAGNOSTICS AI SYSTEM", style="bold white")
    welcome_text.append("\n\n", style="white")
    welcome_text.append("Advanced AI-powered medical analysis with multi-agent orchestration\n", style="cyan")
    welcome_text.append("Powered by Gemini, Groq, and Qdrant vector search", style="dim white")
    
    panel = Panel(
        Align.center(welcome_text),
        title="[bold green]Welcome[/bold green]",
        border_style="green",
        box=box.ROUNDED
    )
    console.print(panel)

def display_main_menu() -> str:
    """Display main menu and get user choice."""
    table = Table(title="[bold blue]Main Menu[/bold blue]", box=box.ROUNDED)
    table.add_column("Option", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    
    table.add_row("1", "📝 Text Input - Enter medical text directly")
    table.add_row("2", "📁 Document Upload - Process medical documents")
    table.add_row("3", "📊 System Status - Check Qdrant and database status")
    table.add_row("4", "❓ Help - Show usage instructions")
    table.add_row("5", "🚪 Exit - Close the application")
    
    console.print(table)
    
    while True:
        choice = Prompt.ask(
            "\n[bold green]Select an option[/bold green]",
            choices=["1", "2", "3", "4", "5"],
            default="1"
        )
        
        if choice == "5":
            if Confirm.ask("[bold red]Are you sure you want to exit?[/bold red]"):
                console.print("[bold green]Thank you for using Medical Diagnostics AI! 👋[/bold green]")
                sys.exit(0)
        else:
            return choice

def get_text_input() -> str:
    """Get text input from user."""
    console.print("\n[bold cyan]📝 TEXT INPUT MODE[/bold cyan]")
    console.print("Enter your medical text below. Press [bold red]Enter[/bold red] twice to finish.\n")
    
    lines = []
    line_count = 0
    
    while True:
        line = console.input(f"[dim]Line {line_count + 1}:[/dim] ")
        if line == '':
            if lines:  # If we have content and user pressed Enter twice
                break
            else:
                console.print("[yellow]Please enter some text first.[/yellow]")
                continue
        lines.append(line)
        line_count += 1
    
    text = '\n'.join(lines)
    console.print(f"\n[green]✓[/green] Received {len(text)} characters of text")
    return text

def get_document_input() -> str:
    """Get document input from user."""
    console.print("\n[bold cyan]📁 DOCUMENT UPLOAD MODE[/bold cyan]")
    console.print("Supported formats: PDF, DOCX, TXT, CSV, XLSX, JPG, PNG\n")
    
    while True:
        file_path = Prompt.ask("[bold green]Enter the path to your document[/bold green]")
        
        if not os.path.isfile(file_path):
            console.print(f"[bold red]❌ File not found: {file_path}[/bold red]")
            if not Confirm.ask("[yellow]Try again?[/yellow]"):
                return None
            continue
        
        # Check file extension
        ext = Path(file_path).suffix.lower()
        supported_extensions = ['.pdf', '.docx', '.txt', '.csv', '.xlsx', '.xls', '.jpg', '.jpeg', '.png']
        
        if ext not in supported_extensions:
            console.print(f"[bold red]❌ Unsupported file type: {ext}[/bold red]")
            console.print(f"Supported types: {', '.join(supported_extensions)}")
            if not Confirm.ask("[yellow]Try again?[/yellow]"):
                return None
            continue
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Extracting text from document...", total=None)
                
                text = extract_text_from_document(file_path)
                
                progress.update(task, description="Text extraction completed!")
            
            console.print(f"[green]✓[/green] Successfully extracted {len(text)} characters from: {Path(file_path).name}")
            return text
            
        except Exception as e:
            console.print(f"[bold red]❌ Error extracting text: {e}[/bold red]")
            if not Confirm.ask("[yellow]Try again?[/yellow]"):
                return None

def display_system_status():
    """Display system status including Qdrant and database."""
    console.print("\n[bold cyan]📊 SYSTEM STATUS[/bold cyan]")
    
    # Check database
    try:
        init_db()
        console.print("[green]✓[/green] SQLite database: Connected")
    except Exception as e:
        console.print(f"[red]✗[/red] SQLite database: Error - {e}")
    
    # Check Qdrant
    try:
        from medical_analysis.utils.vector_store import MedicalVectorStore
        vector_store = MedicalVectorStore()
        
        if vector_store.client:
            stats = vector_store.get_collection_stats()
            if stats:
                console.print(f"[green]✓[/green] Qdrant vector database: Connected")
                console.print(f"   Collection: {stats.get('name', 'N/A')}")
                console.print(f"   Vectors: {stats.get('vectors_count', 0)}")
                console.print(f"   Points: {stats.get('points_count', 0)}")
            else:
                console.print("[yellow]⚠[/yellow] Qdrant: Connected but no stats available")
        else:
            console.print("[red]✗[/red] Qdrant vector database: Not connected")
    except Exception as e:
        console.print(f"[red]✗[/red] Qdrant vector database: Error - {e}")
    
    # Check API keys
    config = get_config()
    gemini_key = os.getenv("GEMINI_API_KEY") or config.get('api_keys', {}).get('gemini', '')
    groq_key = os.getenv("GROQ_API_KEY") or config.get('api_keys', {}).get('groq', '')
    
    console.print(f"[green]✓[/green] Gemini API: {'Available' if gemini_key else 'Not configured'}")
    console.print(f"[green]✓[/green] Groq API: {'Available' if groq_key else 'Not configured'}")
    
    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()

def display_help():
    """Display help information."""
    help_text = """
[bold]Medical Diagnostics AI System - Help[/bold]

[bright_blue]What this system does:[/bright_blue]
• Analyzes medical text and documents using AI
• Provides detailed medical reports with multiple specialist perspectives
• Allows interactive chat about the analysis
• Uses vector search for enhanced context understanding

[bright_blue]Supported file formats:[/bright_blue]
• PDF documents
• Word documents (DOCX)
• Text files (TXT)
• Spreadsheets (CSV, XLSX)
• Images (JPG, PNG) - OCR supported

[bright_blue]Chat features:[/bright_blue]
• Ask questions about your medical analysis
• Get general medical information
• Context-aware responses using vector search
• No medication prescriptions (safety first!)

[bright_blue]Commands in chat:[/bright_blue]
• Type your question normally
• 'history' - Show recent conversation
• 'help' - Show chat help
• 'quit' or 'exit' - End chat session

[bright_blue]Important notes:[/bright_blue]
• This is for educational and informational purposes only
• Always consult healthcare professionals for medical advice
• The system does not prescribe medications
• All data is stored locally for privacy
"""
    
    panel = Panel(help_text, title="[bold green]Help[/bold green]", border_style="green")
    console.print(panel)
    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()

def process_analysis(text: str) -> tuple[str, str]:
    """Process the analysis and return session_id and report."""
    config = get_config()
    
    # Chunk and summarize if needed
    tokenizer = get_default_tokenizer()
    max_tokens = config['models']['max_tokens']
    overlap = config['models']['chunk_overlap']
    chunks = chunk_text(text, tokenizer=tokenizer, max_tokens=max_tokens, overlap=overlap)
    aggregated_text = '\n\n'.join(chunks)
    
    # Generate analysis
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating medical analysis...", total=None)
        
        orchestrator = OrchestratorAgent()
        session_id = str(uuid.uuid4())
        review_agent = ReviewAgent(threshold=config.get('review_threshold', 0.7))
        attempt = 0
        
        while True:
            attempt += 1
            progress.update(task, description=f"Analysis attempt {attempt}...")
            
            report = orchestrator.orchestrate(aggregated_text)
            status, score = review_agent.review(text, report)
            save_analysis(session_id, text, report, status, score)
            
            if status == "approved":
                progress.update(task, description="Analysis completed successfully!")
                break
            
            if attempt >= 3:
                progress.update(task, description="Analysis completed with warnings")
                break
    
    return session_id, report

def chat_interface(conversation_id: str, chat_manager: ChatManager):
    """Enhanced chat interface with Rich console."""
    console.print("\n" + "="*80)
    console.print("[bold cyan]🤖 CHAT INTERFACE[/bold cyan] - Ask questions about your medical analysis")
    console.print("="*80)
    
    help_text = """
[bold]You can ask questions about:[/bold]
• The analysis report and findings
• General medical information  
• Specific symptoms or conditions
• Treatment recommendations (general only)
• Follow-up care suggestions

[bold]Commands:[/bold]
• Type your question normally
• [cyan]history[/cyan] - Show recent conversation
• [cyan]help[/cyan] - Show this help
• [cyan]quit[/cyan] or [cyan]exit[/cyan] - End chat session

[bold red]⚠️  IMPORTANT:[/bold red] This system provides general information only.
For specific medical advice, always consult healthcare professionals.
"""
    
    console.print(Panel(help_text, title="[bold green]Chat Help[/bold green]", border_style="green"))
    console.print("-"*80)
    
    while True:
        try:
            user_input = console.input("\n[bold green]❓ Your question: [/bold green]").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("\n[bold green]👋 Chat session ended. Thank you for using the Medical Analysis System![/bold green]")
                break
            
            elif user_input.lower() == 'history':
                console.print("\n[bold cyan]📚 Recent Conversation History:[/bold cyan]")
                console.print("-"*40)
                history = chat_manager.get_chat_history(conversation_id, limit=10)
                if history:
                    for msg in history[-10:]:
                        role, content, agent_used = msg[1], msg[2], msg[3] or 'System'
                        console.print(f"[bold]{role.title()}:[/bold] {content[:100]}{'...' if len(content) > 100 else ''}")
                        if agent_used and agent_used != 'System':
                            console.print(f"   [dim](Answered by: {agent_used})[/dim]")
                else:
                    console.print("[dim]No conversation history yet.[/dim]")
                continue
            
            elif user_input.lower() == 'help':
                console.print(Panel(help_text, title="[bold green]Chat Help[/bold green]", border_style="green"))
                continue
            
            elif not user_input:
                console.print("[yellow]Please enter a question.[/yellow]")
                continue
            
            # Process the question
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing your question...", total=None)
                result = chat_manager.ask_question(conversation_id, user_input)
            
            if result['success']:
                # Create response panel
                agent_info = f"[dim]via {result['agent_used']} specialist[/dim]"
                confidence_info = f"[dim]Confidence: {result['confidence']:.1%}[/dim]"
                
                if result.get('vector_context_used'):
                    confidence_info += " [green]✓ Vector search used[/green]"
                
                response_text = Text()
                response_text.append("🤖 AI Assistant ", style="bold blue")
                response_text.append(agent_info, style="dim")
                response_text.append(f"\n{confidence_info}\n", style="dim")
                response_text.append("-"*60 + "\n", style="dim")
                response_text.append(result['response'], style="white")
                
                panel = Panel(
                    response_text,
                    title="[bold green]Response[/bold green]",
                    border_style="green",
                    box=box.ROUNDED
                )
                console.print(panel)
            else:
                console.print(f"[bold red]❌ Error: {result.get('error', 'Unknown error')}[/bold red]")
                console.print("Please try again or contact support.")
                
        except KeyboardInterrupt:
            console.print("\n\n[bold green]👋 Chat session interrupted. Thank you for using the Medical Analysis System![/bold green]")
            break
        except Exception as e:
            console.print(f"\n[bold red]❌ Unexpected error: {e}[/bold red]")
            console.print("Please try again.")

def main():
    """Main application loop."""
    logger = get_logger("main")
    config = get_config()
    
    # Initialize database
    try:
        init_db()
        logger.info("Medical Diagnostics CLI started.")
    except Exception as e:
        console.print(f"[bold red]❌ Failed to initialize database: {e}[/bold red]")
        return
    
    # Display welcome
    display_welcome()
    
    while True:
        try:
            # Display main menu
            choice = display_main_menu()
            
            if choice == "1":  # Text input
                text = get_text_input()
                if not text:
                    continue
                file_path = None
                
            elif choice == "2":  # Document upload
                text = get_document_input()
                if not text:
                    continue
                file_path = "document_upload"
                
            elif choice == "3":  # System status
                display_system_status()
                continue
                
            elif choice == "4":  # Help
                display_help()
                continue
            
            # Process analysis
            console.print("\n[bold cyan]🔬 GENERATING MEDICAL ANALYSIS[/bold cyan]")
            session_id, report = process_analysis(text)
            
            # Display report
            console.print("\n" + "="*80)
            console.print("[bold green]📋 FINAL MEDICAL ANALYSIS REPORT[/bold green]")
            console.print("="*80)
            
            # Create a scrollable panel for the report
            report_panel = Panel(
                report,
                title="[bold green]Analysis Report[/bold green]",
                border_style="green",
                box=box.ROUNDED,
                width=console.width - 4
            )
            console.print(report_panel)
            console.print("="*80)
            
            logger.info("Report presented to user.")
            
            # Store context in vector database
            try:
                chat_manager = ChatManager()
                chat_manager.store_session_context(session_id, text, report)
            except Exception as e:
                logger.warning(f"Failed to store context in vector database: {e}")
            
            # Offer chat
            if Confirm.ask("\n[bold green]💬 Would you like to ask questions about your analysis?[/bold green]"):
                try:
                    # Initialize chat manager
                    chat_manager = ChatManager()
                    
                    # Create conversation for this session
                    conversation_title = f"Analysis Session {session_id[:8]}"
                    conversation_id = chat_manager.start_conversation(session_id, conversation_title)
                    
                    console.print(f"[green]✅[/green] Chat session started! Conversation ID: {conversation_id}")
                    
                    # Start interactive chat
                    chat_interface(conversation_id, chat_manager)
                    
                except Exception as e:
                    console.print(f"[bold red]❌ Failed to start chat session: {e}[/bold red]")
                    logger.error(f"Chat session failed: {e}")
            else:
                console.print("\n[bold green]👋 Thank you for using the Medical Analysis System![/bold green]")
                console.print("[dim]💡 You can always start a new session to ask questions later.[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n\n[bold green]👋 Thank you for using Medical Diagnostics AI![/bold green]")
            break
        except Exception as e:
            console.print(f"\n[bold red]❌ Unexpected error: {e}[/bold red]")
            logger.error(f"Unexpected error in main loop: {e}")
            if not Confirm.ask("[yellow]Continue with new session?[/yellow]"):
                break

if __name__ == "__main__":
    main()
