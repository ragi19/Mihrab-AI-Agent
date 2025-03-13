"""
Example demonstrating a multi-agent setup with Groq provider, memory capabilities, and task execution
"""
import os
import asyncio
import sys
from pathlib import Path

# Add the src directory to path if running from the repository
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from llm_agents.core.memory_task_agent import MemoryEnabledTaskAgent
from llm_agents.models.multi_provider import MultiProviderModel, OptimizationStrategy
from llm_agents.runtime.memory_runner import MemoryAgentRunner
from llm_agents.models.base import ModelCapability
from llm_agents.tools.standard import (
    CalculatorTool, 
    DateTimeTool, 
    WebSearchTool
)
from llm_agents.core.message import Message, MessageRole
from llm_agents.utils.logging import configure_logging, get_logger
from llm_agents.utils.tracing import FileTraceProvider

# Configure logging
configure_logging(console_level="INFO")
logger = get_logger("groq_memory_task_example")

# Optional OpenAI API key for fallback
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

async def setup_multi_provider_model() -> MultiProviderModel:
    """Create a multi-provider model with Groq as primary and OpenAI as fallback"""
    fallback_models = []
    if OPENAI_API_KEY:
        fallback_models.append("gpt-3.5-turbo")
        
    # Create a multi-provider model
    model = await MultiProviderModel.create(
        primary_model="llama3-70b-8192",  # Groq's Llama 3 model
        fallback_models=fallback_models,
        required_capabilities={ModelCapability.CHAT},
        optimize_for=OptimizationStrategy.PERFORMANCE
    )
    
    return model

async def main():
    # Check for API keys
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("GROQ_API_KEY environment variable not found. Please set it before running this example.")
        print("You can get an API key from https://console.groq.com/keys")
        return
    
    # Initialize tracing for monitoring agent behavior
    trace_file = Path("traces") / "groq_memory_trace.json"
    trace_file.parent.mkdir(exist_ok=True)
    trace_provider = FileTraceProvider(file_path=str(trace_file))
    
    logger.info("Initializing multi-provider model...")
    
    # Create multi-provider model with Groq as primary
    model = await setup_multi_provider_model()
    
    logger.info(f"Using provider: {model.current_provider}")
    
    # Create tools for the agent
    tools = [
        CalculatorTool(),
        DateTimeTool(),
        WebSearchTool(api_key=os.environ.get("SERPER_API_KEY"))  # Optional web search
    ]
    
    # Create memory-enabled task agent
    agent = MemoryEnabledTaskAgent(
        model=model,
        system_message=(
            "You are a helpful assistant with memory capabilities. "
            "You can remember past interactions and use tools to help answer questions."
        ),
        tools=tools,
        max_memory_items=50,
        memory_retrieval_count=3,
        automatic_memory=True,
        trace_provider=trace_provider
    )
    
    # Create memory-enabled runner
    memory_dir = Path("./memories")
    memory_dir.mkdir(exist_ok=True)
    
    runner = MemoryAgentRunner(
        agent=agent,
        trace_provider=trace_provider,
        memory_persistence_path=str(memory_dir),
        auto_save_interval_seconds=60
    )
    
    # Start interactive session with memory
    session_id = "groq_memory_session"
    
    print("\n===== Groq Memory Task Agent Demo =====")
    print("Type 'exit' to quit, 'stats' to view provider stats, or 'clear' to reset memory\n")
    
    try:
        # Load previous session if it exists
        loaded_count = await runner.load_memory(session_id)
        if loaded_count > 0:
            print(f"Loaded {loaded_count} memories from previous session")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == "exit":
                break
                
            elif user_input.lower() == "stats":
                stats = model.get_provider_stats()
                print("\n----- Provider Statistics -----")
                for provider, metrics in stats.items():
                    print(f"\n{provider}:")
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            print(f"  {k}: {v:.4f}")
                        else:
                            print(f"  {k}: {v}")
                continue
                
            elif user_input.lower() == "clear":
                await runner.clear_conversation()
                print("Memory and conversation history cleared")
                continue
                
            # Process user message
            print("\nProcessing...")
            response = await runner.run(user_input, session_id=session_id)
            
            # Display response
            print(f"\nAssistant ({model.current_provider}): {response.content}")
        
        # Save session memory before exit
        await runner.save_memory(session_id)
        print(f"\nSession saved. Memory items: {runner.get_memory_stats()['total_memories']}")
        
    except KeyboardInterrupt:
        print("\nExiting...")
        await runner.save_memory(session_id)
        print("Session saved")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())