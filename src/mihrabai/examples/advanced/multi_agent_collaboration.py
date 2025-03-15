"""
Multi-Agent Collaboration Example

This example demonstrates how to create multiple specialized agents that collaborate
on complex tasks. It shows how to create a research agent, an analysis agent, and
a writing agent that work together to create a comprehensive report.
"""

import asyncio
import os
import sys
import tempfile
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

from mihrabai import create_task_agent, Message, MessageRole
from mihrabai.models.base import ModelCapability
from mihrabai.tools.registry import ToolRegistry
from mihrabai.tools.standard.web import HTTPRequestTool, WebScraperTool
from mihrabai.tools.standard.filesystem import FileReaderTool, FileWriterTool
from mihrabai.tools.base import BaseTool
from mihrabai.core.types import JSON

# Load environment variables
load_dotenv()

# Get Groq API key
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("Groq API key not found in environment variables.")
    groq_api_key = input("Please enter your Groq API key: ")
    if not groq_api_key:
        print("Error: Groq API key is required to run this example.")
        sys.exit(1)

# Define a custom tool for agent-to-agent communication
class AgentCommunicationTool(BaseTool):
    def __init__(self, target_agent_name: str, target_agent, task_tracker=None):
        super().__init__(
            name=f"ask_{target_agent_name}",
            description=f"Send a question or request to the {target_agent_name} agent"
        )
        self.target_agent = target_agent
        self.target_agent_name = target_agent_name
        self.task_tracker = task_tracker
    
    async def execute(self, parameters: Dict[str, Any]) -> JSON:
        """Execute the agent communication tool with the given parameters"""
        return await self._execute(parameters)
    
    async def _execute(self, parameters: Dict[str, Any]) -> JSON:
        question = parameters.get("question", "")
        context = parameters.get("context", "")
        
        if not question:
            return {
                "status": "error",
                "message": "No question provided for the target agent"
            }
        
        # Create a message for the target agent
        message = Message(
            role=MessageRole.USER,
            content=f"[Request from another agent] {question}\n\nContext: {context}"
        )
        
        # Track the task if a tracker is provided
        if self.task_tracker:
            task_id = self.task_tracker.start_task(
                agent=self.target_agent_name,
                task_type="communication",
                description=question[:50] + "..." if len(question) > 50 else question
            )
        
        # Process the message with the target agent
        response = await self.target_agent.process_message(message)
        
        # Mark task as complete if tracking
        if self.task_tracker and 'task_id' in locals():
            self.task_tracker.complete_task(task_id)
        
        return {
            "status": "success",
            "agent": self.target_agent_name,
            "question": question,
            "response": response.content
        }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or request to send to the target agent"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context information to provide to the target agent"
                }
            },
            "required": ["question"]
        }

# Define a task tracker to monitor progress
class TaskTracker:
    def __init__(self):
        self.tasks = {}
        self.task_id_counter = 0
        self.completed_tasks = 0
        self.total_tasks = 0
        self.start_time = time.time()
    
    def start_task(self, agent: str, task_type: str, description: str) -> int:
        """Start a new task and return its ID"""
        task_id = self.task_id_counter
        self.task_id_counter += 1
        self.total_tasks += 1
        
        self.tasks[task_id] = {
            "agent": agent,
            "type": task_type,
            "description": description,
            "status": "in_progress",
            "start_time": time.time(),
            "end_time": None
        }
        
        print(f"[Task {task_id}] Started: {agent} - {description}")
        return task_id
    
    def complete_task(self, task_id: int) -> None:
        """Mark a task as completed"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["end_time"] = time.time()
            self.completed_tasks += 1
            
            duration = self.tasks[task_id]["end_time"] - self.tasks[task_id]["start_time"]
            print(f"[Task {task_id}] Completed: {self.tasks[task_id]['agent']} - {self.tasks[task_id]['description']} (took {duration:.2f}s)")
    
    def fail_task(self, task_id: int, reason: str) -> None:
        """Mark a task as failed"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["end_time"] = time.time()
            self.tasks[task_id]["failure_reason"] = reason
            
            print(f"[Task {task_id}] Failed: {self.tasks[task_id]['agent']} - {self.tasks[task_id]['description']} - Reason: {reason}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get the current progress of all tasks"""
        elapsed_time = time.time() - self.start_time
        
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "in_progress_tasks": sum(1 for t in self.tasks.values() if t["status"] == "in_progress"),
            "failed_tasks": sum(1 for t in self.tasks.values() if t["status"] == "failed"),
            "completion_percentage": (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0,
            "elapsed_time": elapsed_time
        }
    
    def print_summary(self) -> None:
        """Print a summary of all tasks"""
        progress = self.get_progress()
        
        print("\n=== Task Execution Summary ===")
        print(f"Total tasks: {progress['total_tasks']}")
        print(f"Completed tasks: {progress['completed_tasks']}")
        print(f"Failed tasks: {progress['failed_tasks']}")
        print(f"Completion percentage: {progress['completion_percentage']:.2f}%")
        print(f"Total execution time: {progress['elapsed_time']:.2f} seconds")
        
        # Print tasks by agent
        agent_tasks = {}
        for task in self.tasks.values():
            agent = task["agent"]
            if agent not in agent_tasks:
                agent_tasks[agent] = {"completed": 0, "failed": 0, "in_progress": 0}
            
            if task["status"] == "completed":
                agent_tasks[agent]["completed"] += 1
            elif task["status"] == "failed":
                agent_tasks[agent]["failed"] += 1
            else:
                agent_tasks[agent]["in_progress"] += 1
        
        print("\nTasks by agent:")
        for agent, counts in agent_tasks.items():
            print(f"  {agent}: {counts['completed']} completed, {counts['failed']} failed, {counts['in_progress']} in progress")

async def main():
    # Create a task tracker
    task_tracker = TaskTracker()
    
    # Create a temporary directory for our example
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Create tool instances
        http_tool = HTTPRequestTool()
        scraper_tool = WebScraperTool()
        file_read_tool = FileReaderTool()
        file_write_tool = FileWriterTool()
        
        # Create specialized agents
        
        # 1. Research Agent - Specializes in finding information
        research_agent = await create_task_agent(
            provider_name="groq",
            model_name="llama3-70b-8192",
            system_message="""You are a specialized research agent that excels at finding information.
            Your task is to search for relevant information on topics and provide comprehensive research results.
            Focus on finding factual information from reliable sources. Include citations when possible.
            
            When asked to research a topic, provide detailed information with facts and figures.
            Structure your response in a way that will be useful for analysis.""",
            provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
            tools=[http_tool, scraper_tool]  # Pass tools directly
        )
        
        # 2. Analysis Agent - Specializes in analyzing information
        analysis_agent = await create_task_agent(
            provider_name="groq",
            model_name="llama3-70b-8192",
            system_message="""You are a specialized analysis agent that excels at analyzing information.
            Your task is to analyze research data, identify patterns, extract insights, and draw conclusions.
            Focus on critical thinking and providing thoughtful analysis.
            
            When given research information, analyze it thoroughly and provide insights that would be valuable
            for a written report. Structure your analysis in sections that can be directly used by a writing agent.""",
            provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
            tools=[]  # Will add communication tools after creation
        )
        
        # 3. Writing Agent - Specializes in writing reports
        writing_agent = await create_task_agent(
            provider_name="groq",
            model_name="llama3-70b-8192",
            system_message="""You are a specialized writing agent that excels at creating well-structured content.
            Your task is to take research and analysis and create comprehensive, well-written reports.
            Focus on clarity, organization, and engaging writing style.
            
            When asked to create a report, use the provided research and analysis to create a cohesive document.
            Always save your final report to the specified file path using the file_writer tool.
            Make sure to include all sections requested in the original task.""",
            provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
            tools=[file_write_tool, file_read_tool]  # Pass tools directly
        )
        
        # Create communication tools for agent-to-agent interaction
        research_comm_tool = AgentCommunicationTool("research", research_agent, task_tracker)
        analysis_comm_tool = AgentCommunicationTool("analysis", analysis_agent, task_tracker)
        writing_comm_tool = AgentCommunicationTool("writing", writing_agent, task_tracker)
        
        # Recreate analysis agent with the research communication tool
        analysis_agent = await create_task_agent(
            provider_name="groq",
            model_name="llama3-70b-8192",
            system_message="""You are a specialized analysis agent that excels at analyzing information.
            Your task is to analyze research data, identify patterns, extract insights, and draw conclusions.
            Focus on critical thinking and providing thoughtful analysis.
            
            When given research information, analyze it thoroughly and provide insights that would be valuable
            for a written report. Structure your analysis in sections that can be directly used by a writing agent.""",
            provider_kwargs={"api_key": groq_api_key},
            tools=[research_comm_tool]  # Include the research communication tool
        )
        
        # Recreate writing agent with both research and analysis communication tools
        writing_agent = await create_task_agent(
            provider_name="groq",
            model_name="llama3-70b-8192",
            system_message="""You are a specialized writing agent that excels at creating well-structured content.
            Your task is to take research and analysis and create comprehensive, well-written reports.
            Focus on clarity, organization, and engaging writing style.
            
            When asked to create a report, use the provided research and analysis to create a cohesive document.
            Always save your final report to the specified file path using the file_writer tool.
            Make sure to include all sections requested in the original task.""",
            provider_kwargs={"api_key": groq_api_key},
            tools=[file_write_tool, file_read_tool, research_comm_tool, analysis_comm_tool]  # Include all needed tools
        )
        
        # Update the communication tools with the new agent instances
        analysis_comm_tool.target_agent = analysis_agent
        writing_comm_tool.target_agent = writing_agent
        
        # Create a coordinator agent that orchestrates the collaboration
        coordinator_agent = await create_task_agent(
            provider_name="groq",
            model_name="llama3-70b-8192",
            system_message="""You are a coordinator agent that orchestrates collaboration between specialized agents.
            You have access to:
            1. A research agent that can find information on topics
            2. An analysis agent that can analyze information and extract insights
            3. A writing agent that can create well-structured reports
            
            Your task is to coordinate these agents to complete complex tasks that require research, analysis, and writing.
            Break down tasks into appropriate subtasks for each specialized agent.
            
            IMPORTANT: You must execute each step in your plan by using the appropriate communication tools.
            For each step:
            1. Use the ask_research tool to get information from the research agent
            2. Use the ask_analysis tool to get analysis from the analysis agent
            3. Use the ask_writing tool to get the writing agent to create the final report
            
            Make sure to pass the output from one agent to the next as context.
            Track the progress of each step and ensure the final report is created.""",
            provider_kwargs={"api_key": groq_api_key},  # Explicitly pass the API key
            tools=[research_comm_tool, analysis_comm_tool, writing_comm_tool]  # Pass tools directly
        )
        
        # Example: Create a report on renewable energy
        coordinator_message = Message(
            role=MessageRole.USER,
            content=f"""Create a brief report on renewable energy trends. The report should include:
            1. Current state of renewable energy adoption globally
            2. Analysis of the most promising renewable technologies
            3. Future outlook and challenges
            
            The final report should be saved as 'renewable_energy_report.txt' in the directory: {temp_dir}."""
        )
        
        print("Starting multi-agent collaboration task...")
        print("This may take some time as multiple agents work together...")
        
        # Start the main coordination task
        main_task_id = task_tracker.start_task(
            agent="coordinator",
            task_type="coordination",
            description="Create renewable energy report"
        )
        
        # Process the message with the coordinator agent
        coordinator_response = await coordinator_agent.process_message(coordinator_message)
        
        # Mark the main task as complete
        task_tracker.complete_task(main_task_id)
        
        print(f"Coordinator: {coordinator_response.content}\n")
        
        # Print task execution summary
        task_tracker.print_summary()
        
        # Read the final report
        try:
            with open(os.path.join(temp_dir, "renewable_energy_report.txt"), "r") as f:
                report_content = f.read()
            
            print("\n--- FINAL REPORT ---\n")
            print(report_content)
        except FileNotFoundError:
            print("Report file not found. The writing agent may not have created it yet.")
            
            # Attempt to directly instruct the writing agent to create the report if it wasn't created
            if "renewable_energy_report.txt" not in os.listdir(temp_dir):
                print("\nAttempting to directly create the report with the writing agent...")
                
                # Get research information
                research_message = Message(
                    role=MessageRole.USER,
                    content="Research the current state of renewable energy adoption globally, the most promising renewable technologies, and future outlook and challenges."
                )
                research_response = await research_agent.process_message(research_message)
                
                # Get analysis
                analysis_message = Message(
                    role=MessageRole.USER,
                    content=f"Analyze this research on renewable energy:\n\n{research_response.content}"
                )
                analysis_response = await analysis_agent.process_message(analysis_message)
                
                # Create the report
                writing_message = Message(
                    role=MessageRole.USER,
                    content=f"""Create a brief report on renewable energy trends using this research and analysis.
                    
                    Research:
                    {research_response.content}
                    
                    Analysis:
                    {analysis_response.content}
                    
                    The report should include:
                    1. Current state of renewable energy adoption globally
                    2. Analysis of the most promising renewable technologies
                    3. Future outlook and challenges
                    
                    IMPORTANT: You MUST save the final report as 'renewable_energy_report.txt' in the directory: {temp_dir}
                    Use the file_writer tool with the exact path: {os.path.join(temp_dir, "renewable_energy_report.txt")}
                    
                    After writing the file, confirm that you have saved it to the specified location."""
                )
                writing_response = await writing_agent.process_message(writing_message)
                
                print(f"Writing Agent: {writing_response.content}\n")
                
                # Try to read the report again
                try:
                    with open(os.path.join(temp_dir, "renewable_energy_report.txt"), "r") as f:
                        report_content = f.read()
                    
                    print("\n--- FINAL REPORT ---\n")
                    print(report_content)
                except FileNotFoundError:
                    print("Still unable to create the report. Attempting to write the file directly...")
                    
                    # Create a basic report from the responses we have
                    basic_report = f"""# Renewable Energy Trends Report

## Current State of Renewable Energy Adoption Globally
{research_response.content}

## Analysis of the Most Promising Renewable Technologies
{analysis_response.content}

## Future Outlook and Challenges
Based on the research and analysis above, renewable energy adoption faces both opportunities and challenges.
Opportunities include decreasing costs, technological improvements, and increasing policy support.
Challenges include grid integration, energy storage, and the need for continued investment.
"""
                    
                    # Write the file directly
                    try:
                        with open(os.path.join(temp_dir, "renewable_energy_report.txt"), "w") as f:
                            f.write(basic_report)
                        
                        print("Successfully created a basic report directly.")
                        
                        # Read and display the report
                        with open(os.path.join(temp_dir, "renewable_energy_report.txt"), "r") as f:
                            report_content = f.read()
                        
                        print("\n--- FINAL REPORT ---\n")
                        print(report_content)
                    except Exception as e:
                        print(f"Error writing file directly: {e}")
                        print(f"Temp directory permissions: {os.access(temp_dir, os.W_OK)}")
                        print(f"Files in temp directory: {os.listdir(temp_dir)}")

if __name__ == "__main__":
    asyncio.run(main()) 