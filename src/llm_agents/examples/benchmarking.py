"""
Example of benchmarking and profiling different models in the LLM Agents framework
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv

from llm_agents import Message, MessageRole, create_agent
from llm_agents.models.base import ModelCapability

# Load environment variables from .env file
load_dotenv()

# Sample benchmark questions covering different domains
BENCHMARK_QUESTIONS = [
    "Explain the concept of quantum computing in simple terms.",
    "What are three effective strategies for improving time management?",
    "Write a short poem about artificial intelligence.",
    "Describe the process of photosynthesis in plants.",
    "What are the key differences between machine learning and deep learning?",
    "Explain how blockchain technology works.",
    "What are the main causes of climate change?",
    "Provide a brief history of the internet.",
    "What are the ethical considerations in artificial intelligence development?",
    "Explain the concept of compound interest in finance.",
]


class ModelProfiler:
    """Utility for profiling and benchmarking LLM models"""

    def __init__(self):
        self.results = {}

    async def profile_model(
        self, provider_name: str, model_name: str, questions: List[str]
    ):
        """
        Profile a model's performance on a set of questions

        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            questions: List of questions to benchmark

        Returns:
            Dictionary of performance metrics
        """
        print(f"\n=== Profiling {provider_name.upper()} ({model_name}) ===")

        # Create an agent with the specified provider and model
        try:
            agent = await create_agent(
                provider_name=provider_name,
                model_name=model_name,
                system_message="You are a helpful AI assistant. Provide concise and accurate responses.",
            )
        except Exception as e:
            print(f"Error creating agent: {e}")
            return None

        metrics = {
            "provider": provider_name,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "questions": len(questions),
            "successful": 0,
            "failed": 0,
            "total_tokens": 0,
            "total_time": 0,
            "average_time": 0,
            "results": [],
        }

        # Process each question and measure performance
        for i, question in enumerate(questions):
            print(f"\nQuestion {i+1}/{len(questions)}: {question[:50]}...")

            # Create message
            message = Message(role=MessageRole.USER, content=question)

            # Measure performance
            try:
                start_time = time.time()
                response = await agent.process_message(message)
                end_time = time.time()

                # Calculate metrics
                elapsed_time = end_time - start_time
                response_length = len(response.content)

                # Get token usage if available
                prompt_tokens = getattr(response, "prompt_tokens", 0) or 0
                completion_tokens = getattr(response, "completion_tokens", 0) or 0
                total_tokens = prompt_tokens + completion_tokens

                # Store result
                result = {
                    "question": question,
                    "response": (
                        response.content[:100] + "..."
                        if len(response.content) > 100
                        else response.content
                    ),
                    "response_length": response_length,
                    "time": elapsed_time,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "success": True,
                }

                metrics["successful"] += 1
                metrics["total_time"] += elapsed_time
                metrics["total_tokens"] += total_tokens

                print(f"  Time: {elapsed_time:.2f}s, Tokens: {total_tokens}")

            except Exception as e:
                # Record failure
                result = {"question": question, "error": str(e), "success": False}

                metrics["failed"] += 1
                print(f"  Error: {e}")

            metrics["results"].append(result)

        # Calculate averages
        if metrics["successful"] > 0:
            metrics["average_time"] = metrics["total_time"] / metrics["successful"]
            metrics["average_tokens"] = metrics["total_tokens"] / metrics["successful"]

        # Store results
        model_key = f"{provider_name}_{model_name}"
        self.results[model_key] = metrics

        # Print summary
        print(f"\n--- {model_name} Summary ---")
        print(f"Successful: {metrics['successful']}/{len(questions)}")
        print(f"Average time: {metrics['average_time']:.2f}s")
        if metrics["total_tokens"] > 0:
            print(f"Total tokens: {metrics['total_tokens']}")
            print(f"Average tokens per question: {metrics['average_tokens']:.1f}")

        return metrics

    def compare_models(self):
        """Compare all profiled models and print a comparison table"""
        if not self.results:
            print("No models have been profiled yet.")
            return

        print("\n=== Model Comparison ===")
        print(
            f"{'Provider':<12} {'Model':<20} {'Success':<10} {'Avg Time':<10} {'Avg Tokens':<12}"
        )
        print("-" * 70)

        for model_key, metrics in self.results.items():
            provider = metrics["provider"]
            model = metrics["model"]
            success_rate = f"{metrics['successful']}/{metrics['questions']}"
            avg_time = f"{metrics['average_time']:.2f}s"

            if "average_tokens" in metrics:
                avg_tokens = f"{metrics['average_tokens']:.1f}"
            else:
                avg_tokens = "N/A"

            print(
                f"{provider:<12} {model:<20} {success_rate:<10} {avg_time:<10} {avg_tokens:<12}"
            )

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to a JSON file"""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nBenchmark results saved to {filename}")


async def run_benchmark():
    """Run benchmarks on available models"""
    profiler = ModelProfiler()

    # Define models to benchmark based on available API keys
    models_to_benchmark = []

    if os.getenv("OPENAI_API_KEY"):
        models_to_benchmark.extend([("openai", "gpt-3.5-turbo"), ("openai", "gpt-4")])

    if os.getenv("ANTHROPIC_API_KEY"):
        models_to_benchmark.extend(
            [("anthropic", "claude-instant-1"), ("anthropic", "claude-2")]
        )

    if os.getenv("GROQ_API_KEY"):
        models_to_benchmark.extend(
            [("groq", "llama3-8b-8192"), ("groq", "llama3-70b-8192")]
        )

    if not models_to_benchmark:
        print(
            "\nNo API keys found. Please set at least one of the following environment variables:"
        )
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
        print("- GROQ_API_KEY")
        return

    # Run benchmarks for each model
    for provider_name, model_name in models_to_benchmark:
        try:
            await profiler.profile_model(provider_name, model_name, BENCHMARK_QUESTIONS)
        except Exception as e:
            print(f"Error benchmarking {provider_name} {model_name}: {e}")

    # Compare results
    profiler.compare_models()

    # Save results
    profiler.save_results()


async def main():
    """Run the benchmarking example"""
    await run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
