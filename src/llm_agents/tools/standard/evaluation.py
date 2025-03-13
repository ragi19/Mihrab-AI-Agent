"""
Evaluation tools for assessing model performance and outputs
"""

import json
import statistics
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from ...utils.logging import get_logger
from ..base import BaseTool

logger = get_logger("tools.evaluation")


class ResponseEvaluatorTool(BaseTool):
    """Tool for evaluating model responses against criteria"""

    def __init__(self):
        super().__init__(
            name="response_evaluator",
            description="Evaluate model responses against defined criteria",
        )
        self._parameters = {
            "response": {
                "type": "string",
                "description": "The model response to evaluate",
            },
            "criteria": {
                "type": "array",
                "description": "List of criteria to evaluate against",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "weight": {"type": "number", "default": 1.0},
                    },
                    "required": ["name", "description"],
                },
            },
            "reference_text": {
                "type": "string",
                "description": "Optional reference text to compare against",
                "default": "",
            },
            "scoring_method": {
                "type": "string",
                "description": "Method to use for scoring",
                "enum": ["rubric", "similarity", "custom"],
                "default": "rubric",
            },
            "max_score": {
                "type": "number",
                "description": "Maximum score per criterion",
                "default": 5.0,
            },
        }
        self._required_params = ["response", "criteria"]
        logger.info("Initialized tool: response_evaluator")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self,
        response: str,
        criteria: List[Dict[str, Any]],
        reference_text: str = "",
        scoring_method: str = "rubric",
        max_score: float = 5.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the response evaluator tool

        Args:
            response: The model response to evaluate
            criteria: List of criteria to evaluate against
            reference_text: Optional reference text to compare against
            scoring_method: Method to use for scoring
            max_score: Maximum score per criterion

        Returns:
            Dictionary with evaluation results
        """
        try:
            # Validate parameters
            if not response:
                return {"error": "Response cannot be empty"}

            if not criteria:
                return {"error": "At least one criterion is required"}

            # Initialize results
            evaluation_results = []
            total_score = 0.0
            total_weight = 0.0

            # Evaluate each criterion
            for criterion in criteria:
                name = criterion.get("name", "Unnamed")
                description = criterion.get("description", "")
                weight = criterion.get("weight", 1.0)

                # Score the criterion based on the method
                if scoring_method == "rubric":
                    score, feedback = self._score_with_rubric(
                        response, name, description, max_score
                    )
                elif scoring_method == "similarity":
                    if not reference_text:
                        return {
                            "error": "Reference text is required for similarity scoring"
                        }
                    score, feedback = self._score_with_similarity(
                        response, reference_text, name, description, max_score
                    )
                elif scoring_method == "custom":
                    # Custom scoring would typically use an external model or API
                    # For now, we'll use a simple heuristic
                    score, feedback = self._score_with_custom(
                        response, name, description, max_score
                    )
                else:
                    return {"error": f"Invalid scoring method: {scoring_method}"}

                # Calculate weighted score
                weighted_score = score * weight
                total_score += weighted_score
                total_weight += weight

                # Add to results
                evaluation_results.append(
                    {
                        "criterion": name,
                        "description": description,
                        "score": score,
                        "max_score": max_score,
                        "weight": weight,
                        "weighted_score": weighted_score,
                        "feedback": feedback,
                    }
                )

            # Calculate overall score
            overall_score = total_score / total_weight if total_weight > 0 else 0.0

            # Determine rating based on score percentage
            score_percentage = (overall_score / max_score) * 100
            if score_percentage >= 90:
                rating = "Excellent"
            elif score_percentage >= 80:
                rating = "Very Good"
            elif score_percentage >= 70:
                rating = "Good"
            elif score_percentage >= 60:
                rating = "Satisfactory"
            elif score_percentage >= 50:
                rating = "Needs Improvement"
            else:
                rating = "Poor"

            return {
                "overall_score": overall_score,
                "max_possible_score": max_score,
                "score_percentage": score_percentage,
                "rating": rating,
                "criteria_results": evaluation_results,
                "response_length": len(response),
                "scoring_method": scoring_method,
                "evaluation_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Response evaluation error: {e}")
            return {"error": f"Response evaluation error: {str(e)}"}

    def _score_with_rubric(
        self,
        response: str,
        criterion_name: str,
        criterion_description: str,
        max_score: float,
    ) -> tuple[float, str]:
        """Score response using a rubric-based approach

        Args:
            response: The model response to evaluate
            criterion_name: Name of the criterion
            criterion_description: Description of the criterion
            max_score: Maximum possible score

        Returns:
            Tuple of (score, feedback)
        """
        # This is a simplified scoring method based on text characteristics
        # In a real implementation, this would use a more sophisticated approach

        # Check if response addresses the criterion
        keywords = criterion_description.lower().split()
        relevant_words = [
            word for word in keywords if len(word) > 3
        ]  # Filter out short words

        # Count how many relevant words appear in the response
        response_lower = response.lower()
        matches = sum(1 for word in relevant_words if word in response_lower)

        # Calculate match percentage
        match_percentage = matches / max(1, len(relevant_words))

        # Score based on match percentage
        score = match_percentage * max_score

        # Generate feedback
        if match_percentage > 0.8:
            feedback = f"Excellent coverage of '{criterion_name}'. The response thoroughly addresses this criterion."
        elif match_percentage > 0.6:
            feedback = f"Good coverage of '{criterion_name}'. The response addresses most aspects of this criterion."
        elif match_percentage > 0.4:
            feedback = f"Partial coverage of '{criterion_name}'. The response addresses some aspects but could be more comprehensive."
        elif match_percentage > 0.2:
            feedback = f"Limited coverage of '{criterion_name}'. The response only minimally addresses this criterion."
        else:
            feedback = f"Poor coverage of '{criterion_name}'. The response does not adequately address this criterion."

        return round(score, 2), feedback

    def _score_with_similarity(
        self,
        response: str,
        reference: str,
        criterion_name: str,
        criterion_description: str,
        max_score: float,
    ) -> tuple[float, str]:
        """Score response using text similarity to a reference

        Args:
            response: The model response to evaluate
            reference: Reference text to compare against
            criterion_name: Name of the criterion
            criterion_description: Description of the criterion
            max_score: Maximum possible score

        Returns:
            Tuple of (score, feedback)
        """
        # This is a simplified similarity calculation using Jaccard similarity
        # In a real implementation, this would use more sophisticated NLP techniques

        # Tokenize texts (simple word-based tokenization)
        response_tokens = set(response.lower().split())
        reference_tokens = set(reference.lower().split())

        # Calculate Jaccard similarity
        intersection = len(response_tokens.intersection(reference_tokens))
        union = len(response_tokens.union(reference_tokens))
        similarity = intersection / max(1, union)

        # Score based on similarity
        score = similarity * max_score

        # Generate feedback
        if similarity > 0.8:
            feedback = f"Excellent similarity for '{criterion_name}'. The response closely matches the reference."
        elif similarity > 0.6:
            feedback = f"Good similarity for '{criterion_name}'. The response is fairly similar to the reference."
        elif similarity > 0.4:
            feedback = f"Moderate similarity for '{criterion_name}'. The response has some similarities to the reference."
        elif similarity > 0.2:
            feedback = f"Low similarity for '{criterion_name}'. The response differs significantly from the reference."
        else:
            feedback = f"Very low similarity for '{criterion_name}'. The response is very different from the reference."

        return round(score, 2), feedback

    def _score_with_custom(
        self,
        response: str,
        criterion_name: str,
        criterion_description: str,
        max_score: float,
    ) -> tuple[float, str]:
        """Score response using custom heuristics

        Args:
            response: The model response to evaluate
            criterion_name: Name of the criterion
            criterion_description: Description of the criterion
            max_score: Maximum possible score

        Returns:
            Tuple of (score, feedback)
        """
        # This is a simplified custom scoring method
        # In a real implementation, this would use more sophisticated techniques

        # Calculate basic metrics
        response_length = len(response)
        sentence_count = response.count(".") + response.count("!") + response.count("?")
        avg_sentence_length = response_length / max(1, sentence_count)

        # Score based on length and structure
        length_score = min(1.0, response_length / 500)  # Normalize to 500 chars
        structure_score = min(1.0, sentence_count / 10)  # Normalize to 10 sentences
        readability_score = min(
            1.0, 20 / max(1, avg_sentence_length)
        )  # Prefer ~20 chars per sentence

        # Combined score
        combined_score = (length_score + structure_score + readability_score) / 3
        score = combined_score * max_score

        # Generate feedback
        feedback = f"Evaluation of '{criterion_name}':\n"
        feedback += (
            f"- Length: {response_length} characters ({int(length_score*100)}%)\n"
        )
        feedback += (
            f"- Structure: {sentence_count} sentences ({int(structure_score*100)}%)\n"
        )
        feedback += f"- Readability: {int(avg_sentence_length)} chars/sentence ({int(readability_score*100)}%)"

        return round(score, 2), feedback


class PerformanceBenchmarkTool(BaseTool):
    """Tool for benchmarking model performance metrics"""

    def __init__(self):
        super().__init__(
            name="performance_benchmark",
            description="Benchmark model performance metrics",
        )
        self._parameters = {
            "metrics": {
                "type": "array",
                "description": "List of performance metrics to benchmark",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "number"},
                        "unit": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "value"],
                },
            },
            "baseline_metrics": {
                "type": "array",
                "description": "Optional baseline metrics for comparison",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "number"},
                        "unit": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "value"],
                },
            },
            "model_name": {
                "type": "string",
                "description": "Name of the model being benchmarked",
            },
            "baseline_name": {
                "type": "string",
                "description": "Name of the baseline model",
                "default": "Baseline",
            },
        }
        self._required_params = ["metrics", "model_name"]
        logger.info("Initialized tool: performance_benchmark")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self,
        metrics: List[Dict[str, Any]],
        model_name: str,
        baseline_metrics: Optional[List[Dict[str, Any]]] = None,
        baseline_name: str = "Baseline",
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the performance benchmark tool

        Args:
            metrics: List of performance metrics to benchmark
            model_name: Name of the model being benchmarked
            baseline_metrics: Optional baseline metrics for comparison
            baseline_name: Name of the baseline model

        Returns:
            Dictionary with benchmark results
        """
        try:
            # Validate parameters
            if not metrics:
                return {"error": "At least one metric is required"}

            # Process metrics
            processed_metrics = []
            for metric in metrics:
                name = metric.get("name", "Unnamed")
                value = metric.get("value")
                unit = metric.get("unit", "")
                description = metric.get("description", "")

                processed_metric = {
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "description": description,
                }

                # Compare with baseline if available
                if baseline_metrics:
                    baseline_metric = next(
                        (m for m in baseline_metrics if m.get("name") == name), None
                    )
                    if baseline_metric:
                        baseline_value = baseline_metric.get("value")
                        if baseline_value is not None and value is not None:
                            # Calculate difference and percentage change
                            difference = value - baseline_value
                            if baseline_value != 0:
                                percentage_change = (difference / baseline_value) * 100
                            else:
                                percentage_change = (
                                    float("inf")
                                    if difference > 0
                                    else float("-inf") if difference < 0 else 0
                                )

                            processed_metric["baseline_value"] = baseline_value
                            processed_metric["difference"] = difference
                            processed_metric["percentage_change"] = round(
                                percentage_change, 2
                            )

                            # Determine if this is an improvement
                            # Assuming higher values are better by default
                            is_higher_better = True

                            # Override for common metrics where lower is better
                            lower_better_metrics = [
                                "latency",
                                "response_time",
                                "error_rate",
                                "token_usage",
                                "memory_usage",
                                "processing_time",
                                "cost",
                            ]

                            for lower_metric in lower_better_metrics:
                                if lower_metric in name.lower():
                                    is_higher_better = False
                                    break

                            if (is_higher_better and difference > 0) or (
                                not is_higher_better and difference < 0
                            ):
                                processed_metric["comparison"] = "better"
                            elif (is_higher_better and difference < 0) or (
                                not is_higher_better and difference > 0
                            ):
                                processed_metric["comparison"] = "worse"
                            else:
                                processed_metric["comparison"] = "same"

                processed_metrics.append(processed_metric)

            # Calculate summary statistics
            improvements = [
                m for m in processed_metrics if m.get("comparison") == "better"
            ]
            regressions = [
                m for m in processed_metrics if m.get("comparison") == "worse"
            ]
            unchanged = [m for m in processed_metrics if m.get("comparison") == "same"]

            # Generate summary
            if baseline_metrics:
                if len(improvements) > len(regressions):
                    summary = f"{model_name} shows overall improvement compared to {baseline_name}"
                elif len(improvements) < len(regressions):
                    summary = f"{model_name} shows overall regression compared to {baseline_name}"
                else:
                    summary = (
                        f"{model_name} shows mixed results compared to {baseline_name}"
                    )

                summary += f" ({len(improvements)} improvements, {len(regressions)} regressions, {len(unchanged)} unchanged)"
            else:
                summary = f"Benchmark results for {model_name} (no baseline comparison)"

            return {
                "model_name": model_name,
                "baseline_name": baseline_name if baseline_metrics else None,
                "metrics": processed_metrics,
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "has_baseline": baseline_metrics is not None,
                "improvement_count": len(improvements),
                "regression_count": len(regressions),
                "unchanged_count": len(unchanged),
            }
        except Exception as e:
            logger.error(f"Performance benchmark error: {e}")
            return {"error": f"Performance benchmark error: {str(e)}"}


class ModelComparisonTool(BaseTool):
    """Tool for comparing multiple model outputs"""

    def __init__(self):
        super().__init__(
            name="model_comparison",
            description="Compare outputs from multiple models on the same input",
        )
        self._parameters = {
            "prompt": {
                "type": "string",
                "description": "The input prompt used for all models",
            },
            "model_outputs": {
                "type": "array",
                "description": "List of model outputs to compare",
                "items": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "output": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["model_name", "output"],
                },
            },
            "evaluation_criteria": {
                "type": "array",
                "description": "Optional criteria for evaluation",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "weight": {"type": "number", "default": 1.0},
                    },
                    "required": ["name", "description"],
                },
            },
            "reference_output": {
                "type": "string",
                "description": "Optional reference output for comparison",
                "default": "",
            },
        }
        self._required_params = ["prompt", "model_outputs"]
        logger.info("Initialized tool: model_comparison")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self,
        prompt: str,
        model_outputs: List[Dict[str, Any]],
        evaluation_criteria: Optional[List[Dict[str, Any]]] = None,
        reference_output: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the model comparison tool

        Args:
            prompt: The input prompt used for all models
            model_outputs: List of model outputs to compare
            evaluation_criteria: Optional criteria for evaluation
            reference_output: Optional reference output for comparison

        Returns:
            Dictionary with comparison results
        """
        try:
            # Validate parameters
            if not prompt:
                return {"error": "Prompt cannot be empty"}

            if len(model_outputs) < 2:
                return {
                    "error": "At least two model outputs are required for comparison"
                }

            # Process each model output
            comparison_results = []

            for output_data in model_outputs:
                model_name = output_data.get("model_name", "Unnamed Model")
                output = output_data.get("output", "")
                metadata = output_data.get("metadata", {})

                # Basic metrics
                result = {
                    "model_name": model_name,
                    "output_length": len(output),
                    "word_count": len(output.split()),
                    "sentence_count": output.count(".")
                    + output.count("!")
                    + output.count("?"),
                    "metadata": metadata,
                }

                # Calculate similarity to reference if provided
                if reference_output:
                    similarity = self._calculate_similarity(output, reference_output)
                    result["reference_similarity"] = similarity

                # Evaluate against criteria if provided
                if evaluation_criteria:
                    criteria_scores = []

                    for criterion in evaluation_criteria:
                        name = criterion.get("name", "Unnamed")
                        description = criterion.get("description", "")
                        weight = criterion.get("weight", 1.0)

                        # Simple scoring based on keyword presence
                        score = self._score_criterion(output, description)

                        criteria_scores.append(
                            {
                                "criterion": name,
                                "score": score,
                                "weight": weight,
                                "weighted_score": score * weight,
                            }
                        )

                    # Calculate overall score
                    total_weighted_score = sum(
                        c["weighted_score"] for c in criteria_scores
                    )
                    total_weight = sum(c["weight"] for c in criteria_scores)
                    overall_score = (
                        total_weighted_score / total_weight if total_weight > 0 else 0
                    )

                    result["criteria_scores"] = criteria_scores
                    result["overall_score"] = round(overall_score, 2)

                comparison_results.append(result)

            # Rank models if criteria were provided
            if evaluation_criteria:
                ranked_results = sorted(
                    comparison_results,
                    key=lambda x: x.get("overall_score", 0),
                    reverse=True,
                )
                rankings = []

                for i, result in enumerate(ranked_results):
                    rankings.append(
                        {
                            "rank": i + 1,
                            "model_name": result["model_name"],
                            "overall_score": result["overall_score"],
                        }
                    )

                best_model = rankings[0]["model_name"] if rankings else None
            else:
                rankings = None
                best_model = None

            # Compare outputs for similarity to each other
            similarity_matrix = {}

            for i, output1 in enumerate(model_outputs):
                model1 = output1.get("model_name", f"Model {i+1}")
                similarities = {}

                for j, output2 in enumerate(model_outputs):
                    if i == j:
                        continue

                    model2 = output2.get("model_name", f"Model {j+1}")
                    similarity = self._calculate_similarity(
                        output1.get("output", ""), output2.get("output", "")
                    )
                    similarities[model2] = similarity

                similarity_matrix[model1] = similarities

            return {
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "model_count": len(model_outputs),
                "comparison_results": comparison_results,
                "rankings": rankings,
                "best_model": best_model,
                "similarity_matrix": similarity_matrix,
                "has_reference": bool(reference_output),
                "has_criteria": bool(evaluation_criteria),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Model comparison error: {e}")
            return {"error": f"Model comparison error: {str(e)}"}

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Simple Jaccard similarity
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return round(intersection / max(1, union), 3)

    def _score_criterion(self, text: str, criterion_description: str) -> float:
        """Score text against a criterion description

        Args:
            text: Text to evaluate
            criterion_description: Description of the criterion

        Returns:
            Score (0-1)
        """
        # Extract keywords from criterion description
        keywords = [
            word.lower() for word in criterion_description.split() if len(word) > 3
        ]
        text_lower = text.lower()

        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        score = matches / max(1, len(keywords))

        return round(score, 2)
