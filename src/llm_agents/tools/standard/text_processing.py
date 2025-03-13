"""
Text processing tools for common NLP operations
"""

import json
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional

from ...utils.logging import get_logger
from ..base import BaseTool

logger = get_logger("tools.text_processing")


class TextSummarizerTool(BaseTool):
    """Tool for summarizing text content"""

    def __init__(self):
        super().__init__(
            name="text_summarizer",
            description="Summarize text content by extracting key sentences",
        )
        self._parameters = {
            "text": {"type": "string", "description": "The text content to summarize"},
            "max_sentences": {
                "type": "integer",
                "description": "Maximum number of sentences to include in summary",
                "default": 3,
            },
            "min_length": {
                "type": "integer",
                "description": "Minimum length of sentences to consider",
                "default": 10,
            },
        }
        self._required_params = ["text"]
        logger.info("Initialized tool: text_summarizer")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self, text: str, max_sentences: int = 3, min_length: int = 10, **kwargs
    ) -> Dict[str, Any]:
        """Execute the text summarizer tool

        Args:
            text: The text content to summarize
            max_sentences: Maximum number of sentences to include in summary
            min_length: Minimum length of sentences to consider

        Returns:
            Dictionary with summary result
        """
        try:
            # Split text into sentences
            sentences = re.split(r"(?<=[.!?])\s+", text)

            # Filter out short sentences
            valid_sentences = [s for s in sentences if len(s) >= min_length]

            if not valid_sentences:
                return {
                    "summary": text,
                    "original_length": len(text),
                    "summary_length": len(text),
                    "reduction_percentage": 0,
                }

            # Extract keywords
            words = re.findall(r"\b\w+\b", text.lower())
            stop_words = set(
                [
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "is",
                    "are",
                    "was",
                    "were",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "with",
                    "by",
                    "about",
                    "as",
                    "of",
                ]
            )
            keywords = [word for word in words if word not in stop_words]

            # Count keyword frequency
            keyword_freq = Counter(keywords)

            # Score sentences based on keyword frequency
            sentence_scores = []
            for sentence in valid_sentences:
                score = 0
                for word in re.findall(r"\b\w+\b", sentence.lower()):
                    if word in keyword_freq:
                        score += keyword_freq[word]
                sentence_scores.append((sentence, score))

            # Sort sentences by score
            ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

            # Select top sentences (up to max_sentences)
            top_sentences = [s[0] for s in ranked_sentences[:max_sentences]]

            # Reorder sentences to maintain original flow
            summary_sentences = []
            for sentence in valid_sentences:
                if sentence in top_sentences and sentence not in summary_sentences:
                    summary_sentences.append(sentence)
                if len(summary_sentences) >= max_sentences:
                    break

            # Join sentences to form summary
            summary = " ".join(summary_sentences)

            # Calculate reduction percentage
            original_length = len(text)
            summary_length = len(summary)
            reduction_percentage = round(
                (1 - (summary_length / original_length)) * 100, 2
            )

            return {
                "summary": summary,
                "original_length": original_length,
                "summary_length": summary_length,
                "reduction_percentage": reduction_percentage,
            }
        except Exception as e:
            logger.error(f"Text summarizer error: {e}")
            return {
                "error": f"Summarization error: {str(e)}",
                "text": text[:100] + "..." if len(text) > 100 else text,
            }


class TextAnalyzerTool(BaseTool):
    """Tool for analyzing text content"""

    def __init__(self):
        super().__init__(
            name="text_analyzer",
            description="Analyze text content for readability, sentiment, and statistics",
        )
        self._parameters = {
            "text": {"type": "string", "description": "The text content to analyze"},
            "include_readability": {
                "type": "boolean",
                "description": "Include readability metrics",
                "default": True,
            },
            "include_statistics": {
                "type": "boolean",
                "description": "Include text statistics",
                "default": True,
            },
        }
        self._required_params = ["text"]
        logger.info("Initialized tool: text_analyzer")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self,
        text: str,
        include_readability: bool = True,
        include_statistics: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the text analyzer tool

        Args:
            text: The text content to analyze
            include_readability: Include readability metrics
            include_statistics: Include text statistics

        Returns:
            Dictionary with analysis results
        """
        try:
            result = {"text_sample": text[:100] + "..." if len(text) > 100 else text}

            # Basic text statistics
            if include_statistics:
                # Count characters, words, sentences
                char_count = len(text)
                word_count = len(re.findall(r"\b\w+\b", text))
                sentence_count = len(re.split(r"(?<=[.!?])\s+", text))

                # Calculate average word length
                words = re.findall(r"\b\w+\b", text)
                avg_word_length = sum(len(word) for word in words) / max(1, len(words))

                # Calculate average sentence length
                sentences = re.split(r"(?<=[.!?])\s+", text)
                avg_sentence_length = sum(
                    len(re.findall(r"\b\w+\b", sentence)) for sentence in sentences
                ) / max(1, len(sentences))

                result["statistics"] = {
                    "character_count": char_count,
                    "word_count": word_count,
                    "sentence_count": sentence_count,
                    "avg_word_length": round(avg_word_length, 2),
                    "avg_sentence_length": round(avg_sentence_length, 2),
                }

            # Readability metrics
            if include_readability:
                # Flesch Reading Ease score
                words = re.findall(r"\b\w+\b", text)
                sentences = re.split(r"(?<=[.!?])\s+", text)

                # Count syllables (approximation)
                def count_syllables(word):
                    word = word.lower()
                    if len(word) <= 3:
                        return 1

                    # Remove common endings
                    if word.endswith("es") or word.endswith("ed"):
                        word = word[:-2]
                    elif word.endswith("e"):
                        word = word[:-1]

                    # Count vowel groups
                    vowels = "aeiouy"
                    count = 0
                    prev_is_vowel = False

                    for char in word:
                        is_vowel = char in vowels
                        if is_vowel and not prev_is_vowel:
                            count += 1
                        prev_is_vowel = is_vowel

                    return max(1, count)

                syllable_count = sum(count_syllables(word) for word in words)

                # Calculate Flesch Reading Ease score
                if word_count > 0 and sentence_count > 0:
                    flesch_score = (
                        206.835
                        - 1.015 * (word_count / sentence_count)
                        - 84.6 * (syllable_count / word_count)
                    )
                    flesch_score = max(
                        0, min(100, flesch_score)
                    )  # Clamp between 0 and 100
                else:
                    flesch_score = 0

                # Interpret Flesch score
                if flesch_score >= 90:
                    readability_level = "Very Easy - 5th grade"
                elif flesch_score >= 80:
                    readability_level = "Easy - 6th grade"
                elif flesch_score >= 70:
                    readability_level = "Fairly Easy - 7th grade"
                elif flesch_score >= 60:
                    readability_level = "Standard - 8th-9th grade"
                elif flesch_score >= 50:
                    readability_level = "Fairly Difficult - 10th-12th grade"
                elif flesch_score >= 30:
                    readability_level = "Difficult - College level"
                else:
                    readability_level = "Very Difficult - College graduate level"

                result["readability"] = {
                    "flesch_reading_ease": round(flesch_score, 2),
                    "readability_level": readability_level,
                    "syllable_count": syllable_count,
                    "syllables_per_word": round(syllable_count / max(1, word_count), 2),
                }

            return result
        except Exception as e:
            logger.error(f"Text analyzer error: {e}")
            return {
                "error": f"Analysis error: {str(e)}",
                "text": text[:100] + "..." if len(text) > 100 else text,
            }


class TextFormatterTool(BaseTool):
    """Tool for formatting and transforming text"""

    def __init__(self):
        super().__init__(
            name="text_formatter", description="Format and transform text content"
        )
        self._parameters = {
            "text": {"type": "string", "description": "The text content to format"},
            "operation": {
                "type": "string",
                "description": "The formatting operation to perform",
                "enum": [
                    "uppercase",
                    "lowercase",
                    "titlecase",
                    "sentencecase",
                    "strip",
                    "replace",
                    "truncate",
                    "wordwrap",
                ],
                "default": "strip",
            },
            "find": {
                "type": "string",
                "description": "Text to find (for replace operation)",
            },
            "replace": {
                "type": "string",
                "description": "Text to replace with (for replace operation)",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum length (for truncate operation)",
                "default": 100,
            },
            "width": {
                "type": "integer",
                "description": "Line width (for wordwrap operation)",
                "default": 80,
            },
        }
        self._required_params = ["text", "operation"]
        logger.info("Initialized tool: text_formatter")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self,
        text: str,
        operation: str,
        find: Optional[str] = None,
        replace: Optional[str] = None,
        max_length: int = 100,
        width: int = 80,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the text formatter tool

        Args:
            text: The text content to format
            operation: The formatting operation to perform
            find: Text to find (for replace operation)
            replace: Text to replace with (for replace operation)
            max_length: Maximum length (for truncate operation)
            width: Line width (for wordwrap operation)

        Returns:
            Dictionary with formatted text
        """
        try:
            original_text = text

            if operation == "uppercase":
                result = text.upper()
            elif operation == "lowercase":
                result = text.lower()
            elif operation == "titlecase":
                result = string.capwords(text)
            elif operation == "sentencecase":
                # Split by sentence endings and capitalize first letter of each
                sentences = re.split(r"(?<=[.!?])\s+", text)
                result = " ".join(s[0].upper() + s[1:] if s else "" for s in sentences)
            elif operation == "strip":
                result = text.strip()
            elif operation == "replace":
                if find is None:
                    return {"error": "Missing 'find' parameter for replace operation"}
                replace = replace or ""
                result = text.replace(find, replace)
            elif operation == "truncate":
                if len(text) > max_length:
                    result = text[:max_length] + "..."
                else:
                    result = text
            elif operation == "wordwrap":
                # Simple word wrap implementation
                words = text.split()
                lines = []
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + (1 if current_line else 0) <= width:
                        current_line.append(word)
                        current_length += len(word) + (1 if current_length > 0 else 0)
                    else:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                        current_length = len(word)

                if current_line:
                    lines.append(" ".join(current_line))

                result = "\n".join(lines)
            else:
                return {"error": f"Unknown operation: {operation}"}

            return {
                "original": (
                    original_text[:100] + "..."
                    if len(original_text) > 100
                    else original_text
                ),
                "formatted": result[:100] + "..." if len(result) > 100 else result,
                "operation": operation,
                "original_length": len(original_text),
                "formatted_length": len(result),
            }
        except Exception as e:
            logger.error(f"Text formatter error: {e}")
            return {
                "error": f"Formatting error: {str(e)}",
                "text": text[:100] + "..." if len(text) > 100 else text,
                "operation": operation,
            }
