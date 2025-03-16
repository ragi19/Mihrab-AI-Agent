"""
Data Agent for collecting and preprocessing real-time data
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
from pathlib import Path

from ..core.memory_task_agent import MemoryEnabledTaskAgent
from ..core.message import Message, MessageRole
from ..models.base import BaseModel
from ..tools.standard.data_collector import DataCollectorTool
from ..tools.standard.data_preprocessor import DataPreprocessorTool
from ..tools.standard.data import CSVParserTool, JSONParserTool
from ..tools.standard.filesystem import FileReaderTool, FileWriterTool
from ..utils.logging import get_logger
from ..utils.tracing import TraceProvider


class DataAgent(MemoryEnabledTaskAgent):
    """
    Agent specialized for collecting and preprocessing real-time data
    
    This agent combines data collection capabilities with preprocessing
    to provide a complete pipeline for real-time data processing.
    """
    
    def __init__(
        self,
        model: BaseModel,
        system_message: Optional[str] = None,
        max_steps: Optional[int] = 15,
        max_history_tokens: Optional[int] = 4000,
        memory_size: int = 100,
        data_dir: Optional[str] = None,
        trace_provider: Optional[TraceProvider] = None,
    ):
        # Set default system message if not provided
        if system_message is None:
            system_message = """You are a specialized Data Agent focused on collecting and preprocessing real-time data.
Your primary responsibilities are:
1. Collecting data from various sources (APIs, files, etc.)
2. Preprocessing data to clean, transform, and prepare it for analysis
3. Monitoring data streams for changes and anomalies
4. Storing processed data in appropriate formats

You have access to specialized tools for data collection and preprocessing.
Always provide clear summaries of the data you collect and the preprocessing steps you apply.
"""
        
        # Initialize tools
        tools = [
            DataCollectorTool(),
            DataPreprocessorTool(),
            JSONParserTool(),
            CSVParserTool(),
            FileReaderTool(),
            FileWriterTool(),
        ]
        
        # Initialize parent class
        super().__init__(
            model=model,
            system_message=system_message,
            max_steps=max_steps,
            max_history_tokens=max_history_tokens,
            memory_size=memory_size,
            tools=tools,
            trace_provider=trace_provider,
        )
        
        # Set up data directory
        self.data_dir = data_dir or os.path.join(os.getcwd(), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Set up subdirectories
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = get_logger("agent.data")
        
        # Track active data streams
        self.active_streams = {}
        
        # Initialize data collection state
        self.collection_state = {
            "last_collection": None,
            "collections_count": 0,
            "active_streams": 0,
            "data_sources": set(),
        }
    
    async def collect_data(
        self,
        source: str,
        source_type: str = "api",
        query_params: Dict[str, Any] = {},
        headers: Dict[str, Any] = {},
        save_raw: bool = True,
    ) -> Dict[str, Any]:
        """
        Collect data from a specified source
        
        Args:
            source: URL or file path to collect data from
            source_type: Type of source ("api" or "file")
            query_params: Query parameters for API requests
            headers: Headers for API requests
            save_raw: Whether to save the raw data
            
        Returns:
            Dictionary with collection results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = Path(source).stem if source_type == "file" else source.split("/")[-1]
        
        # Generate output path for raw data
        output_path = None
        if save_raw:
            output_path = os.path.join(
                self.raw_data_dir,
                f"{source_name}_{timestamp}.json"
            )
        
        # Use data collector tool
        collector_tool = next(t for t in self.tools if t.name == "data_collector")
        result = await collector_tool.execute({
            "action": "collect",
            "source": source,
            "source_type": source_type,
            "query_params": query_params,
            "headers": headers,
            "output_path": output_path
        })
        
        # Update collection state
        self.collection_state["last_collection"] = timestamp
        self.collection_state["collections_count"] += 1
        self.collection_state["data_sources"].add(source)
        
        # Log collection
        self.logger.info(f"Collected data from {source} ({source_type})")
        
        return result
    
    async def start_data_stream(
        self,
        source: str,
        source_type: str = "api",
        interval: int = 60,
        query_params: Dict[str, Any] = {},
        headers: Dict[str, Any] = {},
        max_records: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Start a streaming data collection process
        
        Args:
            source: URL or file path to collect data from
            source_type: Type of source ("api" or "file")
            interval: Interval in seconds between collections
            query_params: Query parameters for API requests
            headers: Headers for API requests
            max_records: Maximum number of records to collect
            
        Returns:
            Dictionary with stream information
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = Path(source).stem if source_type == "file" else source.split("/")[-1]
        
        # Generate output path template for raw data
        output_path = os.path.join(
            self.raw_data_dir,
            f"{source_name}_{timestamp}_{'{timestamp}'}.json"
        )
        
        # Use data collector tool
        collector_tool = next(t for t in self.tools if t.name == "data_collector")
        result = await collector_tool.execute({
            "action": "start_stream",
            "source": source,
            "source_type": source_type,
            "interval": interval,
            "query_params": query_params,
            "headers": headers,
            "output_path": output_path,
            "max_records": max_records
        })
        
        # Update collection state
        if result.get("success", False):
            stream_id = result.get("stream_id")
            self.active_streams[stream_id] = {
                "source": source,
                "source_type": source_type,
                "interval": interval,
                "start_time": timestamp,
                "status": "running"
            }
            self.collection_state["active_streams"] = len(self.active_streams)
            self.collection_state["data_sources"].add(source)
        
        # Log stream start
        self.logger.info(f"Started data stream from {source} ({source_type}) with interval {interval}s")
        
        return result
    
    async def stop_data_stream(self, stream_id: str) -> Dict[str, Any]:
        """
        Stop an active data collection stream
        
        Args:
            stream_id: ID of the stream to stop
            
        Returns:
            Dictionary with stop result
        """
        # Use data collector tool
        collector_tool = next(t for t in self.tools if t.name == "data_collector")
        result = await collector_tool.execute({
            "action": "stop_stream",
            "stream_id": stream_id
        })
        
        # Update collection state
        if result.get("success", False) and stream_id in self.active_streams:
            self.active_streams[stream_id]["status"] = "stopped"
            self.collection_state["active_streams"] = len([
                s for s in self.active_streams.values() if s["status"] == "running"
            ])
        
        # Log stream stop
        self.logger.info(f"Stopped data stream {stream_id}")
        
        return result
    
    async def list_data_streams(self) -> Dict[str, Any]:
        """
        List all active data collection streams
        
        Returns:
            Dictionary with stream information
        """
        # Use data collector tool
        collector_tool = next(t for t in self.tools if t.name == "data_collector")
        result = await collector_tool.execute({
            "action": "list_streams"
        })
        
        # Update local state from tool result
        if result.get("success", False):
            streams = result.get("streams", {})
            for stream_id, stream_info in streams.items():
                if stream_id in self.active_streams:
                    self.active_streams[stream_id]["status"] = stream_info["status"]
            
            self.collection_state["active_streams"] = result.get("active_streams", 0)
        
        return result
    
    async def preprocess_data(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        action: str = "clean",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Preprocess data with specified operations
        
        Args:
            data_path: Path to the data file to preprocess
            output_path: Path to save the preprocessed data (if None, will generate one)
            action: Preprocessing action to perform
            **kwargs: Additional parameters for the specific action
            
        Returns:
            Dictionary with preprocessing results
        """
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = Path(data_path).stem
            file_ext = Path(data_path).suffix
            output_path = os.path.join(
                self.processed_data_dir,
                f"{file_name}_processed_{timestamp}{file_ext}"
            )
        
        # Prepare parameters
        params = {
            "action": action,
            "data_path": data_path,
            "output_path": output_path,
            **kwargs
        }
        
        # Use data preprocessor tool
        preprocessor_tool = next(t for t in self.tools if t.name == "data_preprocessor")
        result = await preprocessor_tool.execute(params)
        
        # Log preprocessing
        self.logger.info(f"Preprocessed data from {data_path} with action {action}")
        
        return result
    
    async def clean_data(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        operations: List[Dict[str, Any]] = []
    ) -> Dict[str, Any]:
        """
        Clean data by handling missing values, outliers, and duplicates
        
        Args:
            data_path: Path to the data file to clean
            output_path: Path to save the cleaned data
            operations: List of cleaning operations to apply
            
        Returns:
            Dictionary with cleaning results
        """
        return await self.preprocess_data(
            data_path=data_path,
            output_path=output_path,
            action="clean",
            operations=operations
        )
    
    async def transform_data(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        transformations: List[Dict[str, Any]] = []
    ) -> Dict[str, Any]:
        """
        Transform data with various operations
        
        Args:
            data_path: Path to the data file to transform
            output_path: Path to save the transformed data
            transformations: List of transformations to apply
            
        Returns:
            Dictionary with transformation results
        """
        return await self.preprocess_data(
            data_path=data_path,
            output_path=output_path,
            action="transform",
            transformations=transformations
        )
    
    async def normalize_data(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        method: str = "minmax",
        columns: List[str] = []
    ) -> Dict[str, Any]:
        """
        Normalize numerical data using various methods
        
        Args:
            data_path: Path to the data file to normalize
            output_path: Path to save the normalized data
            method: Normalization method to use
            columns: Columns to normalize
            
        Returns:
            Dictionary with normalization results
        """
        return await self.preprocess_data(
            data_path=data_path,
            output_path=output_path,
            action="normalize",
            method=method,
            columns=columns
        )
    
    async def split_data(
        self,
        data_path: str,
        train_output: Optional[str] = None,
        test_output: Optional[str] = None,
        validation_output: Optional[str] = None,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Split data into training, testing, and optional validation sets
        
        Args:
            data_path: Path to the data file to split
            train_output: Path to save the training data
            test_output: Path to save the testing data
            validation_output: Path to save the validation data
            test_size: Proportion of data to use for testing
            validation_size: Proportion of data to use for validation
            random_seed: Random seed for reproducible splitting
            
        Returns:
            Dictionary with split results
        """
        # Generate output paths if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = Path(data_path).stem
        file_ext = Path(data_path).suffix
        
        if train_output is None:
            train_output = os.path.join(
                self.processed_data_dir,
                f"{file_name}_train_{timestamp}{file_ext}"
            )
        
        if test_output is None:
            test_output = os.path.join(
                self.processed_data_dir,
                f"{file_name}_test_{timestamp}{file_ext}"
            )
        
        if validation_output is None and validation_size > 0:
            validation_output = os.path.join(
                self.processed_data_dir,
                f"{file_name}_validation_{timestamp}{file_ext}"
            )
        
        return await self.preprocess_data(
            data_path=data_path,
            action="split",
            train_output=train_output,
            test_output=test_output,
            validation_output=validation_output,
            test_size=test_size,
            validation_size=validation_size,
            random_seed=random_seed
        )
    
    async def process_message(self, message: Message) -> Message:
        """Process a message with the data agent"""
        # Add data agent context to the message
        enriched_message = self._enrich_message_with_data_context(message)
        
        # Process with parent class implementation
        return await super().process_message(enriched_message)
    
    def _enrich_message_with_data_context(self, message: Message) -> Message:
        """Add data agent context to the message"""
        # Create data context
        data_context = f"\nData Agent State:"
        data_context += f"\n- Collections: {self.collection_state['collections_count']}"
        data_context += f"\n- Active Streams: {self.collection_state['active_streams']}"
        data_context += f"\n- Data Sources: {len(self.collection_state['data_sources'])}"
        data_context += f"\n- Raw Data Directory: {self.raw_data_dir}"
        data_context += f"\n- Processed Data Directory: {self.processed_data_dir}"
        
        # Create metadata dict with data state
        metadata = {"data_state": self.collection_state.copy()}
        # Add original metadata if it exists
        if message.metadata is not None:
            metadata.update(message.metadata)
        
        return Message(
            role=message.role,
            content=f"{message.content}\n{data_context}",
            metadata=metadata,
        ) 