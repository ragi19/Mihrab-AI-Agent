"""
Real-time data collection tools for agents
"""

import json
import os
import time
import asyncio
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import aiohttp
from datetime import datetime

from ..base import BaseTool


class DataCollectorTool(BaseTool):
    """Tool for collecting real-time data from various sources"""

    def __init__(self):
        super().__init__(
            name="data_collector",
            description="Collect real-time data from APIs, streams, and other sources",
        )
        self._active_collectors = {}
        self._collection_counter = 0

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the data collection tool with the given parameters"""
        self.logger.debug(f"Executing tool {self.name} with parameters: {parameters}")

        try:
            # Validate parameters
            self._validate_parameters(parameters)

            # Execute tool-specific logic
            result = await self._execute(parameters)
            self.logger.debug(f"Tool execution result: {result}")

            return result
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the data collection tool with the given parameters"""
        action = parameters.get("action", "collect")
        
        if action == "collect":
            return await self._collect_data(
                source=parameters.get("source"),
                source_type=parameters.get("source_type", "api"),
                query_params=parameters.get("query_params", {}),
                headers=parameters.get("headers", {}),
                output_path=parameters.get("output_path"),
            )
        elif action == "start_stream":
            return await self._start_stream(
                source=parameters.get("source"),
                source_type=parameters.get("source_type", "api"),
                interval=parameters.get("interval", 60),
                query_params=parameters.get("query_params", {}),
                headers=parameters.get("headers", {}),
                output_path=parameters.get("output_path"),
                max_records=parameters.get("max_records"),
            )
        elif action == "stop_stream":
            return await self._stop_stream(
                stream_id=parameters.get("stream_id"),
            )
        elif action == "list_streams":
            return await self._list_streams()
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
    
    async def _collect_data(
        self,
        source: str,
        source_type: str = "api",
        query_params: Dict[str, Any] = {},
        headers: Dict[str, Any] = {},
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Collect data from a specified source"""
        if source_type == "api":
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(source, params=query_params, headers=headers) as response:
                        if response.status != 200:
                            return {
                                "success": False,
                                "error": f"API request failed with status {response.status}",
                                "status_code": response.status
                            }
                        
                        try:
                            data = await response.json()
                        except:
                            # Try to get text if not JSON
                            text = await response.text()
                            return {
                                "success": True,
                                "data_type": "text",
                                "data": text[:1000] + ("..." if len(text) > 1000 else ""),
                                "timestamp": datetime.now().isoformat()
                            }
                        
                        # Save to file if output path is provided
                        if output_path:
                            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                            with open(output_path, "w") as f:
                                json.dump(data, f, indent=2)
                        
                        # Return a summary if data is large
                        if isinstance(data, list):
                            summary = {
                                "success": True,
                                "data_type": "list",
                                "count": len(data),
                                "sample": data[:3] if len(data) > 3 else data,
                                "timestamp": datetime.now().isoformat()
                            }
                            if output_path:
                                summary["saved_to"] = output_path
                            return summary
                        elif isinstance(data, dict):
                            summary = {
                                "success": True,
                                "data_type": "object",
                                "keys": list(data.keys()),
                                "timestamp": datetime.now().isoformat()
                            }
                            if output_path:
                                summary["saved_to"] = output_path
                            return summary
                        else:
                            return {
                                "success": True,
                                "data": data,
                                "timestamp": datetime.now().isoformat()
                            }
            except Exception as e:
                return {"success": False, "error": str(e)}
        elif source_type == "file":
            try:
                path = Path(source)
                if not path.exists():
                    return {"success": False, "error": f"File not found: {source}"}
                
                with open(path, "r") as f:
                    if path.suffix.lower() == ".json":
                        data = json.load(f)
                    elif path.suffix.lower() == ".csv":
                        import csv
                        reader = csv.DictReader(f)
                        data = list(reader)
                    else:
                        # Read as text for other file types
                        data = f.read()
                
                # Save to output file if specified
                if output_path and output_path != source:
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    with open(output_path, "w") as f:
                        if isinstance(data, (dict, list)):
                            json.dump(data, f, indent=2)
                        else:
                            f.write(str(data))
                
                # Return a summary
                if isinstance(data, list):
                    return {
                        "success": True,
                        "data_type": "list",
                        "count": len(data),
                        "sample": data[:3] if len(data) > 3 else data,
                        "timestamp": datetime.now().isoformat()
                    }
                elif isinstance(data, dict):
                    return {
                        "success": True,
                        "data_type": "object",
                        "keys": list(data.keys()),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "success": True,
                        "data_type": "text",
                        "data": data[:1000] + ("..." if len(str(data)) > 1000 else ""),
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": f"Unsupported source type: {source_type}"}
    
    async def _start_stream(
        self,
        source: str,
        source_type: str = "api",
        interval: int = 60,
        query_params: Dict[str, Any] = {},
        headers: Dict[str, Any] = {},
        output_path: Optional[str] = None,
        max_records: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start a streaming data collection process"""
        stream_id = f"stream_{self._collection_counter}"
        self._collection_counter += 1
        
        # Create output directory if needed
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Store stream configuration
        self._active_collectors[stream_id] = {
            "source": source,
            "source_type": source_type,
            "interval": interval,
            "query_params": query_params,
            "headers": headers,
            "output_path": output_path,
            "max_records": max_records,
            "start_time": datetime.now().isoformat(),
            "records_collected": 0,
            "last_collection": None,
            "task": None,
            "status": "starting"
        }
        
        # Start collection task
        task = asyncio.create_task(
            self._stream_collection_task(
                stream_id, source, source_type, interval, query_params, headers, output_path, max_records
            )
        )
        self._active_collectors[stream_id]["task"] = task
        
        return {
            "success": True,
            "stream_id": stream_id,
            "message": f"Started data collection stream from {source}",
            "config": {
                "source": source,
                "interval": interval,
                "output_path": output_path,
                "max_records": max_records
            }
        }
    
    async def _stream_collection_task(
        self,
        stream_id: str,
        source: str,
        source_type: str,
        interval: int,
        query_params: Dict[str, Any],
        headers: Dict[str, Any],
        output_path: Optional[str],
        max_records: Optional[int],
    ) -> None:
        """Background task for streaming data collection"""
        self._active_collectors[stream_id]["status"] = "running"
        records_collected = 0
        
        try:
            while True:
                # Check if we've reached max records
                if max_records and records_collected >= max_records:
                    self._active_collectors[stream_id]["status"] = "completed"
                    self.logger.info(f"Stream {stream_id} completed after collecting {records_collected} records")
                    break
                
                # Check if stream has been stopped
                if self._active_collectors[stream_id]["status"] == "stopping":
                    self._active_collectors[stream_id]["status"] = "stopped"
                    self.logger.info(f"Stream {stream_id} stopped after collecting {records_collected} records")
                    break
                
                # Collect data
                timestamp = datetime.now().isoformat()
                result = await self._collect_data(
                    source=source,
                    source_type=source_type,
                    query_params=query_params,
                    headers=headers,
                    output_path=output_path.replace(
                        "{timestamp}", timestamp.replace(":", "-")
                    ) if output_path else None
                )
                
                # Update stream status
                self._active_collectors[stream_id]["last_collection"] = timestamp
                self._active_collectors[stream_id]["records_collected"] += 1
                records_collected += 1
                
                # Wait for next interval
                await asyncio.sleep(interval)
        except Exception as e:
            self._active_collectors[stream_id]["status"] = "error"
            self._active_collectors[stream_id]["error"] = str(e)
            self.logger.error(f"Error in stream {stream_id}: {e}")
    
    async def _stop_stream(self, stream_id: str) -> Dict[str, Any]:
        """Stop an active data collection stream"""
        if stream_id not in self._active_collectors:
            return {"success": False, "error": f"Stream not found: {stream_id}"}
        
        stream = self._active_collectors[stream_id]
        if stream["status"] in ["completed", "stopped", "error"]:
            return {"success": True, "message": f"Stream {stream_id} already in state: {stream['status']}"}
        
        # Mark stream for stopping
        stream["status"] = "stopping"
        
        return {
            "success": True,
            "message": f"Stopping stream {stream_id}",
            "records_collected": stream["records_collected"],
            "start_time": stream["start_time"],
            "last_collection": stream["last_collection"]
        }
    
    async def _list_streams(self) -> Dict[str, Any]:
        """List all active data collection streams"""
        streams = {}
        for stream_id, stream in self._active_collectors.items():
            streams[stream_id] = {
                "source": stream["source"],
                "status": stream["status"],
                "records_collected": stream["records_collected"],
                "start_time": stream["start_time"],
                "last_collection": stream["last_collection"]
            }
        
        return {
            "success": True,
            "active_streams": len([s for s in self._active_collectors.values() if s["status"] == "running"]),
            "total_streams": len(self._active_collectors),
            "streams": streams
        }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["collect", "start_stream", "stop_stream", "list_streams"],
                    "description": "Action to perform"
                },
                "source": {
                    "type": "string",
                    "description": "Data source URL or file path"
                },
                "source_type": {
                    "type": "string",
                    "enum": ["api", "file"],
                    "description": "Type of data source"
                },
                "query_params": {
                    "type": "object",
                    "description": "Query parameters for API requests"
                },
                "headers": {
                    "type": "object",
                    "description": "Headers for API requests"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save collected data"
                },
                "interval": {
                    "type": "integer",
                    "description": "Interval in seconds between data collections for streaming"
                },
                "max_records": {
                    "type": "integer",
                    "description": "Maximum number of records to collect in streaming mode"
                },
                "stream_id": {
                    "type": "string",
                    "description": "ID of the stream to manage"
                }
            },
            "required": ["action"],
            "allOf": [
                {
                    "if": {
                        "properties": {"action": {"enum": ["collect", "start_stream"]}}
                    },
                    "then": {
                        "required": ["source"]
                    }
                },
                {
                    "if": {
                        "properties": {"action": {"enum": ["stop_stream"]}}
                    },
                    "then": {
                        "required": ["stream_id"]
                    }
                }
            ]
        } 