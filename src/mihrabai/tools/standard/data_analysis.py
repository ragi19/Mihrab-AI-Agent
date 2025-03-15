"""
Data analysis tools for agents
"""

import json
import os
from typing import Any, Dict, List, Optional, Union
import asyncio
from pathlib import Path

from ..base import BaseTool


class DataAnalysisTool(BaseTool):
    """Tool for performing basic data analysis on structured data"""

    def __init__(self):
        super().__init__(
            name="data_analysis",
            description="Analyze structured data and provide statistical insights",
        )

    async def _execute(self, parameters: Dict[str, Any]) -> str:
        """Execute the data analysis tool with the given parameters"""
        action = parameters.get("action", "summarize")
        
        if action == "summarize":
            return await self._summarize_data(
                data=parameters.get("data"),
                data_path=parameters.get("data_path"),
            )
        elif action == "filter":
            return await self._filter_data(
                data=parameters.get("data"),
                data_path=parameters.get("data_path"),
                filter_condition=parameters.get("filter_condition", {}),
            )
        elif action == "aggregate":
            return await self._aggregate_data(
                data=parameters.get("data"),
                data_path=parameters.get("data_path"),
                group_by=parameters.get("group_by", []),
                metrics=parameters.get("metrics", []),
            )
        else:
            return f"Unknown action: {action}"
    
    async def _summarize_data(
        self, data: Optional[List[Dict[str, Any]]] = None, data_path: Optional[str] = None
    ) -> str:
        """Summarize a dataset with basic statistics"""
        # Load data from path if provided
        if data is None and data_path:
            data = await self._load_data(data_path)
        
        if not data:
            return "No data provided for analysis"
        
        # Basic data summary
        num_records = len(data)
        
        # Get all unique keys across all records
        all_keys = set()
        for record in data:
            all_keys.update(record.keys())
        
        # Count data types and sample values for each field
        field_info = {}
        for key in all_keys:
            values = [record.get(key) for record in data if key in record]
            non_null_values = [v for v in values if v is not None]
            
            if not non_null_values:
                field_info[key] = {
                    "type": "unknown",
                    "null_count": len(values),
                    "sample": None
                }
                continue
            
            # Determine predominant type
            types = [type(v).__name__ for v in non_null_values]
            predominant_type = max(set(types), key=types.count)
            
            # Basic stats based on type
            if predominant_type in ["int", "float"]:
                numeric_values = [v for v in non_null_values if isinstance(v, (int, float))]
                field_info[key] = {
                    "type": predominant_type,
                    "null_count": len(values) - len(non_null_values),
                    "min": min(numeric_values) if numeric_values else None,
                    "max": max(numeric_values) if numeric_values else None,
                    "avg": sum(numeric_values) / len(numeric_values) if numeric_values else None,
                    "sample": non_null_values[:3]
                }
            elif predominant_type == "str":
                string_values = [v for v in non_null_values if isinstance(v, str)]
                field_info[key] = {
                    "type": "string",
                    "null_count": len(values) - len(non_null_values),
                    "min_length": min(len(s) for s in string_values) if string_values else 0,
                    "max_length": max(len(s) for s in string_values) if string_values else 0,
                    "unique_count": len(set(string_values)),
                    "sample": non_null_values[:3]
                }
            else:
                field_info[key] = {
                    "type": predominant_type,
                    "null_count": len(values) - len(non_null_values),
                    "sample": str(non_null_values[:2])
                }
        
        # Format the summary
        summary = f"Dataset Summary:\n"
        summary += f"Total records: {num_records}\n"
        summary += f"Fields: {len(all_keys)}\n\n"
        
        summary += "Field Analysis:\n"
        for key, info in field_info.items():
            summary += f"- {key} ({info['type']}):\n"
            
            for stat_key, stat_value in info.items():
                if stat_key not in ["type", "sample"]:
                    summary += f"  {stat_key}: {stat_value}\n"
            
            if "sample" in info and info["sample"]:
                sample_str = str(info["sample"])
                if len(sample_str) > 50:
                    sample_str = sample_str[:47] + "..."
                summary += f"  sample: {sample_str}\n"
            
            summary += "\n"
        
        return summary
    
    async def _filter_data(
        self, 
        data: Optional[List[Dict[str, Any]]] = None, 
        data_path: Optional[str] = None,
        filter_condition: Dict[str, Any] = {}
    ) -> str:
        """Filter data based on specified conditions"""
        # Load data from path if provided
        if data is None and data_path:
            data = await self._load_data(data_path)
        
        if not data:
            return "No data provided for filtering"
        
        if not filter_condition:
            return "No filter conditions specified"
        
        # Apply filters
        filtered_data = []
        for record in data:
            matches = True
            for field, value in filter_condition.items():
                if field not in record or record[field] != value:
                    matches = False
                    break
            
            if matches:
                filtered_data.append(record)
        
        # Return summary of filtered data
        return f"Filtered {len(filtered_data)} records from {len(data)} total records.\n\nFirst few records:\n{json.dumps(filtered_data[:3], indent=2)}"
    
    async def _aggregate_data(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        data_path: Optional[str] = None,
        group_by: List[str] = [],
        metrics: List[Dict[str, str]] = []
    ) -> str:
        """Aggregate data by specified fields and calculate metrics"""
        # Load data from path if provided
        if data is None and data_path:
            data = await self._load_data(data_path)
        
        if not data:
            return "No data provided for aggregation"
        
        if not group_by:
            return "No grouping fields specified"
        
        if not metrics:
            return "No metrics specified for aggregation"
        
        # Group data
        groups = {}
        for record in data:
            # Create group key based on the group_by fields
            group_values = []
            for field in group_by:
                group_values.append(str(record.get(field, "null")))
            
            group_key = "|".join(group_values)
            
            if group_key not in groups:
                groups[group_key] = {
                    "records": [],
                    "group_values": {field: record.get(field) for field in group_by}
                }
            
            groups[group_key]["records"].append(record)
        
        # Calculate metrics for each group
        results = []
        for group_key, group_data in groups.items():
            result = {**group_data["group_values"]}
            
            for metric in metrics:
                field = metric.get("field")
                operation = metric.get("operation", "count")
                
                if not field and operation != "count":
                    continue
                
                if operation == "count":
                    result[f"count"] = len(group_data["records"])
                else:
                    values = [r.get(field) for r in group_data["records"] if field in r]
                    numeric_values = [v for v in values if isinstance(v, (int, float))]
                    
                    if not numeric_values:
                        result[f"{operation}_{field}"] = None
                        continue
                    
                    if operation == "sum":
                        result[f"sum_{field}"] = sum(numeric_values)
                    elif operation == "avg":
                        result[f"avg_{field}"] = sum(numeric_values) / len(numeric_values)
                    elif operation == "min":
                        result[f"min_{field}"] = min(numeric_values)
                    elif operation == "max":
                        result[f"max_{field}"] = max(numeric_values)
            
            results.append(result)
        
        # Format the results
        formatted_results = json.dumps(results, indent=2)
        summary = f"Aggregated data into {len(results)} groups based on {', '.join(group_by)}.\n\n"
        summary += formatted_results
        
        return summary
    
    async def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from a file path"""
        path = Path(data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(path, "r") as f:
            if path.suffix.lower() == ".json":
                return json.load(f)
            elif path.suffix.lower() == ".csv":
                import csv
                reader = csv.DictReader(f)
                return list(reader)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["summarize", "filter", "aggregate"],
                    "description": "Analysis action to perform",
                },
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Data to analyze (array of objects)",
                },
                "data_path": {
                    "type": "string",
                    "description": "Path to a JSON or CSV file containing data to analyze",
                },
                "filter_condition": {
                    "type": "object",
                    "description": "Filter conditions as field-value pairs (used with filter action)",
                },
                "group_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to group by (used with aggregate action)",
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "operation": {
                                "type": "string",
                                "enum": ["count", "sum", "avg", "min", "max"]
                            }
                        }
                    },
                    "description": "Metrics to calculate (used with aggregate action)",
                }
            },
            "required": ["action"],
        }
