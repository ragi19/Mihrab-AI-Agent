"""
Data preprocessing tools for agents
"""

import json
import os
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import pandas as pd

from ..base import BaseTool


class DataPreprocessorTool(BaseTool):
    """Tool for preprocessing data for analysis and machine learning"""

    def __init__(self):
        super().__init__(
            name="data_preprocessor",
            description="Preprocess data for analysis and machine learning",
        )

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the data preprocessing tool with the given parameters"""
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
        """Execute the data preprocessing tool with the given parameters"""
        action = parameters.get("action", "clean")
        
        if action == "clean":
            return await self._clean_data(
                data=parameters.get("data"),
                data_path=parameters.get("data_path"),
                output_path=parameters.get("output_path"),
                operations=parameters.get("operations", []),
            )
        elif action == "transform":
            return await self._transform_data(
                data=parameters.get("data"),
                data_path=parameters.get("data_path"),
                output_path=parameters.get("output_path"),
                transformations=parameters.get("transformations", []),
            )
        elif action == "normalize":
            return await self._normalize_data(
                data=parameters.get("data"),
                data_path=parameters.get("data_path"),
                output_path=parameters.get("output_path"),
                method=parameters.get("method", "minmax"),
                columns=parameters.get("columns", []),
            )
        elif action == "split":
            return await self._split_data(
                data=parameters.get("data"),
                data_path=parameters.get("data_path"),
                train_output=parameters.get("train_output"),
                test_output=parameters.get("test_output"),
                validation_output=parameters.get("validation_output"),
                test_size=parameters.get("test_size", 0.2),
                validation_size=parameters.get("validation_size", 0.0),
                random_seed=parameters.get("random_seed", 42),
            )
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
    
    async def _load_data(self, data: Optional[Any] = None, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load data from parameters or file path into a pandas DataFrame"""
        if data is not None:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        
        if data_path:
            path = Path(data_path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            elif path.suffix.lower() == ".json":
                return pd.read_json(path)
            elif path.suffix.lower() == ".xlsx" or path.suffix.lower() == ".xls":
                return pd.read_excel(path)
            elif path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        raise ValueError("Either data or data_path must be provided")
    
    async def _save_data(self, df: pd.DataFrame, output_path: Optional[str] = None) -> Optional[str]:
        """Save DataFrame to file if output path is provided"""
        if not output_path:
            return None
        
        path = Path(output_path)
        os.makedirs(path.parent, exist_ok=True)
        
        if path.suffix.lower() == ".csv":
            df.to_csv(path, index=False)
        elif path.suffix.lower() == ".json":
            df.to_json(path, orient="records", indent=2)
        elif path.suffix.lower() == ".xlsx":
            df.to_excel(path, index=False)
        elif path.suffix.lower() == ".parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {path.suffix}")
        
        return str(path)
    
    async def _clean_data(
        self,
        data: Optional[Any] = None,
        data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        operations: List[Dict[str, Any]] = [],
    ) -> Dict[str, Any]:
        """Clean data by handling missing values, outliers, and duplicates"""
        try:
            # Load data
            df = await self._load_data(data, data_path)
            original_shape = df.shape
            
            # Apply cleaning operations
            for operation in operations:
                op_type = operation.get("type")
                
                if op_type == "drop_na":
                    columns = operation.get("columns", None)
                    if columns:
                        df = df.dropna(subset=columns)
                    else:
                        df = df.dropna()
                
                elif op_type == "fill_na":
                    columns = operation.get("columns", df.columns.tolist())
                    value = operation.get("value")
                    method = operation.get("method")
                    
                    if method == "mean":
                        for col in columns:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                df[col] = df[col].fillna(df[col].mean())
                    elif method == "median":
                        for col in columns:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                df[col] = df[col].fillna(df[col].median())
                    elif method == "mode":
                        for col in columns:
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
                    elif method == "ffill":
                        df[columns] = df[columns].fillna(method="ffill")
                    elif method == "bfill":
                        df[columns] = df[columns].fillna(method="bfill")
                    elif value is not None:
                        df[columns] = df[columns].fillna(value)
                
                elif op_type == "drop_duplicates":
                    columns = operation.get("columns", None)
                    keep = operation.get("keep", "first")
                    df = df.drop_duplicates(subset=columns, keep=keep)
                
                elif op_type == "remove_outliers":
                    columns = operation.get("columns", [])
                    method = operation.get("method", "zscore")
                    threshold = operation.get("threshold", 3.0)
                    
                    for col in columns:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            continue
                            
                        if method == "zscore":
                            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                            df = df[~(z_scores > threshold)]
                        elif method == "iqr":
                            q1 = df[col].quantile(0.25)
                            q3 = df[col].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - threshold * iqr
                            upper_bound = q3 + threshold * iqr
                            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            # Save processed data
            saved_path = await self._save_data(df, output_path)
            
            # Return summary
            result = {
                "success": True,
                "original_rows": original_shape[0],
                "original_columns": original_shape[1],
                "processed_rows": df.shape[0],
                "processed_columns": df.shape[1],
                "rows_removed": original_shape[0] - df.shape[0],
                "operations_applied": len(operations),
                "timestamp": datetime.now().isoformat()
            }
            
            if saved_path:
                result["saved_to"] = saved_path
            
            return result
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _transform_data(
        self,
        data: Optional[Any] = None,
        data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        transformations: List[Dict[str, Any]] = [],
    ) -> Dict[str, Any]:
        """Transform data with various operations"""
        try:
            # Load data
            df = await self._load_data(data, data_path)
            original_columns = set(df.columns)
            
            # Apply transformations
            for transform in transformations:
                transform_type = transform.get("type")
                
                if transform_type == "rename_columns":
                    mapping = transform.get("mapping", {})
                    df = df.rename(columns=mapping)
                
                elif transform_type == "drop_columns":
                    columns = transform.get("columns", [])
                    df = df.drop(columns=columns, errors="ignore")
                
                elif transform_type == "select_columns":
                    columns = transform.get("columns", [])
                    df = df[columns]
                
                elif transform_type == "create_column":
                    name = transform.get("name")
                    expression = transform.get("expression")
                    if name and expression:
                        # Use eval to apply the expression (with safety considerations)
                        df[name] = df.eval(expression)
                
                elif transform_type == "apply_function":
                    column = transform.get("column")
                    function = transform.get("function")
                    new_column = transform.get("new_column", column)
                    
                    if function == "log":
                        df[new_column] = np.log(df[column].replace(0, np.nan))
                    elif function == "sqrt":
                        df[new_column] = np.sqrt(df[column])
                    elif function == "square":
                        df[new_column] = np.square(df[column])
                    elif function == "abs":
                        df[new_column] = np.abs(df[column])
                    elif function == "to_datetime":
                        df[new_column] = pd.to_datetime(df[column], errors="coerce")
                
                elif transform_type == "one_hot_encode":
                    columns = transform.get("columns", [])
                    for col in columns:
                        if col in df.columns:
                            one_hot = pd.get_dummies(df[col], prefix=col)
                            df = pd.concat([df, one_hot], axis=1)
                            if transform.get("drop_original", True):
                                df = df.drop(columns=[col])
                
                elif transform_type == "bin_values":
                    column = transform.get("column")
                    bins = transform.get("bins", 5)
                    labels = transform.get("labels")
                    new_column = transform.get("new_column", f"{column}_binned")
                    
                    if isinstance(bins, int):
                        df[new_column] = pd.cut(df[column], bins=bins, labels=labels)
                    else:
                        df[new_column] = pd.cut(df[column], bins=bins, labels=labels)
            
            # Save processed data
            saved_path = await self._save_data(df, output_path)
            
            # Return summary
            new_columns = set(df.columns) - original_columns
            removed_columns = original_columns - set(df.columns)
            
            result = {
                "success": True,
                "rows": df.shape[0],
                "original_column_count": len(original_columns),
                "new_column_count": df.shape[1],
                "new_columns": list(new_columns),
                "removed_columns": list(removed_columns),
                "transformations_applied": len(transformations),
                "timestamp": datetime.now().isoformat()
            }
            
            if saved_path:
                result["saved_to"] = saved_path
            
            return result
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _normalize_data(
        self,
        data: Optional[Any] = None,
        data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        method: str = "minmax",
        columns: List[str] = [],
    ) -> Dict[str, Any]:
        """Normalize numerical data using various methods"""
        try:
            # Load data
            df = await self._load_data(data, data_path)
            
            # If no columns specified, use all numeric columns
            if not columns:
                columns = df.select_dtypes(include=np.number).columns.tolist()
            
            # Filter to only include numeric columns that exist
            columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            # Store normalization parameters for reference
            normalization_params = {}
            
            # Apply normalization
            if method == "minmax":
                for col in columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    normalization_params[col] = {"min": min_val, "max": max_val}
            
            elif method == "zscore":
                for col in columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    df[col] = (df[col] - mean) / std
                    normalization_params[col] = {"mean": mean, "std": std}
            
            elif method == "robust":
                for col in columns:
                    median = df[col].median()
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    df[col] = (df[col] - median) / iqr
                    normalization_params[col] = {"median": median, "iqr": iqr}
            
            elif method == "log":
                for col in columns:
                    # Add small constant to avoid log(0)
                    min_val = df[col].min()
                    offset = 1.0 if min_val >= 0 else abs(min_val) + 1.0
                    df[col] = np.log(df[col] + offset)
                    normalization_params[col] = {"offset": offset}
            
            # Save processed data
            saved_path = await self._save_data(df, output_path)
            
            # Return summary
            result = {
                "success": True,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "normalized_columns": columns,
                "normalization_method": method,
                "normalization_params": normalization_params,
                "timestamp": datetime.now().isoformat()
            }
            
            if saved_path:
                result["saved_to"] = saved_path
            
            return result
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _split_data(
        self,
        data: Optional[Any] = None,
        data_path: Optional[str] = None,
        train_output: Optional[str] = None,
        test_output: Optional[str] = None,
        validation_output: Optional[str] = None,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Split data into training, testing, and optional validation sets"""
        try:
            # Load data
            df = await self._load_data(data, data_path)
            
            # Calculate split sizes
            total_rows = len(df)
            test_rows = int(total_rows * test_size)
            validation_rows = int(total_rows * validation_size)
            train_rows = total_rows - test_rows - validation_rows
            
            # Shuffle the data
            shuffled_df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            
            # Split the data
            train_df = shuffled_df.iloc[:train_rows]
            test_df = shuffled_df.iloc[train_rows:train_rows + test_rows]
            validation_df = shuffled_df.iloc[train_rows + test_rows:] if validation_rows > 0 else None
            
            # Save the splits
            saved_paths = {}
            
            if train_output:
                train_path = await self._save_data(train_df, train_output)
                saved_paths["train"] = train_path
            
            if test_output:
                test_path = await self._save_data(test_df, test_output)
                saved_paths["test"] = test_path
            
            if validation_output and validation_df is not None:
                validation_path = await self._save_data(validation_df, validation_output)
                saved_paths["validation"] = validation_path
            
            # Return summary
            result = {
                "success": True,
                "total_rows": total_rows,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "validation_rows": validation_rows if validation_df is not None else 0,
                "train_percentage": train_rows / total_rows * 100,
                "test_percentage": test_rows / total_rows * 100,
                "validation_percentage": (validation_rows / total_rows * 100) if validation_df is not None else 0,
                "random_seed": random_seed,
                "saved_paths": saved_paths,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["clean", "transform", "normalize", "split"],
                    "description": "Preprocessing action to perform"
                },
                "data": {
                    "type": ["array", "object"],
                    "description": "Data to preprocess (alternative to data_path)"
                },
                "data_path": {
                    "type": "string",
                    "description": "Path to the data file to preprocess (alternative to data)"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save the preprocessed data"
                },
                "operations": {
                    "type": "array",
                    "description": "List of cleaning operations to apply",
                    "items": {
                        "type": "object"
                    }
                },
                "transformations": {
                    "type": "array",
                    "description": "List of transformations to apply",
                    "items": {
                        "type": "object"
                    }
                },
                "method": {
                    "type": "string",
                    "enum": ["minmax", "zscore", "robust", "log"],
                    "description": "Normalization method to use"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to normalize or transform"
                },
                "train_output": {
                    "type": "string",
                    "description": "Path to save the training data split"
                },
                "test_output": {
                    "type": "string",
                    "description": "Path to save the testing data split"
                },
                "validation_output": {
                    "type": "string",
                    "description": "Path to save the validation data split"
                },
                "test_size": {
                    "type": "number",
                    "description": "Proportion of data to use for testing"
                },
                "validation_size": {
                    "type": "number",
                    "description": "Proportion of data to use for validation"
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Random seed for reproducible data splitting"
                }
            },
            "required": ["action"],
            "allOf": [
                {
                    "if": {
                        "properties": {"action": {"enum": ["clean", "transform", "normalize"]}}
                    },
                    "then": {
                        "anyOf": [
                            {"required": ["data"]},
                            {"required": ["data_path"]}
                        ]
                    }
                },
                {
                    "if": {
                        "properties": {"action": {"enum": ["split"]}}
                    },
                    "then": {
                        "anyOf": [
                            {"required": ["data"]},
                            {"required": ["data_path"]}
                        ]
                    }
                }
            ]
        } 