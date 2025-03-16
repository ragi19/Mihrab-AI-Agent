"""
Example demonstrating the Data Agent for collecting and preprocessing real-time data
"""

import asyncio
import os
import json
from datetime import datetime
from pathlib import Path

from ...agents.data_agent import DataAgent
from ...factory import create_model
from ...utils.logging import get_logger, setup_logging

# Set up logging
setup_logging(level="INFO")
logger = get_logger("examples.data_agent")


async def run_data_collection_example():
    """Run a simple data collection and preprocessing example"""
    logger.info("Starting Data Agent example")
    
    # Create a model for the agent
    model = await create_model(provider_name="groq", model_name="deepseek-r1-distill-llama-70b")
    
    # Create data directory
    data_dir = os.path.join(os.getcwd(), "example_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create the Data Agent
    agent = DataAgent(
        model=model,
        data_dir=data_dir
    )
    logger.info(f"Created Data Agent with data directory: {data_dir}")
    
    # Example 1: Collect data from a public API
    logger.info("Example 1: Collecting data from a public API")
    weather_result = await agent.collect_data(
        source="https://api.open-meteo.com/v1/forecast",
        source_type="api",
        query_params={
            "latitude": 52.52,
            "longitude": 13.41,
            "current_weather": True,
            "hourly": "temperature_2m,precipitation"
        }
    )
    
    if weather_result.get("success", False):
        logger.info(f"Successfully collected weather data: {weather_result.get('data_type')}")
        
        # Get the path to the saved raw data
        weather_data_path = weather_result.get("saved_to")
        if weather_data_path:
            logger.info(f"Weather data saved to: {weather_data_path}")
            
            # Example 2: Preprocess the collected data
            logger.info("Example 2: Preprocessing the collected data")
            
            # Clean the data (example operation)
            clean_result = await agent.clean_data(
                data_path=weather_data_path,
                operations=[
                    {"type": "drop_na"},  # Drop rows with missing values
                ]
            )
            
            if clean_result.get("success", False):
                logger.info(f"Successfully cleaned data: {clean_result}")
                clean_data_path = clean_result.get("saved_to")
                
                # Transform the data (example transformations)
                transform_result = await agent.transform_data(
                    data_path=clean_data_path,
                    transformations=[
                        {
                            "type": "create_column",
                            "name": "temperature_fahrenheit",
                            "expression": "hourly_temperature_2m * 9/5 + 32"
                        }
                    ]
                )
                
                if transform_result.get("success", False):
                    logger.info(f"Successfully transformed data: {transform_result}")
                    transform_data_path = transform_result.get("saved_to")
                    
                    # Split the data for training
                    split_result = await agent.split_data(
                        data_path=transform_data_path,
                        test_size=0.2
                    )
                    
                    if split_result.get("success", False):
                        logger.info(f"Successfully split data: {split_result}")
    
    # Example 3: Start a data stream
    logger.info("Example 3: Starting a data stream")
    stream_result = await agent.start_data_stream(
        source="https://api.open-meteo.com/v1/forecast",
        source_type="api",
        interval=30,  # Collect every 30 seconds
        query_params={
            "latitude": 40.71,
            "longitude": -74.01,
            "current_weather": True
        },
        max_records=3  # Collect 3 records and stop
    )
    
    if stream_result.get("success", False):
        stream_id = stream_result.get("stream_id")
        logger.info(f"Started data stream with ID: {stream_id}")
        
        # Wait for some data to be collected
        logger.info("Waiting for stream to collect data...")
        await asyncio.sleep(65)  # Wait for at least 2 collections
        
        # List active streams
        streams = await agent.list_data_streams()
        logger.info(f"Active streams: {streams}")
        
        # Wait for stream to complete
        logger.info("Waiting for stream to complete...")
        await asyncio.sleep(65)  # Wait for the stream to complete
        
        # List streams again
        streams = await agent.list_data_streams()
        logger.info(f"Streams after completion: {streams}")
    
    # Example 4: Process a message with the agent
    logger.info("Example 4: Processing a message with the agent")
    message_result = await agent.process(
        "Please collect current weather data for Tokyo, Japan and preprocess it to extract temperature trends."
    )
    logger.info(f"Agent response: {message_result}")
    
    logger.info("Data Agent example completed")


if __name__ == "__main__":
    asyncio.run(run_data_collection_example()) 