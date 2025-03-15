"""
Calendar management tools for agents
"""

import json
import os
from typing import Any, Dict, List, Optional
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

from ..base import BaseTool


class CalendarTool(BaseTool):
    """Tool for managing calendar events and appointments"""

    def __init__(self, calendar_file: str = "./calendar_data.json"):
        super().__init__(
            name="calendar",
            description="Manage calendar events and appointments",
        )
        self.calendar_file = Path(calendar_file)
        self._ensure_calendar_file()
    
    def _ensure_calendar_file(self):
        """Ensure the calendar file exists"""
        if not self.calendar_file.exists():
            self.calendar_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.calendar_file, "w") as f:
                json.dump({"events": []}, f)

    async def _execute(self, parameters: Dict[str, Any]) -> str:
        """Execute the calendar tool with the given parameters"""
        action = parameters.get("action", "list")
        
        if action == "add":
            return await self._add_event(
                title=parameters.get("title", "Untitled Event"),
                start_time=parameters.get("start_time"),
                end_time=parameters.get("end_time"),
                description=parameters.get("description", ""),
                location=parameters.get("location", ""),
                participants=parameters.get("participants", []),
            )
        elif action == "list":
            return await self._list_events(
                date=parameters.get("date"),
                start_date=parameters.get("start_date"),
                end_date=parameters.get("end_date"),
            )
        elif action == "update":
            return await self._update_event(
                event_id=parameters.get("event_id"),
                title=parameters.get("title"),
                start_time=parameters.get("start_time"),
                end_time=parameters.get("end_time"),
                description=parameters.get("description"),
                location=parameters.get("location"),
                participants=parameters.get("participants"),
            )
        elif action == "delete":
            return await self._delete_event(
                event_id=parameters.get("event_id"),
            )
        elif action == "find":
            return await self._find_events(
                query=parameters.get("query", ""),
            )
        else:
            return f"Unknown action: {action}"
    
    async def _add_event(
        self,
        title: str,
        start_time: str,
        end_time: str,
        description: str = "",
        location: str = "",
        participants: List[str] = [],
    ) -> str:
        """Add a new event to the calendar"""
        # Validate time formats
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
        except ValueError:
            return "Invalid time format. Please use ISO format (YYYY-MM-DDTHH:MM:SS)"
        
        if end_dt <= start_dt:
            return "End time must be after start time"
        
        # Generate a unique ID for the event
        event_id = f"evt_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Load current events
        with open(self.calendar_file, "r") as f:
            calendar_data = json.load(f)
        
        # Check for conflicts
        conflicts = []
        for event in calendar_data["events"]:
            event_start = datetime.fromisoformat(event["start_time"])
            event_end = datetime.fromisoformat(event["end_time"])
            
            if (start_dt < event_end and end_dt > event_start):
                conflicts.append(event)
        
        # Add the new event
        new_event = {
            "id": event_id,
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "description": description,
            "location": location,
            "participants": participants,
            "created_at": datetime.now().isoformat(),
        }
        
        calendar_data["events"].append(new_event)
        
        # Save updated calendar
        with open(self.calendar_file, "w") as f:
            json.dump(calendar_data, f, indent=2)
        
        # Return result with conflict warnings if any
        if conflicts:
            conflict_details = "\n".join([
                f"- {event['title']} ({event['start_time']} to {event['end_time']})"
                for event in conflicts
            ])
            return (
                f"Added event '{title}' with ID: {event_id}\n\n"
                f"WARNING: This event conflicts with {len(conflicts)} existing events:\n"
                f"{conflict_details}"
            )
        
        return f"Added event '{title}' with ID: {event_id}"
    
    async def _list_events(
        self,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """List events in the calendar for a specific date range"""
        # Load events
        with open(self.calendar_file, "r") as f:
            calendar_data = json.load(f)
        
        events = calendar_data["events"]
        
        # Filter by date if specified
        if date:
            try:
                target_date = datetime.fromisoformat(date).date()
                events = [
                    event for event in events
                    if datetime.fromisoformat(event["start_time"]).date() == target_date
                ]
            except ValueError:
                return f"Invalid date format: {date}. Please use ISO format (YYYY-MM-DD)"
        
        # Filter by date range if specified
        if start_date or end_date:
            try:
                start = datetime.fromisoformat(start_date).date() if start_date else None
                end = datetime.fromisoformat(end_date).date() if end_date else None
                
                filtered_events = []
                for event in events:
                    event_date = datetime.fromisoformat(event["start_time"]).date()
                    if (not start or event_date >= start) and (not end or event_date <= end):
                        filtered_events.append(event)
                
                events = filtered_events
            except ValueError:
                return "Invalid date format. Please use ISO format (YYYY-MM-DD)"
        
        # Sort events by start time
        events.sort(key=lambda e: e["start_time"])
        
        # Format the results
        if not events:
            if date:
                return f"No events found for {date}"
            elif start_date or end_date:
                date_range = f"{start_date or 'any'} to {end_date or 'any'}"
                return f"No events found in the date range: {date_range}"
            else:
                return "No events found in the calendar"
        
        # Group events by date
        events_by_date = {}
        for event in events:
            event_date = datetime.fromisoformat(event["start_time"]).date().isoformat()
            if event_date not in events_by_date:
                events_by_date[event_date] = []
            events_by_date[event_date].append(event)
        
        # Format the output
        result = "Calendar Events:\n\n"
        
        for date, day_events in sorted(events_by_date.items()):
            # Format the date
            date_obj = datetime.fromisoformat(date).date()
            formatted_date = date_obj.strftime("%A, %B %d, %Y")
            result += f"{formatted_date}:\n"
            
            for event in day_events:
                # Format times
                start = datetime.fromisoformat(event["start_time"])
                end = datetime.fromisoformat(event["end_time"])
                time_str = f"{start.strftime('%I:%M %p')} - {end.strftime('%I:%M %p')}"
                
                # Basic event details
                result += f"- {time_str}: {event['title']} (ID: {event['id']})\n"
                
                # Additional details if present
                if event.get("location"):
                    result += f"  Location: {event['location']}\n"
                
                if event.get("participants"):
                    participants = ", ".join(event["participants"])
                    result += f"  Participants: {participants}\n"
                
                if event.get("description"):
                    desc = event["description"]
                    if len(desc) > 50:
                        desc = desc[:47] + "..."
                    result += f"  Description: {desc}\n"
                
                result += "\n"
        
        return result
    
    async def _update_event(
        self,
        event_id: str,
        title: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        participants: Optional[List[str]] = None,
    ) -> str:
        """Update an existing event in the calendar"""
        # Load events
        with open(self.calendar_file, "r") as f:
            calendar_data = json.load(f)
        
        # Find the event
        event_index = None
        for i, event in enumerate(calendar_data["events"]):
            if event["id"] == event_id:
                event_index = i
                break
        
        if event_index is None:
            return f"Event with ID {event_id} not found"
        
        event = calendar_data["events"][event_index]
        
        # Update fields if provided
        if title is not None:
            event["title"] = title
        
        # Handle time updates and validation
        if start_time is not None or end_time is not None:
            new_start = start_time if start_time is not None else event["start_time"]
            new_end = end_time if end_time is not None else event["end_time"]
            
            try:
                start_dt = datetime.fromisoformat(new_start)
                end_dt = datetime.fromisoformat(new_end)
                
                if end_dt <= start_dt:
                    return "End time must be after start time"
                
                event["start_time"] = new_start
                event["end_time"] = new_end
            except ValueError:
                return "Invalid time format. Please use ISO format (YYYY-MM-DDTHH:MM:SS)"
        
        if description is not None:
            event["description"] = description
        
        if location is not None:
            event["location"] = location
        
        if participants is not None:
            event["participants"] = participants
        
        # Update the event
        calendar_data["events"][event_index] = event
        
        # Save updated calendar
        with open(self.calendar_file, "w") as f:
            json.dump(calendar_data, f, indent=2)
        
        return f"Updated event '{event['title']}' (ID: {event_id})"
    
    async def _delete_event(self, event_id: str) -> str:
        """Delete an event from the calendar"""
        # Load events
        with open(self.calendar_file, "r") as f:
            calendar_data = json.load(f)
        
        # Find the event
        event_index = None
        for i, event in enumerate(calendar_data["events"]):
            if event["id"] == event_id:
                event_index = i
                break
        
        if event_index is None:
            return f"Event with ID {event_id} not found"
        
        # Store event details for confirmation
        event = calendar_data["events"][event_index]
        
        # Remove the event
        del calendar_data["events"][event_index]
        
        # Save updated calendar
        with open(self.calendar_file, "w") as f:
            json.dump(calendar_data, f, indent=2)
        
        return f"Deleted event '{event['title']}' (ID: {event_id})"
    
    async def _find_events(self, query: str) -> str:
        """Search for events matching a query string"""
        if not query:
            return "Please provide a search query"
        
        # Load events
        with open(self.calendar_file, "r") as f:
            calendar_data = json.load(f)
        
        # Search for matching events
        query = query.lower()
        matches = []
        
        for event in calendar_data["events"]:
            if (query in event["title"].lower() or
                query in event.get("description", "").lower() or
                query in event.get("location", "").lower() or
                any(query in participant.lower() for participant in event.get("participants", []))):
                matches.append(event)
        
        # Sort matches by start time
        matches.sort(key=lambda e: e["start_time"])
        
        # Format results
        if not matches:
            return f"No events found matching '{query}'"
        
        result = f"Found {len(matches)} events matching '{query}':\n\n"
        
        for event in matches:
            # Format times
            start = datetime.fromisoformat(event["start_time"])
            end = datetime.fromisoformat(event["end_time"])
            date_str = start.strftime("%Y-%m-%d")
            time_str = f"{start.strftime('%I:%M %p')} - {end.strftime('%I:%M %p')}"
            
            result += f"- {date_str}, {time_str}: {event['title']} (ID: {event['id']})\n"
            
            if event.get("location"):
                result += f"  Location: {event['location']}\n"
            
            if event.get("description"):
                desc = event["description"]
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                result += f"  Description: {desc}\n"
            
            result += "\n"
        
        return result
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "update", "delete", "find"],
                    "description": "Action to perform on calendar events",
                },
                "title": {
                    "type": "string",
                    "description": "Title of the event (used with add/update actions)",
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in ISO format (YYYY-MM-DDTHH:MM:SS) (used with add/update actions)",
                },
                "end_time": {
                    "type": "string",
                    "description": "End time in ISO format (YYYY-MM-DDTHH:MM:SS) (used with add/update actions)",
                },
                "description": {
                    "type": "string",
                    "description": "Description of the event (used with add/update actions)",
                },
                "location": {
                    "type": "string",
                    "description": "Location of the event (used with add/update actions)",
                },
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of participants (used with add/update actions)",
                },
                "event_id": {
                    "type": "string",
                    "description": "ID of the event to operate on (used with update/delete actions)",
                },
                "date": {
                    "type": "string",
                    "description": "Date in ISO format (YYYY-MM-DD) to list events for (used with list action)",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in ISO format (YYYY-MM-DD) for date range (used with list action)",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in ISO format (YYYY-MM-DD) for date range (used with list action)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (used with find action)",
                },
            },
            "required": ["action"],
        }
