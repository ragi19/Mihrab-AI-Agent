"""
Knowledge base tool for structured information storage and retrieval
"""

import json
import os
from typing import Any, Dict, List, Optional, Union
import asyncio
from pathlib import Path
from datetime import datetime
import re

from ..base import BaseTool


class KnowledgeBaseTool(BaseTool):
    """Tool for managing a structured knowledge base"""

    def __init__(self, kb_dir: str = "./knowledge_base"):
        super().__init__(
            name="knowledge_base",
            description="Store and retrieve structured information in a knowledge base",
        )
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.kb_dir / "index.json"
        self._ensure_index_file()
    
    def _ensure_index_file(self):
        """Ensure the index file exists"""
        if not self.index_file.exists():
            with open(self.index_file, "w") as f:
                json.dump({
                    "topics": {},
                    "entries": [],
                    "relationships": []
                }, f, indent=2)

    async def _execute(self, parameters: Dict[str, Any]) -> str:
        """Execute the knowledge base tool with the given parameters"""
        action = parameters.get("action", "query")
        
        if action == "add_entry":
            return await self._add_entry(
                title=parameters.get("title", "Untitled Entry"),
                content=parameters.get("content", ""),
                topics=parameters.get("topics", []),
                metadata=parameters.get("metadata", {}),
            )
        elif action == "update_entry":
            return await self._update_entry(
                entry_id=parameters.get("entry_id"),
                title=parameters.get("title"),
                content=parameters.get("content"),
                topics=parameters.get("topics"),
                metadata=parameters.get("metadata"),
            )
        elif action == "get_entry":
            return await self._get_entry(
                entry_id=parameters.get("entry_id"),
            )
        elif action == "query":
            return await self._query_kb(
                query=parameters.get("query", ""),
                topics=parameters.get("topics", []),
                limit=parameters.get("limit", 5),
            )
        elif action == "add_relationship":
            return await self._add_relationship(
                source_id=parameters.get("source_id"),
                target_id=parameters.get("target_id"),
                relationship_type=parameters.get("relationship_type", "related"),
                description=parameters.get("description", ""),
            )
        elif action == "list_topics":
            return await self._list_topics()
        else:
            return f"Unknown action: {action}"
    
    async def _add_entry(
        self,
        title: str,
        content: str,
        topics: List[str] = [],
        metadata: Dict[str, Any] = {},
    ) -> str:
        """Add a new entry to the knowledge base"""
        # Generate a unique ID for the entry
        entry_id = f"kb_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create entry file
        entry_file = self.kb_dir / f"{entry_id}.json"
        entry_data = {
            "id": entry_id,
            "title": title,
            "content": content,
            "topics": topics,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        with open(entry_file, "w") as f:
            json.dump(entry_data, f, indent=2)
        
        # Update index
        with open(self.index_file, "r") as f:
            index = json.load(f)
        
        # Add entry to entries list
        index["entries"].append({
            "id": entry_id,
            "title": title,
            "topics": topics,
            "created_at": entry_data["created_at"],
            "updated_at": entry_data["updated_at"],
        })
        
        # Update topics
        for topic in topics:
            topic_key = self._normalize_topic(topic)
            if topic_key not in index["topics"]:
                index["topics"][topic_key] = {
                    "name": topic,
                    "entry_count": 0
                }
            
            index["topics"][topic_key]["entry_count"] += 1
        
        # Save updated index
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)
        
        return f"Added knowledge base entry '{title}' with ID: {entry_id}"
    
    async def _update_entry(
        self,
        entry_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        topics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Update an existing entry in the knowledge base"""
        entry_file = self.kb_dir / f"{entry_id}.json"
        
        if not entry_file.exists():
            return f"Entry with ID {entry_id} not found"
        
        # Load current entry
        with open(entry_file, "r") as f:
            entry_data = json.load(f)
        
        # Track original topics for index updates
        original_topics = entry_data["topics"]
        
        # Update fields if provided
        if title is not None:
            entry_data["title"] = title
        
        if content is not None:
            entry_data["content"] = content
        
        if topics is not None:
            entry_data["topics"] = topics
        
        if metadata is not None:
            entry_data["metadata"] = metadata
        
        # Update timestamp
        entry_data["updated_at"] = datetime.now().isoformat()
        
        # Save updated entry
        with open(entry_file, "w") as f:
            json.dump(entry_data, f, indent=2)
        
        # Update index
        with open(self.index_file, "r") as f:
            index = json.load(f)
        
        # Update entry in entries list
        for i, entry in enumerate(index["entries"]):
            if entry["id"] == entry_id:
                index["entries"][i]["title"] = entry_data["title"]
                index["entries"][i]["topics"] = entry_data["topics"]
                index["entries"][i]["updated_at"] = entry_data["updated_at"]
                break
        
        # Update topics if they changed
        if topics is not None:
            # Remove entry from old topics
            for topic in original_topics:
                topic_key = self._normalize_topic(topic)
                if topic_key in index["topics"]:
                    index["topics"][topic_key]["entry_count"] -= 1
                    
                    # Remove topic if no more entries
                    if index["topics"][topic_key]["entry_count"] <= 0:
                        del index["topics"][topic_key]
            
            # Add entry to new topics
            for topic in topics:
                topic_key = self._normalize_topic(topic)
                if topic_key not in index["topics"]:
                    index["topics"][topic_key] = {
                        "name": topic,
                        "entry_count": 0
                    }
                
                index["topics"][topic_key]["entry_count"] += 1
        
        # Save updated index
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)
        
        return f"Updated knowledge base entry '{entry_data['title']}' (ID: {entry_id})"
    
    async def _get_entry(self, entry_id: str) -> str:
        """Retrieve an entry from the knowledge base by ID"""
        entry_file = self.kb_dir / f"{entry_id}.json"
        
        if not entry_file.exists():
            return f"Entry with ID {entry_id} not found"
        
        # Load entry
        with open(entry_file, "r") as f:
            entry_data = json.load(f)
        
        # Format the output
        result = f"# {entry_data['title']}\n\n"
        result += f"ID: {entry_data['id']}\n"
        result += f"Topics: {', '.join(entry_data['topics'])}\n"
        result += f"Created: {entry_data['created_at'][:10]}\n"
        result += f"Updated: {entry_data['updated_at'][:10]}\n\n"
        result += f"## Content\n\n{entry_data['content']}\n\n"
        
        # Add metadata if present
        if entry_data.get("metadata"):
            result += "## Metadata\n\n"
            for key, value in entry_data["metadata"].items():
                result += f"- {key}: {value}\n"
        
        # Get related entries
        with open(self.index_file, "r") as f:
            index = json.load(f)
        
        related_entries = []
        for rel in index["relationships"]:
            if rel["source_id"] == entry_id:
                related_entries.append({
                    "id": rel["target_id"],
                    "type": rel["relationship_type"],
                    "description": rel["description"]
                })
            elif rel["target_id"] == entry_id:
                related_entries.append({
                    "id": rel["source_id"],
                    "type": f"reverse_{rel['relationship_type']}",
                    "description": rel["description"]
                })
        
        if related_entries:
            result += "\n## Related Entries\n\n"
            for rel in related_entries:
                # Get the title of the related entry
                related_title = "Unknown Entry"
                for entry in index["entries"]:
                    if entry["id"] == rel["id"]:
                        related_title = entry["title"]
                        break
                
                result += f"- {related_title} (ID: {rel['id']})\n"
                result += f"  Relationship: {rel['type']}\n"
                if rel["description"]:
                    result += f"  Description: {rel['description']}\n"
                result += "\n"
        
        return result
    
    async def _query_kb(
        self,
        query: str = "",
        topics: List[str] = [],
        limit: int = 5
    ) -> str:
        """Query the knowledge base for relevant entries"""
        # Load index
        with open(self.index_file, "r") as f:
            index = json.load(f)
        
        # Filter by topics if specified
        filtered_entries = index["entries"]
        if topics:
            filtered_entries = []
            for entry in index["entries"]:
                if any(topic in entry["topics"] for topic in topics):
                    filtered_entries.append(entry)
        
        if not filtered_entries:
            if topics:
                return f"No entries found for topics: {', '.join(topics)}"
            else:
                return "No entries found in the knowledge base"
        
        # If no query, just return the most recent entries
        if not query:
            filtered_entries.sort(key=lambda e: e["updated_at"], reverse=True)
            entries = filtered_entries[:limit]
            
            result = "Recent Knowledge Base Entries:\n\n"
            for entry in entries:
                result += f"- {entry['title']} (ID: {entry['id']})\n"
                result += f"  Topics: {', '.join(entry['topics'])}\n"
                result += f"  Updated: {entry['updated_at'][:10]}\n\n"
            
            return result
        
        # Simple search implementation
        query = query.lower()
        matches = []
        
        # First search in the index for title matches
        for entry in filtered_entries:
            score = 0
            
            # Title match (highest weight)
            if query in entry["title"].lower():
                score += 10
            
            # Topic match
            for topic in entry["topics"]:
                if query in topic.lower():
                    score += 5
            
            if score > 0:
                matches.append((entry, score))
        
        # Then search in the full content of entries
        for entry_info in filtered_entries:
            # Skip if already matched by title or topic
            if any(m[0]["id"] == entry_info["id"] for m in matches):
                continue
            
            entry_file = self.kb_dir / f"{entry_info['id']}.json"
            if entry_file.exists():
                with open(entry_file, "r") as f:
                    entry_data = json.load(f)
                
                score = 0
                
                # Content match
                if query in entry_data["content"].lower():
                    score += 3
                
                # Metadata match
                for key, value in entry_data.get("metadata", {}).items():
                    if isinstance(value, str) and query in value.lower():
                        score += 2
                
                if score > 0:
                    matches.append((entry_info, score))
        
        # Sort by score (descending)
        matches.sort(key=lambda m: m[1], reverse=True)
        
        # Format results
        if not matches:
            return f"No entries found matching '{query}'"
        
        result = f"Search results for '{query}':\n\n"
        
        for entry, score in matches[:limit]:
            result += f"- {entry['title']} (ID: {entry['id']})\n"
            result += f"  Topics: {', '.join(entry['topics'])}\n"
            result += f"  Relevance: {score}\n\n"
        
        return result
    
    async def _add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str = "related",
        description: str = "",
    ) -> str:
        """Add a relationship between two knowledge base entries"""
        # Verify both entries exist
        source_file = self.kb_dir / f"{source_id}.json"
        target_file = self.kb_dir / f"{target_id}.json"
        
        if not source_file.exists():
            return f"Source entry with ID {source_id} not found"
        
        if not target_file.exists():
            return f"Target entry with ID {target_id} not found"
        
        # Load index
        with open(self.index_file, "r") as f:
            index = json.load(f)
        
        # Check if relationship already exists
        for rel in index["relationships"]:
            if (rel["source_id"] == source_id and 
                rel["target_id"] == target_id and 
                rel["relationship_type"] == relationship_type):
                return f"Relationship of type '{relationship_type}' already exists between these entries"
        
        # Add relationship
        relationship = {
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": relationship_type,
            "description": description,
            "created_at": datetime.now().isoformat(),
        }
        
        index["relationships"].append(relationship)
        
        # Save updated index
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)
        
        # Get entry titles for confirmation message
        source_title = "Unknown Entry"
        target_title = "Unknown Entry"
        
        for entry in index["entries"]:
            if entry["id"] == source_id:
                source_title = entry["title"]
            if entry["id"] == target_id:
                target_title = entry["title"]
        
        return f"Added '{relationship_type}' relationship from '{source_title}' to '{target_title}'"
    
    async def _list_topics(self) -> str:
        """List all topics in the knowledge base"""
        # Load index
        with open(self.index_file, "r") as f:
            index = json.load(f)
        
        topics = index["topics"]
        
        if not topics:
            return "No topics found in the knowledge base"
        
        # Sort topics by entry count (descending)
        sorted_topics = sorted(
            topics.values(),
            key=lambda t: t["entry_count"],
            reverse=True
        )
        
        result = "Knowledge Base Topics:\n\n"
        for topic in sorted_topics:
            result += f"- {topic['name']} ({topic['entry_count']} entries)\n"
        
        return result
    
    def _normalize_topic(self, topic: str) -> str:
        """Normalize a topic string for use as a dictionary key"""
        return re.sub(r'[^a-z0-9]', '', topic.lower())
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add_entry", "update_entry", "get_entry", "query", "add_relationship", "list_topics"],
                    "description": "Action to perform on the knowledge base",
                },
                "title": {
                    "type": "string",
                    "description": "Title for the knowledge base entry (used with add_entry/update_entry actions)",
                },
                "content": {
                    "type": "string",
                    "description": "Content for the knowledge base entry (used with add_entry/update_entry actions)",
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of topics for the entry (used with add_entry/update_entry/query actions)",
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata for the entry (used with add_entry/update_entry actions)",
                },
                "entry_id": {
                    "type": "string",
                    "description": "ID of the entry to operate on (used with update_entry/get_entry actions)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (used with query action)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (used with query action)",
                },
                "source_id": {
                    "type": "string",
                    "description": "ID of the source entry (used with add_relationship action)",
                },
                "target_id": {
                    "type": "string",
                    "description": "ID of the target entry (used with add_relationship action)",
                },
                "relationship_type": {
                    "type": "string",
                    "description": "Type of relationship (used with add_relationship action)",
                },
                "description": {
                    "type": "string",
                    "description": "Description of the relationship (used with add_relationship action)",
                },
            },
            "required": ["action"],
        }
