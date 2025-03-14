"""
Memory and knowledge management tools for storing and retrieving information
"""

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ...utils.logging import get_logger
from ..base import BaseTool

logger = get_logger("tools.memory")


class MemoryStoreTool(BaseTool):
    """Tool for storing information in a persistent memory store"""

    def __init__(self, memory_path: Optional[str] = None):
        super().__init__(
            name="memory_store",
            description="Store information in a persistent memory database",
        )
        self._parameters = {
            "key": {
                "type": "string",
                "description": "Unique identifier for the memory item",
            },
            "value": {
                "type": "object",
                "description": "Information to store (can be any JSON-serializable data)",
            },
            "namespace": {
                "type": "string",
                "description": "Optional namespace for organizing memories",
                "default": "default",
            },
            "ttl": {
                "type": "integer",
                "description": "Time-to-live in seconds (0 for permanent)",
                "default": 0,
            },
            "tags": {
                "type": "array",
                "description": "Optional tags for categorizing memories",
                "items": {"type": "string"},
                "default": [],
            },
        }
        self._required_params = ["key", "value"]

        # Set up memory database
        self.memory_path = memory_path or os.path.join(
            os.path.expanduser("~"), ".mihrabai", "memory.db"
        )
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        self._init_db()

        logger.info(
            f"Initialized tool: memory_store with database at {self.memory_path}"
        )

    def _init_db(self) -> None:
        """Initialize the SQLite database for memory storage"""
        conn = sqlite3.connect(self.memory_path)
        cursor = conn.cursor()

        # Create memories table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            namespace TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER,
            UNIQUE(key, namespace)
        )
        """
        )

        # Create tags table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS memory_tags (
            memory_id INTEGER,
            tag TEXT NOT NULL,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            UNIQUE(memory_id, tag)
        )
        """
        )

        # Create index for faster lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag)"
        )

        conn.commit()
        conn.close()

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: int = 0,
        tags: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the memory store tool

        Args:
            key: Unique identifier for the memory item
            value: Information to store
            namespace: Optional namespace for organizing memories
            ttl: Time-to-live in seconds (0 for permanent)
            tags: Optional tags for categorizing memories

        Returns:
            Dictionary with storage result
        """
        try:
            # Serialize value to JSON
            serialized_value = json.dumps(value)

            # Calculate expiration time
            current_time = int(time.time())
            expires_at = current_time + ttl if ttl > 0 else None

            # Connect to database
            conn = sqlite3.connect(self.memory_path)
            cursor = conn.cursor()

            # Clean up expired memories
            cursor.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                (current_time,),
            )

            # Insert or update memory
            cursor.execute(
                """
            INSERT INTO memories (key, namespace, value, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key, namespace) DO UPDATE SET
                value = excluded.value,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at
            """,
                (key, namespace, serialized_value, current_time, expires_at),
            )

            # Get the memory ID
            cursor.execute(
                "SELECT id FROM memories WHERE key = ? AND namespace = ?",
                (key, namespace),
            )
            memory_id = cursor.fetchone()[0]

            # Delete existing tags
            cursor.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))

            # Insert new tags
            if tags:
                for tag in tags:
                    cursor.execute(
                        "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                        (memory_id, tag),
                    )

            conn.commit()
            conn.close()

            return {
                "status": "success",
                "key": key,
                "namespace": namespace,
                "stored_at": datetime.fromtimestamp(current_time).isoformat(),
                "expires_at": (
                    datetime.fromtimestamp(expires_at).isoformat()
                    if expires_at
                    else None
                ),
                "tags": tags or [],
            }
        except Exception as e:
            logger.error(f"Memory store error: {e}")
            return {
                "status": "error",
                "error": f"Memory storage error: {str(e)}",
                "key": key,
                "namespace": namespace,
            }


class MemoryRetrieveTool(BaseTool):
    """Tool for retrieving information from a persistent memory store"""

    def __init__(self, memory_path: Optional[str] = None):
        super().__init__(
            name="memory_retrieve",
            description="Retrieve information from a persistent memory database",
        )
        self._parameters = {
            "key": {
                "type": "string",
                "description": "Unique identifier for the memory item",
            },
            "namespace": {
                "type": "string",
                "description": "Namespace where the memory is stored",
                "default": "default",
            },
        }
        self._required_params = ["key"]

        # Set up memory database path
        self.memory_path = memory_path or os.path.join(
            os.path.expanduser("~"), ".mihrabai", "memory.db"
        )

        logger.info(
            f"Initialized tool: memory_retrieve with database at {self.memory_path}"
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self, key: str, namespace: str = "default", **kwargs
    ) -> Dict[str, Any]:
        """Execute the memory retrieve tool

        Args:
            key: Unique identifier for the memory item
            namespace: Namespace where the memory is stored

        Returns:
            Dictionary with retrieved memory or error
        """
        try:
            # Check if database exists
            if not os.path.exists(self.memory_path):
                return {
                    "status": "error",
                    "error": "Memory database does not exist",
                    "key": key,
                    "namespace": namespace,
                }

            # Connect to database
            conn = sqlite3.connect(self.memory_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()

            # Clean up expired memories
            current_time = int(time.time())
            cursor.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                (current_time,),
            )

            # Retrieve memory
            cursor.execute(
                """
            SELECT id, key, namespace, value, created_at, expires_at
            FROM memories
            WHERE key = ? AND namespace = ?
            """,
                (key, namespace),
            )

            row = cursor.fetchone()

            if not row:
                conn.close()
                return {
                    "status": "error",
                    "error": "Memory not found",
                    "key": key,
                    "namespace": namespace,
                }

            memory_id = row["id"]

            # Retrieve tags
            cursor.execute(
                "SELECT tag FROM memory_tags WHERE memory_id = ?", (memory_id,)
            )
            tags = [r["tag"] for r in cursor.fetchall()]

            # Parse value
            value = json.loads(row["value"])

            conn.close()

            return {
                "status": "success",
                "key": row["key"],
                "namespace": row["namespace"],
                "value": value,
                "created_at": datetime.fromtimestamp(row["created_at"]).isoformat(),
                "expires_at": (
                    datetime.fromtimestamp(row["expires_at"]).isoformat()
                    if row["expires_at"]
                    else None
                ),
                "tags": tags,
            }
        except Exception as e:
            logger.error(f"Memory retrieve error: {e}")
            return {
                "status": "error",
                "error": f"Memory retrieval error: {str(e)}",
                "key": key,
                "namespace": namespace,
            }


class MemorySearchTool(BaseTool):
    """Tool for searching information in a persistent memory store"""

    def __init__(self, memory_path: Optional[str] = None):
        super().__init__(
            name="memory_search",
            description="Search for information in a persistent memory database",
        )
        self._parameters = {
            "query_type": {
                "type": "string",
                "description": "Type of search query",
                "enum": ["namespace", "tag", "key_prefix"],
                "default": "namespace",
            },
            "query": {"type": "string", "description": "Search query value"},
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10,
            },
        }
        self._required_params = ["query_type", "query"]

        # Set up memory database path
        self.memory_path = memory_path or os.path.join(
            os.path.expanduser("~"), ".mihrabai", "memory.db"
        )

        logger.info(
            f"Initialized tool: memory_search with database at {self.memory_path}"
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self, query_type: str, query: str, limit: int = 10, **kwargs
    ) -> Dict[str, Any]:
        """Execute the memory search tool

        Args:
            query_type: Type of search query (namespace, tag, key_prefix)
            query: Search query value
            limit: Maximum number of results to return

        Returns:
            Dictionary with search results
        """
        try:
            # Check if database exists
            if not os.path.exists(self.memory_path):
                return {
                    "status": "error",
                    "error": "Memory database does not exist",
                    "query_type": query_type,
                    "query": query,
                }

            # Connect to database
            conn = sqlite3.connect(self.memory_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()

            # Clean up expired memories
            current_time = int(time.time())
            cursor.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                (current_time,),
            )

            results = []

            if query_type == "namespace":
                # Search by namespace
                cursor.execute(
                    """
                SELECT id, key, namespace, value, created_at, expires_at
                FROM memories
                WHERE namespace = ?
                LIMIT ?
                """,
                    (query, limit),
                )
            elif query_type == "tag":
                # Search by tag
                cursor.execute(
                    """
                SELECT m.id, m.key, m.namespace, m.value, m.created_at, m.expires_at
                FROM memories m
                JOIN memory_tags t ON m.id = t.memory_id
                WHERE t.tag = ?
                LIMIT ?
                """,
                    (query, limit),
                )
            elif query_type == "key_prefix":
                # Search by key prefix
                cursor.execute(
                    """
                SELECT id, key, namespace, value, created_at, expires_at
                FROM memories
                WHERE key LIKE ?
                LIMIT ?
                """,
                    (f"{query}%", limit),
                )
            else:
                conn.close()
                return {
                    "status": "error",
                    "error": f"Invalid query type: {query_type}",
                    "query": query,
                }

            rows = cursor.fetchall()

            for row in rows:
                memory_id = row["id"]

                # Retrieve tags
                cursor.execute(
                    "SELECT tag FROM memory_tags WHERE memory_id = ?", (memory_id,)
                )
                tags = [r["tag"] for r in cursor.fetchall()]

                # Parse value
                value = json.loads(row["value"])

                results.append(
                    {
                        "key": row["key"],
                        "namespace": row["namespace"],
                        "value": value,
                        "created_at": datetime.fromtimestamp(
                            row["created_at"]
                        ).isoformat(),
                        "expires_at": (
                            datetime.fromtimestamp(row["expires_at"]).isoformat()
                            if row["expires_at"]
                            else None
                        ),
                        "tags": tags,
                    }
                )

            conn.close()

            return {
                "status": "success",
                "query_type": query_type,
                "query": query,
                "count": len(results),
                "results": results,
            }
        except Exception as e:
            logger.error(f"Memory search error: {e}")
            return {
                "status": "error",
                "error": f"Memory search error: {str(e)}",
                "query_type": query_type,
                "query": query,
            }


class KnowledgeBaseTool(BaseTool):
    """Tool for managing a simple knowledge base with facts and relationships"""

    def __init__(self, kb_path: Optional[str] = None):
        super().__init__(
            name="knowledge_base",
            description="Manage a knowledge base of facts and relationships",
        )
        self._parameters = {
            "operation": {
                "type": "string",
                "description": "Operation to perform on the knowledge base",
                "enum": [
                    "add_fact",
                    "query_facts",
                    "add_relationship",
                    "query_relationships",
                ],
                "default": "query_facts",
            },
            "entity": {
                "type": "string",
                "description": "Entity name for fact or relationship",
            },
            "attribute": {"type": "string", "description": "Attribute name for fact"},
            "value": {
                "type": "string",
                "description": "Value for fact or relationship",
            },
            "relation": {
                "type": "string",
                "description": "Relation type for relationship",
            },
            "target": {
                "type": "string",
                "description": "Target entity for relationship",
            },
            "query": {
                "type": "string",
                "description": "Query string for searching facts or relationships",
            },
        }
        self._required_params = ["operation"]

        # Set up knowledge base path
        self.kb_path = kb_path or os.path.join(
            os.path.expanduser("~"), ".mihrabai", "knowledge.db"
        )
        os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
        self._init_db()

        logger.info(f"Initialized tool: knowledge_base with database at {self.kb_path}")

    def _init_db(self) -> None:
        """Initialize the SQLite database for knowledge base"""
        conn = sqlite3.connect(self.kb_path)
        cursor = conn.cursor()

        # Create facts table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT NOT NULL,
            attribute TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            created_at INTEGER NOT NULL,
            UNIQUE(entity, attribute)
        )
        """
        )

        # Create relationships table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT NOT NULL,
            relation TEXT NOT NULL,
            target TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            created_at INTEGER NOT NULL,
            UNIQUE(entity, relation, target)
        )
        """
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_attribute ON facts(attribute)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_entity ON relationships(entity)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_relation ON relationships(relation)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target)"
        )

        conn.commit()
        conn.close()

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": self._required_params,
        }

    async def _execute(
        self,
        operation: str,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
        value: Optional[str] = None,
        relation: Optional[str] = None,
        target: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the knowledge base tool

        Args:
            operation: Operation to perform
            entity: Entity name for fact or relationship
            attribute: Attribute name for fact
            value: Value for fact or relationship
            relation: Relation type for relationship
            target: Target entity for relationship
            query: Query string for searching

        Returns:
            Dictionary with operation result
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.kb_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            current_time = int(time.time())

            if operation == "add_fact":
                # Validate parameters
                if not entity or not attribute or value is None:
                    return {
                        "status": "error",
                        "error": "Missing required parameters for add_fact operation",
                        "required": ["entity", "attribute", "value"],
                    }

                # Add or update fact
                cursor.execute(
                    """
                INSERT INTO facts (entity, attribute, value, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entity, attribute) DO UPDATE SET
                    value = excluded.value,
                    created_at = excluded.created_at
                """,
                    (entity, attribute, value, current_time),
                )

                conn.commit()

                return {
                    "status": "success",
                    "operation": "add_fact",
                    "entity": entity,
                    "attribute": attribute,
                    "value": value,
                }

            elif operation == "query_facts":
                # Validate parameters
                if not entity and not attribute and not query:
                    return {
                        "status": "error",
                        "error": "Missing at least one search parameter for query_facts operation",
                        "required": ["entity or attribute or query"],
                    }

                results = []

                if query:
                    # Full-text search
                    search_term = f"%{query}%"
                    cursor.execute(
                        """
                    SELECT entity, attribute, value, created_at
                    FROM facts
                    WHERE entity LIKE ? OR attribute LIKE ? OR value LIKE ?
                    """,
                        (search_term, search_term, search_term),
                    )
                elif entity and attribute:
                    # Specific fact lookup
                    cursor.execute(
                        """
                    SELECT entity, attribute, value, created_at
                    FROM facts
                    WHERE entity = ? AND attribute = ?
                    """,
                        (entity, attribute),
                    )
                elif entity:
                    # All facts about entity
                    cursor.execute(
                        """
                    SELECT entity, attribute, value, created_at
                    FROM facts
                    WHERE entity = ?
                    """,
                        (entity,),
                    )
                elif attribute:
                    # All facts with attribute
                    cursor.execute(
                        """
                    SELECT entity, attribute, value, created_at
                    FROM facts
                    WHERE attribute = ?
                    """,
                        (attribute,),
                    )

                for row in cursor.fetchall():
                    results.append(
                        {
                            "entity": row["entity"],
                            "attribute": row["attribute"],
                            "value": row["value"],
                            "created_at": datetime.fromtimestamp(
                                row["created_at"]
                            ).isoformat(),
                        }
                    )

                return {
                    "status": "success",
                    "operation": "query_facts",
                    "count": len(results),
                    "results": results,
                }

            elif operation == "add_relationship":
                # Validate parameters
                if not entity or not relation or not target:
                    return {
                        "status": "error",
                        "error": "Missing required parameters for add_relationship operation",
                        "required": ["entity", "relation", "target"],
                    }

                # Add or update relationship
                cursor.execute(
                    """
                INSERT INTO relationships (entity, relation, target, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entity, relation, target) DO UPDATE SET
                    created_at = excluded.created_at
                """,
                    (entity, relation, target, current_time),
                )

                conn.commit()

                return {
                    "status": "success",
                    "operation": "add_relationship",
                    "entity": entity,
                    "relation": relation,
                    "target": target,
                }

            elif operation == "query_relationships":
                # Validate parameters
                if not entity and not relation and not target and not query:
                    return {
                        "status": "error",
                        "error": "Missing at least one search parameter for query_relationships operation",
                        "required": ["entity or relation or target or query"],
                    }

                results = []

                if query:
                    # Full-text search
                    search_term = f"%{query}%"
                    cursor.execute(
                        """
                    SELECT entity, relation, target, created_at
                    FROM relationships
                    WHERE entity LIKE ? OR relation LIKE ? OR target LIKE ?
                    """,
                        (search_term, search_term, search_term),
                    )
                elif entity and relation and target:
                    # Specific relationship lookup
                    cursor.execute(
                        """
                    SELECT entity, relation, target, created_at
                    FROM relationships
                    WHERE entity = ? AND relation = ? AND target = ?
                    """,
                        (entity, relation, target),
                    )
                elif entity and relation:
                    # All relationships from entity with relation
                    cursor.execute(
                        """
                    SELECT entity, relation, target, created_at
                    FROM relationships
                    WHERE entity = ? AND relation = ?
                    """,
                        (entity, relation),
                    )
                elif entity:
                    # All relationships from entity
                    cursor.execute(
                        """
                    SELECT entity, relation, target, created_at
                    FROM relationships
                    WHERE entity = ?
                    """,
                        (entity,),
                    )
                elif relation:
                    # All relationships with relation
                    cursor.execute(
                        """
                    SELECT entity, relation, target, created_at
                    FROM relationships
                    WHERE relation = ?
                    """,
                        (relation,),
                    )
                elif target:
                    # All relationships to target
                    cursor.execute(
                        """
                    SELECT entity, relation, target, created_at
                    FROM relationships
                    WHERE target = ?
                    """,
                        (target,),
                    )

                for row in cursor.fetchall():
                    results.append(
                        {
                            "entity": row["entity"],
                            "relation": row["relation"],
                            "target": row["target"],
                            "created_at": datetime.fromtimestamp(
                                row["created_at"]
                            ).isoformat(),
                        }
                    )

                return {
                    "status": "success",
                    "operation": "query_relationships",
                    "count": len(results),
                    "results": results,
                }

            else:
                return {"status": "error", "error": f"Invalid operation: {operation}"}
        except Exception as e:
            logger.error(f"Knowledge base error: {e}")
            return {
                "status": "error",
                "error": f"Knowledge base error: {str(e)}",
                "operation": operation,
            }
