"""
WebSocket Manager — handles real-time connections to the dashboard.
"""

import asyncio
import json
from typing import Dict, List

from fastapi import WebSocket


class ConnectionManager:
    """
    Manages active WebSocket connections and broadcasts messages to all clients.
    """

    def __init__(self):
        # Active WebSocket connections
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection and assign it a unique ID.
        Returns the connection ID.
        """
        await websocket.accept()
        # Generate a simple unique ID based on the websocket's id property
        conn_id = id(websocket)
        self.active[str(conn_id)] = websocket
        return str(conn_id)

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection from active connections.
        """
        conn_id = str(id(websocket))
        if conn_id in self.active:
            del self.active[conn_id]

    async def send_personal(self, message: dict, websocket: WebSocket) -> None:
        """
        Send a JSON message to a specific client.
        """
        try:
            await websocket.send_json(message)
        except Exception:
            # Client may have disconnected
            self.disconnect(websocket)

    async def broadcast(self, message: dict) -> None:
        """
        Broadcast a JSON message to all connected clients.
        """
        if not self.active:
            return

        # Create a list of IDs to avoid "dictionary changed size during iteration"
        # Also handle disconnected clients
        disconnected = []

        for conn_id, websocket in list(self.active.items()):
            try:
                await websocket.send_json(message)
            except Exception:
                # Mark for removal
                disconnected.append(conn_id)

        # Clean up disconnected clients
        for conn_id in disconnected:
            if conn_id in self.active:
                del self.active[conn_id]

    async def broadcast_text(self, message: str) -> None:
        """
        Broadcast a text message to all connected clients.
        """
        if not self.active:
            return

        disconnected = []

        for conn_id, websocket in list(self.active.items()):
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(conn_id)

        for conn_id in disconnected:
            if conn_id in self.active:
                del self.active[conn_id]

