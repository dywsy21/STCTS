"""WebRTC signaling client."""

import asyncio
import json
from typing import Optional, Callable, Any
from dataclasses import dataclass

try:
    import websockets
except ImportError:
    websockets = None

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SignalingMessage:
    """Signaling message."""
    type: str
    data: dict


class SignalingClient:
    """WebRTC signaling client."""
    
    def __init__(self, peer_id: str, signaling_url: str):
        """Initialize signaling client.
        
        Args:
            peer_id: This peer's identifier
            signaling_url: Signaling server URL (e.g., ws://localhost:8765)
        """
        if websockets is None:
            raise ImportError("websockets package required. Install with: uv pip install websockets")
        
        self.peer_id = peer_id
        self.signaling_url = signaling_url
        self.websocket: Optional[Any] = None
        self.is_connected = False
        
        # Callbacks
        self.on_offer: Optional[Callable] = None
        self.on_answer: Optional[Callable] = None
        self.on_ice_candidate: Optional[Callable] = None
        self.on_peer_list: Optional[Callable] = None
        self.on_peer_connected: Optional[Callable] = None
        self.on_peer_disconnected: Optional[Callable] = None
        
        logger.info(f"Signaling client initialized for peer {peer_id}")
    
    async def connect(self) -> bool:
        """Connect to signaling server.
        
        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to signaling server: {self.signaling_url}")
            self.websocket = await websockets.connect(self.signaling_url)
            self.is_connected = True  # Set this BEFORE sending register message
            
            # Register with server
            await self.send_message({
                "type": "register",
                "peer_id": self.peer_id
            })
            
            # Wait for registration confirmation
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "registered":
                logger.info(f"Registered with signaling server as {self.peer_id}")
                
                # Start receiving messages
                asyncio.create_task(self._receive_loop())
                
                return True
            else:
                logger.error(f"Registration failed: {data}")
                self.is_connected = False
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to signaling server: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from signaling server."""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from signaling server")
    
    async def send_offer(self, target_peer_id: str, sdp: str) -> None:
        """Send WebRTC offer to peer.
        
        Args:
            target_peer_id: Target peer identifier
            sdp: Session Description Protocol offer
        """
        await self.send_message({
            "type": "offer",
            "target": target_peer_id,
            "sdp": sdp
        })
        logger.info(f"Sent offer to {target_peer_id}")
    
    async def send_answer(self, target_peer_id: str, sdp: str) -> None:
        """Send WebRTC answer to peer.
        
        Args:
            target_peer_id: Target peer identifier
            sdp: Session Description Protocol answer
        """
        await self.send_message({
            "type": "answer",
            "target": target_peer_id,
            "sdp": sdp
        })
        logger.info(f"Sent answer to {target_peer_id}")
    
    async def send_ice_candidate(self, target_peer_id: str, candidate: dict) -> None:
        """Send ICE candidate to peer.
        
        Args:
            target_peer_id: Target peer identifier
            candidate: ICE candidate
        """
        await self.send_message({
            "type": "ice_candidate",
            "target": target_peer_id,
            "candidate": candidate
        })
        logger.debug(f"Sent ICE candidate to {target_peer_id}")
    
    async def list_peers(self) -> None:
        """Request list of available peers."""
        await self.send_message({
            "type": "list_peers"
        })
        logger.debug("Requested peer list")
    
    async def send_message(self, message: dict) -> None:
        """Send message to signaling server.
        
        Args:
            message: Message dictionary
        """
        if not self.websocket or not self.is_connected:
            logger.error("Not connected to signaling server")
            return
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def _receive_loop(self) -> None:
        """Receive messages from signaling server."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        except Exception as e:
            if "ConnectionClosed" not in str(type(e)):
                logger.error(f"Receive loop error: {e}")
        finally:
            self.is_connected = False
            logger.info("Signaling connection closed")
    
    async def _handle_message(self, data: dict) -> None:
        """Handle received message.
        
        Args:
            data: Message data
        """
        msg_type = data.get("type")
        
        if msg_type == "offer":
            sender = data.get("sender")
            sdp = data.get("sdp")
            if self.on_offer and sender and sdp:
                await self.on_offer(sender, sdp)
        
        elif msg_type == "answer":
            sender = data.get("sender")
            sdp = data.get("sdp")
            if self.on_answer and sender and sdp:
                await self.on_answer(sender, sdp)
        
        elif msg_type == "ice_candidate":
            sender = data.get("sender")
            candidate = data.get("candidate")
            if self.on_ice_candidate and sender and candidate:
                await self.on_ice_candidate(sender, candidate)
        
        elif msg_type == "peer_list":
            peers = data.get("peers", [])
            if self.on_peer_list:
                await self.on_peer_list(peers)
        
        elif msg_type == "peer_connected":
            peer_id = data.get("peer_id")
            if self.on_peer_connected and peer_id:
                await self.on_peer_connected(peer_id)
        
        elif msg_type == "peer_disconnected":
            peer_id = data.get("peer_id")
            if self.on_peer_disconnected and peer_id:
                await self.on_peer_disconnected(peer_id)
        
        else:
            logger.debug(f"Unknown message type: {msg_type}")
