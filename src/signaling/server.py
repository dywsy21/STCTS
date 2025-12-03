"""WebRTC signaling server."""

import asyncio
import json
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import websockets
    from websockets.legacy.server import WebSocketServerProtocol
except ImportError:
    websockets = None
    WebSocketServerProtocol = Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Peer:
    """Peer connection info."""
    peer_id: str
    websocket: Any  # WebSocketServerProtocol
    room_id: Optional[str] = None
    connected_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.connected_at is None:
            self.connected_at = datetime.now()


class SignalingServer:
    """WebRTC signaling server."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """Initialize signaling server.
        
        Args:
            host: Server host
            port: Server port
        """
        if websockets is None:
            raise ImportError("websockets package required. Install with: uv pip install websockets")
        
        self.host = host
        self.port = port
        
        # Track peers and rooms
        self.peers: Dict[str, Peer] = {}
        self.rooms: Dict[str, Set[str]] = {}  # room_id -> set of peer_ids
        
        logger.info(f"Signaling server initialized on {host}:{port}")
    
    async def start(self):
        """Start the signaling server."""
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"Signaling server listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever
    
    async def handle_client(self, websocket: Any):
        """Handle client connection.
        
        Args:
            websocket: WebSocket connection
        """
        peer_id: Optional[str] = None
        
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "register":
                    peer_id = data.get("peer_id")
                    if peer_id:
                        await self._handle_register(peer_id, websocket)
                
                elif msg_type == "join_room":
                    room_id = data.get("room_id")
                    if peer_id and room_id:
                        await self._handle_join_room(peer_id, room_id)
                
                elif msg_type == "leave_room":
                    if peer_id:
                        await self._handle_leave_room(peer_id)
                
                elif msg_type == "offer":
                    target_id = data.get("target")
                    if peer_id and target_id:
                        await self._handle_offer(peer_id, target_id, data.get("sdp"))
                
                elif msg_type == "answer":
                    target_id = data.get("target")
                    if peer_id and target_id:
                        await self._handle_answer(peer_id, target_id, data.get("sdp"))
                
                elif msg_type == "ice_candidate":
                    target_id = data.get("target")
                    if peer_id and target_id:
                        await self._handle_ice_candidate(peer_id, target_id, data.get("candidate"))
                
                elif msg_type == "list_peers":
                    if peer_id:
                        await self._handle_list_peers(peer_id)
                
                else:
                    logger.warning(f"Unknown message type: {msg_type}")
        
        except Exception as e:
            if "ConnectionClosed" in str(type(e)):
                logger.info(f"Connection closed for peer {peer_id}")
            else:
                logger.error(f"Error handling client: {e}")
        finally:
            if peer_id:
                await self._handle_disconnect(peer_id)
    
    async def _handle_register(self, peer_id: str, websocket: Any):
        """Handle peer registration.
        
        Args:
            peer_id: Peer identifier
            websocket: WebSocket connection
        """
        if peer_id in self.peers:
            logger.warning(f"Peer {peer_id} already registered, replacing")
            await self._handle_disconnect(peer_id)
        
        self.peers[peer_id] = Peer(peer_id=peer_id, websocket=websocket)
        
        await websocket.send(json.dumps({
            "type": "registered",
            "peer_id": peer_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        logger.info(f"Peer registered: {peer_id} (total peers: {len(self.peers)})")
        
        # Notify all peers about the new peer
        await self._broadcast_peer_joined(peer_id)
    
    async def _handle_join_room(self, peer_id: str, room_id: str):
        """Handle peer joining a room.
        
        Args:
            peer_id: Peer identifier
            room_id: Room identifier
        """
        if peer_id not in self.peers:
            logger.error(f"Peer {peer_id} not registered")
            return
        
        peer = self.peers[peer_id]
        
        # Leave current room if any
        if peer.room_id:
            await self._handle_leave_room(peer_id)
        
        # Join new room
        peer.room_id = room_id
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        self.rooms[room_id].add(peer_id)
        
        # Notify peer
        await peer.websocket.send(json.dumps({
            "type": "joined_room",
            "room_id": room_id,
            "peers": list(self.rooms[room_id] - {peer_id})
        }))
        
        # Notify other peers in room
        await self._broadcast_to_room(
            room_id,
            {
                "type": "peer_joined",
                "peer_id": peer_id
            },
            exclude_peer=peer_id
        )
        
        logger.info(f"Peer {peer_id} joined room {room_id}")
    
    async def _handle_leave_room(self, peer_id: str):
        """Handle peer leaving a room.
        
        Args:
            peer_id: Peer identifier
        """
        if peer_id not in self.peers:
            return
        
        peer = self.peers[peer_id]
        if not peer.room_id:
            return
        
        room_id = peer.room_id
        
        # Remove from room
        if room_id in self.rooms:
            self.rooms[room_id].discard(peer_id)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
        
        # Notify other peers
        await self._broadcast_to_room(
            room_id,
            {
                "type": "peer_left",
                "peer_id": peer_id
            },
            exclude_peer=peer_id
        )
        
        peer.room_id = None
        logger.info(f"Peer {peer_id} left room {room_id}")
    
    async def _handle_offer(self, peer_id: str, target_id: str, sdp: dict):
        """Handle WebRTC offer.
        
        Args:
            peer_id: Sender peer ID
            target_id: Target peer ID
            sdp: SDP offer
        """
        if target_id not in self.peers:
            logger.error(f"Target peer {target_id} not found")
            return
        
        target_peer = self.peers[target_id]
        await target_peer.websocket.send(json.dumps({
            "type": "offer",
            "from": peer_id,
            "sdp": sdp
        }))
        
        logger.debug(f"Forwarded offer from {peer_id} to {target_id}")
    
    async def _handle_answer(self, peer_id: str, target_id: str, sdp: dict):
        """Handle WebRTC answer.
        
        Args:
            peer_id: Sender peer ID
            target_id: Target peer ID
            sdp: SDP answer
        """
        if target_id not in self.peers:
            logger.error(f"Target peer {target_id} not found")
            return
        
        target_peer = self.peers[target_id]
        await target_peer.websocket.send(json.dumps({
            "type": "answer",
            "from": peer_id,
            "sdp": sdp
        }))
        
        logger.debug(f"Forwarded answer from {peer_id} to {target_id}")
    
    async def _handle_ice_candidate(self, peer_id: str, target_id: str, candidate: dict):
        """Handle ICE candidate.
        
        Args:
            peer_id: Sender peer ID
            target_id: Target peer ID
            candidate: ICE candidate
        """
        if target_id not in self.peers:
            logger.error(f"Target peer {target_id} not found")
            return
        
        target_peer = self.peers[target_id]
        await target_peer.websocket.send(json.dumps({
            "type": "ice_candidate",
            "from": peer_id,
            "candidate": candidate
        }))
        
        logger.debug(f"Forwarded ICE candidate from {peer_id} to {target_id}")
    
    async def _handle_list_peers(self, peer_id: str):
        """Handle list peers request.
        
        Args:
            peer_id: Requesting peer ID
        """
        if peer_id not in self.peers:
            return
        
        peer = self.peers[peer_id]
        
        # Get peers in same room (if in a room)
        if peer.room_id and peer.room_id in self.rooms:
            # Return peers in the same room
            room_peers = list(self.rooms[peer.room_id] - {peer_id})
        else:
            # If not in a room, return all registered peers except self
            room_peers = [p for p in self.peers.keys() if p != peer_id]
        
        await peer.websocket.send(json.dumps({
            "type": "peer_list",
            "peers": room_peers
        }))
        
        logger.debug(f"Sent peer list to {peer_id}: {room_peers}")
    
    async def _handle_disconnect(self, peer_id: str):
        """Handle peer disconnection.
        
        Args:
            peer_id: Disconnecting peer ID
        """
        if peer_id not in self.peers:
            return
        
        # Leave room if in one
        await self._handle_leave_room(peer_id)
        
        # Remove peer
        del self.peers[peer_id]
        
        logger.info(f"Peer disconnected: {peer_id} (remaining peers: {len(self.peers)})")
        
        # Notify other peers about disconnection
        await self._broadcast_peer_left(peer_id)
    
    async def _broadcast_peer_joined(self, peer_id: str):
        """Broadcast to all peers that a new peer joined.
        
        Args:
            peer_id: Peer that just joined
        """
        message = json.dumps({
            "type": "peer_connected",
            "peer_id": peer_id
        })
        
        for other_peer_id, peer in self.peers.items():
            if other_peer_id != peer_id:
                try:
                    await peer.websocket.send(message)
                except Exception as e:
                    logger.error(f"Error notifying {other_peer_id} about new peer: {e}")
    
    async def _broadcast_peer_left(self, peer_id: str):
        """Broadcast to all peers that a peer disconnected.
        
        Args:
            peer_id: Peer that just left
        """
        message = json.dumps({
            "type": "peer_disconnected",
            "peer_id": peer_id
        })
        
        for other_peer_id, peer in self.peers.items():
            try:
                await peer.websocket.send(message)
            except Exception as e:
                logger.error(f"Error notifying {other_peer_id} about peer leaving: {e}")
    
    async def _broadcast_to_room(
        self,
        room_id: str,
        message: dict,
        exclude_peer: Optional[str] = None
    ):
        """Broadcast message to all peers in a room.
        
        Args:
            room_id: Room identifier
            message: Message to broadcast
            exclude_peer: Optional peer ID to exclude
        """
        if room_id not in self.rooms:
            return
        
        message_json = json.dumps(message)
        
        for peer_id in self.rooms[room_id]:
            if peer_id == exclude_peer:
                continue
            
            if peer_id in self.peers:
                try:
                    await self.peers[peer_id].websocket.send(message_json)
                except Exception as e:
                    logger.error(f"Error broadcasting to {peer_id}: {e}")


async def main():
    """Run the signaling server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebRTC Signaling Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    
    args = parser.parse_args()
    
    server = SignalingServer(host=args.host, port=args.port)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
