"""WebRTC data channel wrapper."""

import asyncio
from typing import Optional, Callable
from aiortc import RTCPeerConnection, RTCDataChannel
from aiortc.contrib.signaling import object_from_string, object_to_string

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataChannelWrapper:
    """Wrapper for WebRTC data channel."""
    
    def __init__(
        self,
        peer_connection: RTCPeerConnection,
        channel_label: str = "stt-tts",
        ordered: bool = False,
        max_retransmits: Optional[int] = 0
    ):
        """Initialize data channel wrapper.
        
        Args:
            peer_connection: RTCPeerConnection instance
            channel_label: Data channel label
            ordered: Whether messages should be delivered in order
            max_retransmits: Maximum number of retransmissions
        """
        self.pc = peer_connection
        self.channel_label = channel_label
        self.ordered = ordered
        self.max_retransmits = max_retransmits
        
        self.channel: Optional[RTCDataChannel] = None
        self.on_message_callback: Optional[Callable] = None
        self.on_open_callback: Optional[Callable] = None
        self.on_close_callback: Optional[Callable] = None
        
        logger.info(f"DataChannel wrapper created: {channel_label}")
    
    def create_channel(self) -> RTCDataChannel:
        """Create a data channel (caller side).
        
        Returns:
            RTCDataChannel instance
        """
        self.channel = self.pc.createDataChannel(
            self.channel_label,
            ordered=self.ordered,
            maxRetransmits=self.max_retransmits
        )
        
        # Set up event handlers
        self._setup_handlers()
        
        logger.info(f"Data channel created: {self.channel_label}")
        return self.channel
    
    def set_channel(self, channel: RTCDataChannel) -> None:
        """Set the data channel (answerer side).
        
        Args:
            channel: Existing RTCDataChannel
        """
        self.channel = channel
        self._setup_handlers()
        logger.info(f"Data channel set: {self.channel_label}")
    
    def _setup_handlers(self) -> None:
        """Set up event handlers for the channel."""
        if not self.channel:
            return
        
        @self.channel.on("open")
        def on_open():
            logger.info(f"Data channel opened: {self.channel_label}")
            if self.on_open_callback:
                asyncio.create_task(self._call_async(self.on_open_callback))
        
        @self.channel.on("message")
        def on_message(message):
            logger.debug(f"Received message: {len(message) if isinstance(message, bytes) else 'text'} bytes")
            if self.on_message_callback:
                asyncio.create_task(self._call_async(self.on_message_callback, message))
        
        @self.channel.on("close")
        def on_close():
            logger.info(f"Data channel closed: {self.channel_label}")
            if self.on_close_callback:
                asyncio.create_task(self._call_async(self.on_close_callback))
    
    async def _call_async(self, callback: Callable, *args) -> None:
        """Call a callback asynchronously.
        
        Args:
            callback: Callback function
            *args: Arguments to pass
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
    
    def send(self, data: bytes) -> None:
        """Send data through the channel.
        
        Args:
            data: Data to send
        """
        if not self.channel or self.channel.readyState != "open":
            logger.warning("Cannot send: channel not open")
            return
        
        try:
            self.channel.send(data)
            logger.debug(f"Sent {len(data)} bytes")
        except Exception as e:
            logger.error(f"Error sending data: {e}")
    
    def on_message(self, callback: Callable) -> None:
        """Set message callback.
        
        Args:
            callback: Function to call when message received
        """
        self.on_message_callback = callback
    
    def on_open(self, callback: Callable) -> None:
        """Set open callback.
        
        Args:
            callback: Function to call when channel opens
        """
        self.on_open_callback = callback
    
    def on_close(self, callback: Callable) -> None:
        """Set close callback.
        
        Args:
            callback: Function to call when channel closes
        """
        self.on_close_callback = callback
    
    def is_open(self) -> bool:
        """Check if channel is open.
        
        Returns:
            True if open, False otherwise
        """
        return self.channel is not None and self.channel.readyState == "open"
    
    def close(self) -> None:
        """Close the channel."""
        if self.channel:
            self.channel.close()
            logger.info(f"Data channel closed: {self.channel_label}")


class WebRTCConnection:
    """Manage WebRTC peer connection."""
    
    def __init__(
        self,
        stun_server: str = "stun:stun.l.google.com:19302",
        turn_server: Optional[str] = None
    ):
        """Initialize WebRTC connection.
        
        Args:
            stun_server: STUN server URL
            turn_server: Optional TURN server URL
        """
        self.stun_server = stun_server
        self.turn_server = turn_server
        
        # Create peer connection
        from aiortc import RTCConfiguration, RTCIceServer
        
        config = RTCConfiguration(
            iceServers=[RTCIceServer(urls=[stun_server])]
        )
        
        if turn_server:
            config.iceServers.append(RTCIceServer(urls=[turn_server]))
        
        self.pc = RTCPeerConnection(configuration=config)
        self.data_channel: Optional[DataChannelWrapper] = None
        
        logger.info("WebRTC connection initialized")
    
    async def create_offer(self) -> str:
        """Create SDP offer.
        
        Returns:
            SDP offer as JSON string
        """
        # Create data channel
        self.data_channel = DataChannelWrapper(self.pc)
        self.data_channel.create_channel()
        
        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        
        # Wait for ICE gathering
        await self._wait_for_ice_gathering()
        
        # Return as JSON
        return object_to_string(self.pc.localDescription)
    
    async def create_answer(self, offer_json: str) -> str:
        """Create SDP answer.
        
        Args:
            offer_json: SDP offer as JSON string
        
        Returns:
            SDP answer as JSON string
        """
        # Set remote description
        offer = object_from_string(offer_json)
        await self.pc.setRemoteDescription(offer)
        
        # Set up data channel handler
        @self.pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"Data channel received: {channel.label}")
            self.data_channel = DataChannelWrapper(self.pc)
            self.data_channel.set_channel(channel)
        
        # Create answer
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        # Wait for ICE gathering
        await self._wait_for_ice_gathering()
        
        # Return as JSON
        return object_to_string(self.pc.localDescription)
    
    async def set_remote_answer(self, answer_json: str) -> None:
        """Set remote SDP answer.
        
        Args:
            answer_json: SDP answer as JSON string
        """
        answer = object_from_string(answer_json)
        await self.pc.setRemoteDescription(answer)
        logger.info("Remote answer set")
    
    async def _wait_for_ice_gathering(self) -> None:
        """Wait for ICE gathering to complete."""
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)
        logger.debug("ICE gathering complete")
    
    def get_data_channel(self) -> Optional[DataChannelWrapper]:
        """Get the data channel wrapper.
        
        Returns:
            DataChannelWrapper or None
        """
        return self.data_channel
    
    async def close(self) -> None:
        """Close the connection."""
        if self.data_channel:
            self.data_channel.close()
        await self.pc.close()
        logger.info("WebRTC connection closed")
