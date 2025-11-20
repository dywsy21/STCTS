"""Voice client for STT-Compress-TTS communication."""

import asyncio
from typing import Optional
import numpy as np

from src.signaling.client import SignalingClient
from src.pipeline.sender import SenderPipeline
from src.pipeline.receiver import ReceiverPipeline
from src.network import PacketPriorityQueue
from src.network.packet import Packet
from src.network.channel import WebRTCConnection
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


class VoiceClient:
    """Voice communication client."""
    
    def __init__(
        self,
        peer_id: str,
        signaling_url: str,
        quality_mode: str = "balanced",
        config_path: Optional[str] = None,
        auto_connect_peer: Optional[str] = None,
        output_audio_only: bool = False
    ):
        """Initialize voice client.
        
        Args:
            peer_id: This peer's identifier
            signaling_url: Signaling server URL
            quality_mode: Quality mode (minimal, balanced, high)
            config_path: Path to config file (overrides quality_mode if provided)
            auto_connect_peer: Peer ID to automatically connect to
            output_audio_only: If True, only receive and play audio (don't send)
        """
        self.peer_id = peer_id
        self.signaling_url = signaling_url
        self.quality_mode = quality_mode
        self.config_path = config_path
        self.auto_connect_peer = auto_connect_peer
        self.output_audio_only = output_audio_only
        
        # Load configuration - prefer config_path over quality_mode
        if config_path:
            self.config = load_config(config_path=config_path)
        else:
            self.config = load_config(quality_mode=quality_mode)
        
        # Initialize signaling
        self.signaling = SignalingClient(peer_id, signaling_url)
        
        # Set up signaling callbacks
        self.signaling.on_offer = self._handle_offer
        self.signaling.on_answer = self._handle_answer
        self.signaling.on_ice_candidate = self._handle_ice_candidate
        self.signaling.on_peer_list = self._handle_peer_list
        self.signaling.on_peer_connected = self._handle_peer_connected
        self.signaling.on_peer_disconnected = self._handle_peer_disconnected
        
        # Initialize pipelines
        self.packet_queue = PacketPriorityQueue()
        
        # Only initialize sender if not in output-audio-only mode
        if not output_audio_only:
            self.sender = SenderPipeline(
                config=self.config.model_dump(),
                packet_queue=self.packet_queue,
                sample_rate=16000,
                is_transmitting_callback=lambda: self.is_transmitting
            )
        else:
            self.sender = None
            logger.info("Sender pipeline disabled (output-audio-only mode)")
        
        # Always initialize receiver
        self.receiver = ReceiverPipeline(
            config=self.config.model_dump(),
            sample_rate=16000
        )
        
        # State
        self.is_running = False
        self.is_transmitting = False  # For push-to-talk
        self.connected_peers = {}  # Maps peer_id -> {"connection": WebRTCConnection, "connected_at": timestamp}
        self.webrtc_connections = {}  # Maps peer_id -> WebRTCConnection
        
        # Packet statistics
        self._packet_stats = {
            "TEXT": 0,
            "PROSODY": 0,
            "TIMBRE": 0
        }
        
        mode_info = " (receiver-only)" if output_audio_only else ""
        logger.info(f"Voice client initialized: peer_id={peer_id}, mode={quality_mode}{mode_info}")
    
    async def start(self) -> None:
        """Start the voice client."""
        if self.is_running:
            logger.warning("Voice client already running")
            return
        
        logger.info("Starting voice client...")
        
        # Connect to signaling server
        if not await self.signaling.connect():
            logger.error("Failed to connect to signaling server")
            return
        
        # Start pipelines
        if self.sender:
            await self.sender.start()
        await self.receiver.start()
        
        self.is_running = True
        
        # Start packet transmission loop
        if self.sender:
            asyncio.create_task(self._packet_transmission_loop())
        
        # Request peer list
        await self.signaling.list_peers()
        
        # Auto-connect if specified
        if self.auto_connect_peer:
            logger.info(f"Auto-connecting to peer: {self.auto_connect_peer}")
            await self.connect_to_peer(self.auto_connect_peer)
        
        logger.info("Voice client started successfully")
    
    async def stop(self) -> None:
        """Stop the voice client."""
        if not self.is_running:
            return
        
        logger.info("Stopping voice client...")
        
        self.is_running = False
        self.is_transmitting = False
        
        # Show packet statistics
        if hasattr(self, '_packet_stats'):
            total = sum(self._packet_stats.values())
            logger.info(
                f"📊 Packet statistics - "
                f"TEXT: {self._packet_stats.get('TEXT', 0)}, "
                f"PROSODY: {self._packet_stats.get('PROSODY', 0)}, "
                f"TIMBRE: {self._packet_stats.get('TIMBRE', 0)}, "
                f"Total: {total}"
            )
        
        # Close all WebRTC connections
        for peer_id, webrtc_conn in self.webrtc_connections.items():
            logger.info(f"Closing WebRTC connection to {peer_id}")
            await webrtc_conn.close()
        
        # Disconnect from signaling server
        await self.signaling.disconnect()
        
        logger.info("Voice client stopped")
    
    async def _packet_transmission_loop(self) -> None:
        """Continuously send packets from the queue."""
        logger.info("Packet transmission loop started")
        
        while self.is_running:
            try:
                # Get packet from queue (with timeout)
                packet = await self.packet_queue.get(timeout=0.1)
                
                if packet is None:
                    continue
                
                # Serialize packet
                packet_bytes = packet.to_bytes()
                
                # Send packet to all connected peers via WebRTC data channel
                sent_count = 0
                for peer_id, webrtc_conn in self.webrtc_connections.items():
                    data_channel = webrtc_conn.get_data_channel()
                    if data_channel and data_channel.is_open():
                        data_channel.send(packet_bytes)
                        sent_count += 1
                
                if sent_count > 0:
                    logger.info(
                        f"📤 Sent {packet.packet_type.name} packet to {sent_count} peer(s): "
                        f"seq={packet.sequence_number}, size={len(packet_bytes)} bytes"
                    )
                else:
                    logger.warning(
                        f"⚠️ No open data channels to send {packet.packet_type.name} packet "
                        f"(connected_peers={len(self.connected_peers)})"
                    )
                
                # Track packet statistics
                packet_type_name = packet.packet_type.name
                if packet_type_name in self._packet_stats:
                    self._packet_stats[packet_type_name] += 1
                
            except Exception as e:
                logger.error(f"Error in packet transmission loop: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("Packet transmission loop stopped")
    
    async def connect_to_peer(self, target_peer_id: str) -> None:
        """Connect to another peer.
        
        Args:
            target_peer_id: Peer ID to connect to
        """
        logger.info(f"Initiating connection to peer: {target_peer_id}")
        
        # Create WebRTC connection
        webrtc_conn = WebRTCConnection()
        self.webrtc_connections[target_peer_id] = webrtc_conn
        
        # Create offer
        sdp_offer = await webrtc_conn.create_offer()
        
        # Set up data channel callbacks
        data_channel = webrtc_conn.get_data_channel()
        if data_channel:
            data_channel.on_message(lambda msg: asyncio.create_task(self._handle_data_channel_message(target_peer_id, msg)))
            data_channel.on_open(lambda: logger.info(f"✅ Data channel opened with {target_peer_id}"))
            data_channel.on_close(lambda: logger.warning(f"❌ Data channel closed with {target_peer_id}"))
        
        # Send offer via signaling
        await self.signaling.send_offer(target_peer_id, sdp_offer)
        
        logger.info(f"Sent WebRTC offer to {target_peer_id}")
    
    def start_transmission(self) -> None:
        """Start transmitting audio (push-to-talk pressed)."""
        if self.output_audio_only:
            logger.warning("Cannot transmit in output-audio-only mode")
            return
        
        if not self.is_running:
            logger.warning("Cannot start transmission: client not running")
            return
        
        if self.is_transmitting:
            logger.debug("Already transmitting")
            return
        
        self.is_transmitting = True
        logger.info("🎤 Transmission started (microphone open)")
    
    def stop_transmission(self) -> None:
        """Stop transmitting audio (push-to-talk released)."""
        if self.output_audio_only:
            return  # No-op in output-audio-only mode
        
        if not self.is_transmitting:
            return
        
        self.is_transmitting = False
        logger.info("🔇 Transmission stopped (microphone closed)")
    
    async def _handle_offer(self, sender_peer_id: str, sdp: str) -> None:
        """Handle incoming WebRTC offer.
        
        Args:
            sender_peer_id: Peer who sent the offer
            sdp: Session Description Protocol offer
        """
        logger.info(f"Received WebRTC offer from {sender_peer_id}")
        
        # Create WebRTC connection
        webrtc_conn = WebRTCConnection()
        self.webrtc_connections[sender_peer_id] = webrtc_conn
        
        # Create answer
        sdp_answer = await webrtc_conn.create_answer(sdp)
        
        # Set up data channel callbacks (will be triggered when channel is received)
        async def setup_channel():
            await asyncio.sleep(0.1)  # Give time for channel to be created
            data_channel = webrtc_conn.get_data_channel()
            if data_channel:
                data_channel.on_message(lambda msg: asyncio.create_task(self._handle_data_channel_message(sender_peer_id, msg)))
                data_channel.on_open(lambda: self._on_peer_connected(sender_peer_id))
                data_channel.on_close(lambda: logger.warning(f"❌ Data channel closed with {sender_peer_id}"))
        
        asyncio.create_task(setup_channel())
        
        # Send answer via signaling
        await self.signaling.send_answer(sender_peer_id, sdp_answer)
        
        logger.info(f"Sent WebRTC answer to {sender_peer_id}")
    
    async def _handle_answer(self, sender_peer_id: str, sdp: str) -> None:
        """Handle incoming WebRTC answer.
        
        Args:
            sender_peer_id: Peer who sent the answer
            sdp: Session Description Protocol answer
        """
        logger.info(f"Received WebRTC answer from {sender_peer_id}")
        
        # Get the WebRTC connection we created when sending the offer
        webrtc_conn = self.webrtc_connections.get(sender_peer_id)
        if not webrtc_conn:
            logger.error(f"No WebRTC connection found for {sender_peer_id}")
            return
        
        # Set remote answer
        await webrtc_conn.set_remote_answer(sdp)
        
        # Mark peer as connected
        self.connected_peers[sender_peer_id] = {
            "connected_at": asyncio.get_event_loop().time()
        }
        
        logger.info(f"✅ WebRTC connection established with {sender_peer_id}")
    
    async def _handle_ice_candidate(self, sender_peer_id: str, candidate: dict) -> None:
        """Handle incoming ICE candidate.
        
        Args:
            sender_peer_id: Peer who sent the candidate
            candidate: ICE candidate data
        """
        logger.debug(f"Received ICE candidate from {sender_peer_id}")
        
        # TODO: Add ICE candidate to WebRTC connection
        # Note: aiortc handles ICE candidates automatically via Trickle ICE
    
    async def _handle_data_channel_message(self, sender_peer_id: str, message: bytes) -> None:
        """Handle incoming data channel message.
        
        Args:
            sender_peer_id: Peer who sent the message
            message: Binary message data
        """
        try:
            # Deserialize the packet
            packet = Packet.from_bytes(message)
            
            # Pass the packet to the receiver pipeline
            await self.receiver.receive_packet(packet)
            logger.debug(f"Processed {packet.packet_type.name} packet from {sender_peer_id}: {len(message)} bytes")
        except Exception as e:
            logger.error(f"Error processing packet from {sender_peer_id}: {e}")
    
    def _on_peer_connected(self, peer_id: str) -> None:
        """Called when data channel opens with a peer.
        
        Args:
            peer_id: Peer that connected
        """
        self.connected_peers[peer_id] = {
            "connected_at": asyncio.get_event_loop().time()
        }
        logger.info(f"✅ Data channel opened with {peer_id}")
        print(f"\n✅ Connected to {peer_id}")
    
    async def _handle_peer_list(self, peers: list) -> None:
        """Handle peer list update.
        
        Args:
            peers: List of available peer IDs
        """
        # Filter out self
        other_peers = [p for p in peers if p != self.peer_id]
        
        logger.info(f"Available peers: {other_peers}")
        
        if other_peers:
            print(f"\n📡 Available peers: {', '.join(other_peers)}")
            print("Use the 'connect <peer_id>' command to connect\n")
    
    async def _handle_peer_connected(self, peer_id: str) -> None:
        """Handle new peer connection notification.
        
        Args:
            peer_id: Peer that just connected
        """
        logger.info(f"✅ New peer connected: {peer_id}")
        print(f"\n✅ Peer '{peer_id}' is now online")
        
        # Request updated peer list
        await self.signaling.list_peers()
    
    async def _handle_peer_disconnected(self, peer_id: str) -> None:
        """Handle peer disconnection notification.
        
        Args:
            peer_id: Peer that disconnected
        """
        logger.info(f"❌ Peer disconnected: {peer_id}")
        print(f"\n❌ Peer '{peer_id}' went offline")
        
        # Remove from connected peers if present
        if peer_id in self.connected_peers:
            del self.connected_peers[peer_id]
        
        # Request updated peer list
        await self.signaling.list_peers()
    
    async def run_interactive(self) -> None:
        """Run interactive CLI session."""
        print("=" * 60)
        print("🎙️  STT-Compress-TTS Voice Client")
        print(f"Peer ID: {self.peer_id}")
        print(f"Quality Mode: {self.quality_mode}")
        print(f"Signaling Server: {self.signaling_url}")
        if self.output_audio_only:
            print("Mode: 🔊 RECEIVER ONLY (output-audio-only)")
        else:
            print("Mode: 🔄 Full-Duplex (send + receive)")
        print("=" * 60)
        print("\nCommands:")
        print("  connect <peer_id>  - Connect to a peer")
        print("  peers              - List available peers")
        if not self.output_audio_only:
            print("  start              - Start transmitting (push-to-talk)")
            print("  stop               - Stop transmitting")
        print("  status             - Show client status")
        print("  help               - Show help message")
        print("  quit               - Exit the application")
        print("=" * 60)
        print()
        
        await self.start()
        
        # Run command loop using aioconsole for non-blocking input
        import aioconsole
        
        try:
            while self.is_running:
                try:
                    # Use aioconsole for async input (doesn't block event loop)
                    command = await aioconsole.ainput(f"[{self.peer_id}]> ")
                    command = command.strip()
                    
                    if not command:
                        continue
                    
                    # Process command
                    await self._process_command(command)
                    
                except (KeyboardInterrupt, EOFError):
                    print("\n\nShutting down...")
                    break
                except Exception as e:
                    logger.error(f"Error in command loop: {e}")
                    await asyncio.sleep(0.1)
        finally:
            await self.stop()
    
    async def _process_command(self, command: str) -> None:
        """Process user command.
        
        Args:
            command: User command string
        """
        parts = command.split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        
        if cmd == "quit" or cmd == "exit" or cmd == "q":
            print("Shutting down...")
            self.is_running = False
        
        elif cmd == "peers":
            await self.signaling.list_peers()
        
        elif cmd == "connect":
            if len(parts) < 2:
                print("Usage: connect <peer_id>")
            else:
                target_peer = parts[1]
                await self.connect_to_peer(target_peer)
        
        elif cmd == "start":
            if self.output_audio_only:
                print("❌ Cannot transmit in output-audio-only mode")
            else:
                self.start_transmission()
        
        elif cmd == "stop":
            if not self.output_audio_only:
                self.stop_transmission()
        
        elif cmd == "status":
            status = self.get_status()
            print("\n📊 Client Status:")
            print(f"  Running: {status['is_running']}")
            print(f"  Transmitting: {status['is_transmitting']}")
            print(f"  Signaling: {'Connected' if status['signaling_connected'] else 'Disconnected'}")
            print(f"  Connected Peers: {', '.join(status['connected_peers']) if status['connected_peers'] else 'None'}")
            
            if hasattr(self, '_packet_stats'):
                total = sum(self._packet_stats.values())
                print("\n📦 Packet Statistics:")
                print(f"  TEXT: {self._packet_stats.get('TEXT', 0)}")
                print(f"  PROSODY: {self._packet_stats.get('PROSODY', 0)}")
                print(f"  TIMBRE: {self._packet_stats.get('TIMBRE', 0)}")
                print(f"  Total: {total}")
        
        elif cmd == "help":
            print("\nAvailable Commands:")
            print("  connect <peer_id>  - Connect to a peer")
            print("  peers              - List available peers")
            if not self.output_audio_only:
                print("  start              - Start transmitting (push-to-talk)")
                print("  stop               - Stop transmitting")
            print("  status             - Show client status")
            print("  help               - Show this help message")
            print("  quit               - Exit the application")
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for available commands")
    
    def get_status(self) -> dict:
        """Get current client status.
        
        Returns:
            Status dictionary
        """
        return {
            "peer_id": self.peer_id,
            "is_running": self.is_running,
            "is_transmitting": self.is_transmitting,
            "connected_peers": list(self.connected_peers.keys()),
            "signaling_connected": self.signaling.is_connected
        }
