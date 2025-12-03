import { useState } from 'react';

import CallInterface from './components/CallInterface';

function App() {
  const [isInCall, setIsInCall] = useState(false)
  const [peerId, setPeerId] = useState('')
  const [ownPeerId, setOwnPeerId] = useState('')
  const [signalingUrl, setSignalingUrl] = useState('ws://localhost:8765')
  const [qualityMode, setQualityMode] = useState('balanced')
  const [isConnected, setIsConnected] = useState(false)

  const handleConnect = () => {
    if (ownPeerId.trim()) {
      // TODO: Connect to signaling server
      setIsConnected(true)
      console.log(`Connecting as ${ownPeerId} to ${signalingUrl}`)
    }
  }

  const handleStartCall = () => {
    if (peerId.trim() && isConnected) {
      setIsInCall(true)
    }
  }

  const handleEndCall = () => {
    setIsInCall(false)
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">
            🎙️ STT-Compression-TTS Voice Call
          </h1>
          <p className="text-gray-400">
            Ultra-low bandwidth voice communication (&lt;650 bps)
          </p>
        </header>

        {!isInCall ? (
          <div className="max-w-md mx-auto bg-gray-800 rounded-lg p-8 shadow-xl">
            <h2 className="text-2xl font-semibold mb-6">Connection Setup</h2>
            
            {/* Connection Section */}
            {!isConnected ? (
              <>
                <div className="mb-4">
                  <label className="block text-sm font-medium mb-2">
                    Your Peer ID
                  </label>
                  <input
                    type="text"
                    value={ownPeerId}
                    onChange={(e) => setOwnPeerId(e.target.value)}
                    placeholder="my-peer-id"
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div className="mb-4">
                  <label className="block text-sm font-medium mb-2">
                    Signaling Server
                  </label>
                  <input
                    type="text"
                    value={signalingUrl}
                    onChange={(e) => setSignalingUrl(e.target.value)}
                    placeholder="ws://localhost:8765"
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div className="mb-4">
                  <label className="block text-sm font-medium mb-2">
                    Quality Mode
                  </label>
                  <select
                    value={qualityMode}
                    onChange={(e) => setQualityMode(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-500"
                  >
                    <option value="minimal">Minimal (~300 bps)</option>
                    <option value="balanced">Balanced (~450 bps)</option>
                    <option value="high">High Quality (~600 bps)</option>
                  </select>
                </div>

                <button
                  onClick={handleConnect}
                  disabled={!ownPeerId.trim()}
                  className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-6 py-3 rounded-lg font-semibold transition-colors"
                >
                  Connect to Server
                </button>
              </>
            ) : (
              <>
                <div className="mb-6 p-4 bg-green-900/30 border border-green-700 rounded-lg">
                  <div className="flex items-center space-x-2 text-green-400">
                    <span className="text-2xl">✓</span>
                    <span>Connected as <strong>{ownPeerId}</strong></span>
                  </div>
                </div>

                <div className="mb-4">
                  <label className="block text-sm font-medium mb-2">
                    Call Peer ID
                  </label>
                  <input
                    type="text"
                    value={peerId}
                    onChange={(e) => setPeerId(e.target.value)}
                    placeholder="peer-id-to-call"
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-500"
                  />
                </div>

                <button
                  onClick={handleStartCall}
                  disabled={!peerId.trim()}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-6 py-3 rounded-lg font-semibold transition-colors"
                >
                  Start Call
                </button>

                <button
                  onClick={() => setIsConnected(false)}
                  className="w-full mt-2 bg-gray-700 hover:bg-gray-600 px-6 py-2 rounded-lg font-semibold transition-colors"
                >
                  Disconnect
                </button>
              </>
            )}

            <div className="mt-6 pt-6 border-t border-gray-700">
              <h3 className="text-sm font-medium mb-2">System Status</h3>
              <div className="space-y-1 text-sm text-gray-400">
                <div>{isConnected ? '🟢' : '🔴'} Signaling Server: {isConnected ? 'Connected' : 'Disconnected'}</div>
                <div>🟢 Audio Devices: Ready</div>
                <div>� Models: Loading...</div>
              </div>
            </div>
          </div>
        ) : (
          <CallInterface peerId={peerId} onEndCall={handleEndCall} />
        )}

        <footer className="mt-12 text-center text-sm text-gray-500">
          <p>Built with React + WebRTC + Faster-Whisper + XTTS-v2</p>
        </footer>
      </div>
    </div>
  )
}

export default App
