import { useEffect, useState } from 'react';

interface CallInterfaceProps {
  peerId: string
  onEndCall: () => void
}

function CallInterface({ peerId, onEndCall }: CallInterfaceProps) {
  const [isMuted, setIsMuted] = useState(false)
  const [isTransmitting, setIsTransmitting] = useState(false)
  const [callDuration, setCallDuration] = useState(0)
  const [bandwidth, setBandwidth] = useState(0)
  const [latency, setLatency] = useState(0)

  useEffect(() => {
    // Simulate call duration
    const timer = setInterval(() => {
      setCallDuration((prev) => prev + 1)
    }, 1000)

    // Simulate bandwidth metrics
    const metricsTimer = setInterval(() => {
      setBandwidth(Math.floor(Math.random() * 100) + 350) // 350-450 bps
      setLatency(Math.floor(Math.random() * 100) + 400) // 400-500 ms
    }, 1000)

    return () => {
      clearInterval(timer)
      clearInterval(metricsTimer)
    }
  }, [])

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-gray-800 rounded-lg p-8 shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-2xl font-semibold">In Call with {peerId}</h2>
            <p className="text-gray-400">Duration: {formatDuration(callDuration)}</p>
          </div>
          <div className="flex items-center space-x-2">
            <span className="flex items-center">
              <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse mr-2"></span>
              Connected
            </span>
          </div>
        </div>

        {/* Audio Visualizer Placeholder */}
        <div className="bg-gray-700 rounded-lg p-8 mb-6 h-32 flex items-center justify-center">
          <div className="flex items-end space-x-1 h-16">
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="w-2 bg-blue-500 rounded-t animate-pulse"
                style={{
                  height: `${Math.random() * 100}%`,
                  animationDelay: `${i * 0.05}s`,
                }}
              ></div>
            ))}
          </div>
        </div>

        {/* Network Stats */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-1">Bandwidth</div>
            <div className="text-2xl font-bold text-green-400">{bandwidth} bps</div>
            <div className="text-xs text-gray-500 mt-1">Target: &lt;650 bps</div>
          </div>
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-1">Latency</div>
            <div className="text-2xl font-bold text-blue-400">{latency} ms</div>
            <div className="text-xs text-gray-500 mt-1">Target: &lt;500 ms</div>
          </div>
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-1">Packet Loss</div>
            <div className="text-2xl font-bold text-yellow-400">0.2%</div>
            <div className="text-xs text-gray-500 mt-1">Excellent</div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-col items-center space-y-4">
          {/* Push-to-Talk Button */}
          <div className="flex flex-col items-center">
            <button
              onMouseDown={() => setIsTransmitting(true)}
              onMouseUp={() => setIsTransmitting(false)}
              onMouseLeave={() => setIsTransmitting(false)}
              onTouchStart={() => setIsTransmitting(true)}
              onTouchEnd={() => setIsTransmitting(false)}
              className={`px-12 py-6 rounded-full font-bold text-xl transition-all transform ${
                isTransmitting
                  ? 'bg-red-600 scale-105 shadow-lg shadow-red-500/50'
                  : 'bg-blue-600 hover:bg-blue-700 hover:scale-105'
              }`}
            >
              {isTransmitting ? '🔴 TRANSMITTING' : '🎤 HOLD TO TALK'}
            </button>
            <p className="text-sm text-gray-400 mt-2">
              {isTransmitting ? 'Release to stop' : 'Press and hold to speak'}
            </p>
          </div>

          {/* Secondary Controls */}
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setIsMuted(!isMuted)}
              className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                isMuted
                  ? 'bg-red-600 hover:bg-red-700'
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              {isMuted ? '🔇 Unmute' : '🎤 Mute'}
            </button>

            <button
              onClick={onEndCall}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-semibold transition-colors"
            >
              📞 End Call
            </button>

            <button
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-semibold transition-colors"
            >
              ⚙️ Settings
            </button>
          </div>
        </div>

        {/* Quality Mode */}
        <div className="mt-6 pt-6 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">Quality Mode:</span>
            <select className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm">
              <option>Minimal (300 bps)</option>
              <option selected>Balanced (450 bps)</option>
              <option>High Quality (600 bps)</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  )
}

export default CallInterface
