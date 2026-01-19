import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import InputPanel from './components/InputPanel'
import Results from './components/Results'
import axios from 'axios'

const API_URL = import.meta.env.DEV ? '/api' : 'http://localhost:8000'

function App() {
  const [inputs, setInputs] = useState({
    temp: 18,
    rain: 850,
    ph: 6.5,
    clay: 25,
    sand: 40
  })
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handlePredict = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_URL}/predict`, inputs)
      setResults(response.data)
    } catch (err) {
      console.error('Prediction error:', err)
      setError(err.response?.data?.detail || 'Failed to get prediction. Is the backend running?')
      setResults(null)
    } finally {
      setLoading(false)
    }
  }, [inputs])

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="max-w-6xl mx-auto mb-8 text-center"
      >
        <div className="flex items-center justify-center gap-3 mb-4">
          <motion.div
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
          >
            <svg className="w-10 h-10 text-emerald-400" viewBox="0 0 24 24" fill="currentColor">
              <path d="M17 8C8 10 5.9 16.17 3.82 21.34l1.89.66.95-2.3c.48.17.98.3 1.34.3C19 20 22 3 22 3c-1 2-8 2.25-13 3.25S2 11.5 2 13.5s1.75 3.75 1.75 3.75C7 8 17 8 17 8z" />
            </svg>
          </motion.div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-300 via-teal-200 to-cyan-300 bg-clip-text text-transparent">
            Crop Recommender
          </h1>
        </div>
        <p className="text-lg text-emerald-100/70">
          AI-Powered Crop Recommendations for Optimal Yields
        </p>
      </motion.header>

      {/* Main Dashboard */}
      <main className="max-w-6xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left: Input Panel */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <InputPanel
              inputs={inputs}
              setInputs={setInputs}
              onPredict={handlePredict}
              loading={loading}
            />
          </motion.div>

          {/* Right: Results */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <AnimatePresence mode="wait">
              {error ? (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="glass rounded-2xl p-8 text-center"
                >
                  <div className="text-red-400 text-lg mb-2">⚠️ Error</div>
                  <p className="text-red-300/80">{error}</p>
                </motion.div>
              ) : results ? (
                <Results key="results" results={results} />
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="glass rounded-2xl p-12 text-center h-full flex flex-col items-center justify-center min-h-[400px]"
                >
                  <motion.div
                    animate={{
                      scale: [1, 1.05, 1],
                      opacity: [0.5, 0.8, 0.5]
                    }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="text-6xl mb-6"
                  >
                    🌾
                  </motion.div>
                  <h3 className="text-xl font-semibold text-emerald-100 mb-2">
                    Ready to Analyze
                  </h3>
                  <p className="text-emerald-100/60">
                    Adjust the environmental parameters and click<br />
                    <span className="text-emerald-400 font-medium">"Get Recommendation"</span> to discover the best crops
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </main>

      <motion.footer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="max-w-6xl mx-auto mt-12 text-center text-emerald-100/30 text-xs flex flex-col gap-1"
      >
        <p>Powered by Machine Learning • 47 Crop Classes • Built with 🌱</p>
      </motion.footer>
    </div>
  )
}

export default App
