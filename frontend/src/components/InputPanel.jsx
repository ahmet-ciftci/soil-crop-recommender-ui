import { motion } from 'framer-motion'

const sliderConfigs = [
  {
    key: 'temp',
    label: 'Temperature',
    unit: '°C',
    min: -15,
    max: 40,
    step: 0.5,
    icon: '🌡️',
    gradient: 'from-blue-400 to-red-400'
  },
  {
    key: 'rain',
    label: 'Annual Rainfall',
    unit: 'mm',
    min: 0,
    max: 5000,
    step: 10,
    icon: '🌧️',
    gradient: 'from-cyan-400 to-blue-500'
  },
  {
    key: 'ph',
    label: 'Soil pH',
    unit: '',
    min: 4.0,
    max: 9.5,
    step: 0.1,
    icon: '🧪',
    gradient: 'from-yellow-400 to-emerald-400'
  },
  {
    key: 'clay',
    label: 'Clay Content',
    unit: '%',
    min: 0,
    max: 70,
    step: 1,
    icon: '🏔️',
    gradient: 'from-amber-600 to-amber-400'
  },
  {
    key: 'sand',
    label: 'Sand Content',
    unit: '%',
    min: 0,
    max: 100,
    step: 1,
    icon: '🏖️',
    gradient: 'from-yellow-300 to-orange-400'
  }
]

function InputPanel({ inputs, setInputs, onPredict, loading }) {
  const handleChange = (key, value) => {
    setInputs(prev => ({ ...prev, [key]: parseFloat(value) }))
  }

  return (
    <div className="glass-strong rounded-2xl p-6 sm:p-8">
      <h2 className="text-xl font-semibold text-emerald-100 mb-6 flex items-center gap-2">
        <span className="text-2xl">🎛️</span>
        Environmental Parameters
      </h2>

      <div className="space-y-6">
        {sliderConfigs.map((config, index) => (
          <motion.div
            key={config.key}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="space-y-2"
          >
            <div className="flex justify-between items-center">
              <label className="text-emerald-100/80 font-medium flex items-center gap-2">
                <span>{config.icon}</span>
                {config.label}
              </label>
              <motion.span 
                key={inputs[config.key]}
                initial={{ scale: 1.2, color: '#34d399' }}
                animate={{ scale: 1, color: '#a7f3d0' }}
                className="text-emerald-200 font-semibold tabular-nums"
              >
                {inputs[config.key]}{config.unit}
              </motion.span>
            </div>
            
            <div className="relative">
              <div 
                className={`absolute inset-0 h-2 rounded-full bg-gradient-to-r ${config.gradient} opacity-30`}
                style={{ top: '3px' }}
              />
              <input
                type="range"
                min={config.min}
                max={config.max}
                step={config.step}
                value={inputs[config.key]}
                onChange={(e) => handleChange(config.key, e.target.value)}
                className="relative z-10 w-full"
              />
            </div>
            
            <div className="flex justify-between text-xs text-emerald-100/40">
              <span>{config.min}{config.unit}</span>
              <span>{config.max}{config.unit}</span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Predict Button */}
      <motion.button
        onClick={onPredict}
        disabled={loading}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        className={`
          w-full mt-8 py-4 px-6 rounded-xl font-semibold text-lg
          transition-all duration-300
          ${loading 
            ? 'bg-emerald-700/50 cursor-wait' 
            : 'bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40'
          }
          text-white
        `}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <motion.span
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            >
              ⚙️
            </motion.span>
            Analyzing...
          </span>
        ) : (
          <span className="flex items-center justify-center gap-2">
            <span>🌱</span>
            Get Recommendation
          </span>
        )}
      </motion.button>
    </div>
  )
}

export default InputPanel
