import { motion } from 'framer-motion'

// Category emoji mapping
const categoryEmojis = {
  CEREALS_TEMPERATE: '🌾',
  CEREALS_WARM: '🌽',
  RICE: '🍚',
  PULSES: '🫘',
  OILSEEDS: '🌻',
  ROOTS: '🥔',
  TROPICAL: '🌴',
  FRUITS_TEMPERATE: '🍎',
  VEGETABLES: '🥬',
  STIMULANTS: '☕',
  FIBER: '🧶',
  SUGAR: '🍬',
  OTHER: '🌿'
}

// Category display names
const categoryNames = {
  CEREALS_TEMPERATE: 'Temperate Cereals',
  CEREALS_WARM: 'Warm Cereals',
  RICE: 'Rice',
  PULSES: 'Pulses & Legumes',
  OILSEEDS: 'Oilseed Crops',
  ROOTS: 'Root & Tuber Crops',
  TROPICAL: 'Tropical Crops',
  FRUITS_TEMPERATE: 'Temperate Fruits',
  VEGETABLES: 'Vegetables',
  STIMULANTS: 'Stimulant Crops',
  FIBER: 'Fiber Crops',
  SUGAR: 'Sugar Crops',
  OTHER: 'Other Crops'
}

// Category descriptions (examples of crops)
const categoryExamples = {
  CEREALS_TEMPERATE: 'Wheat, Barley, Oats',
  CEREALS_WARM: 'Maize, Sorghum, Millet',
  RICE: 'Paddy Rice',
  PULSES: 'Soybeans, Beans, Chickpeas, Groundnuts',
  OILSEEDS: 'Sunflower, Rapeseed, Sesame, Oil Palm',
  ROOTS: 'Potatoes, Cassava, Sweet Potatoes, Yams',
  TROPICAL: 'Banana, Cocoa, Rubber, Tropical Fruits',
  FRUITS_TEMPERATE: 'Apples, Citrus, Grapes, Pears',
  VEGETABLES: 'Tomatoes, Onions, Mixed Vegetables',
  STIMULANTS: 'Coffee, Tea, Tobacco',
  FIBER: 'Cotton, Jute, Hemp',
  SUGAR: 'Sugarcane, Sugarbeet',
  OTHER: 'Spices, Fodder, Misc Crops'
}

const getCategoryEmoji = (cat) => categoryEmojis[cat] || '🌱'
const getCategoryName = (code) => categoryNames[code] || code
const getCategoryExamples = (code) => categoryExamples[code] || ''

function Results({ results }) {
  const maxProb = Math.max(...results.top_5.map(c => c.prob))

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
      className="glass-strong rounded-2xl p-6 sm:p-8"
    >
      {/* Winner Section */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
        className="text-center mb-8"
      >
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.3, type: 'spring', stiffness: 300 }}
          className="text-6xl mb-4"
        >
          {getCategoryEmoji(results.winner)}
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <p className="text-emerald-100/60 text-sm uppercase tracking-wider mb-1">
            Recommended Category
          </p>
          <h2 className="text-3xl font-bold bg-gradient-to-r from-emerald-300 to-teal-200 bg-clip-text text-transparent">
            {getCategoryName(results.winner)}
          </h2>
          <p className="text-emerald-100/50 text-sm mt-2">
            {getCategoryExamples(results.winner)}
          </p>
        </motion.div>
      </motion.div>

      {/* Divider */}
      <div className="h-px bg-gradient-to-r from-transparent via-emerald-500/30 to-transparent mb-6" />

      {/* Top 5 Section */}
      <div>
        <h3 className="text-lg font-semibold text-emerald-100 mb-4 flex items-center gap-2">
          <span>📊</span>
          Top 5 Categories
        </h3>

        <div className="space-y-3">
          {results.top_5.map((category, index) => (
            <motion.div
              key={category.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
              className="relative"
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-lg">{getCategoryEmoji(category.name)}</span>
                  <span className="text-emerald-100 font-medium">
                    {getCategoryName(category.name)}
                  </span>
                </div>
                <span className="text-emerald-300 font-semibold tabular-nums">
                  {(category.prob * 100).toFixed(1)}%
                </span>
              </div>
              
              {/* Progress Bar */}
              <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(category.prob / maxProb) * 100}%` }}
                  transition={{ 
                    delay: 0.6 + index * 0.1, 
                    duration: 0.8,
                    ease: [0.16, 1, 0.3, 1]
                  }}
                  className={`h-full rounded-full ${
                    index === 0 
                      ? 'bg-gradient-to-r from-emerald-400 to-teal-400' 
                      : 'bg-gradient-to-r from-emerald-600/80 to-teal-600/80'
                  }`}
                />
              </div>
              
              {/* Category examples */}
              <p className="text-emerald-100/40 text-xs mt-1">
                {getCategoryExamples(category.name)}
              </p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Confidence Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="mt-6 pt-4 border-t border-white/5"
      >
        <div className="flex items-center justify-between text-sm">
          <span className="text-emerald-100/50">Model Confidence</span>
          <span className={`font-semibold ${
            results.top_5[0].prob > 0.4 ? 'text-emerald-400' :
            results.top_5[0].prob > 0.2 ? 'text-yellow-400' :
            'text-orange-400'
          }`}>
            {results.top_5[0].prob > 0.4 ? '🟢 High' :
             results.top_5[0].prob > 0.2 ? '🟡 Medium' :
             '🟠 Low'}
          </span>
        </div>
      </motion.div>
    </motion.div>
  )
}

export default Results
