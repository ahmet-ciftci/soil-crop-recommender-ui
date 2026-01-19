#!/usr/bin/env python3
"""
soil-crop-recommender-ui - Crop Recommendation Model Training
==============================================================
Trains a Random Forest Classifier to predict crop suitability
based on environmental features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "crop_recommender_global.csv"
MODEL_FILE = DATA_DIR / "crop_recommender_model.pkl"

# Feature columns (NO Lat/Lon - we want environmental rules, not location memory)
FEATURE_COLUMNS = ['Temp', 'Rain', 'pH', 'Clay', 'Sand']
TARGET_COLUMN = 'Label'

# Model hyperparameters
N_ESTIMATORS = 200
MAX_DEPTH = 20
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Minimum samples required per crop (filter out rare crops)
MIN_SAMPLES_PER_CROP = 50

# =============================================================================
# CROP CATEGORY MAPPING
# =============================================================================
# Group crops with similar growing conditions to reduce class confusion

CROP_CATEGORIES = {
    # Cereals - Cool/Temperate
    'WHEA': 'CEREALS_TEMPERATE',
    'BARL': 'CEREALS_TEMPERATE',
    'OCER': 'CEREALS_TEMPERATE',
    
    # Cereals - Warm
    'MAIZ': 'CEREALS_WARM',
    'SORG': 'CEREALS_WARM',
    'MILL': 'CEREALS_WARM',
    'PMIL': 'CEREALS_WARM',
    
    # Rice (unique water requirements)
    'RICE': 'RICE',
    
    # Pulses/Legumes
    'BEAN': 'PULSES',
    'SOYB': 'PULSES',
    'CHIC': 'PULSES',
    'COWP': 'PULSES',
    'PIGE': 'PULSES',
    'LENT': 'PULSES',
    'OPUL': 'PULSES',
    'GROU': 'PULSES',  # Groundnut is a legume
    
    # Oilseeds (non-legume)
    'SUNF': 'OILSEEDS',
    'RAPE': 'OILSEEDS',
    'SESA': 'OILSEEDS',
    'OOIL': 'OILSEEDS',
    'OILP': 'OILSEEDS',
    'CNUT': 'OILSEEDS',
    
    # Root crops
    'POTA': 'ROOTS',
    'SWPO': 'ROOTS',
    'CASS': 'ROOTS',
    'YAMS': 'ROOTS',
    'ORTS': 'ROOTS',
    
    # Tropical fruits & plantations
    'BANA': 'TROPICAL',
    'PLNT': 'TROPICAL',
    'TROF': 'TROPICAL',
    'COCO': 'TROPICAL',
    'RUBB': 'TROPICAL',
    
    # Temperate fruits
    'TEMF': 'FRUITS_TEMPERATE',
    'CITR': 'FRUITS_TEMPERATE',
    
    # Vegetables
    'VEGE': 'VEGETABLES',
    'TOMA': 'VEGETABLES',
    'ONIO': 'VEGETABLES',
    
    # Stimulants (Coffee, Tea, Cocoa already in TROPICAL)
    'COFF': 'STIMULANTS',
    'RCOF': 'STIMULANTS',
    'TEAS': 'STIMULANTS',
    'TOBA': 'STIMULANTS',
    
    # Industrial/Fiber crops
    'COTT': 'FIBER',
    'OFIB': 'FIBER',
    
    # Sugar crops
    'SUGC': 'SUGAR',
    'SUGB': 'SUGAR',
    
    # Catch-all
    'REST': 'OTHER',
}

# Category display names
CATEGORY_NAMES = {
    'CEREALS_TEMPERATE': 'Temperate Cereals (Wheat, Barley)',
    'CEREALS_WARM': 'Warm Cereals (Maize, Sorghum, Millet)',
    'RICE': 'Rice',
    'PULSES': 'Pulses & Legumes',
    'OILSEEDS': 'Oilseed Crops',
    'ROOTS': 'Root & Tuber Crops',
    'TROPICAL': 'Tropical Crops',
    'FRUITS_TEMPERATE': 'Temperate Fruits',
    'VEGETABLES': 'Vegetables',
    'STIMULANTS': 'Stimulant Crops (Coffee, Tea)',
    'FIBER': 'Fiber Crops',
    'SUGAR': 'Sugar Crops',
    'OTHER': 'Other Crops',
}


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_model():
    """
    Main function to train the crop recommendation model.
    """
    print("=" * 60)
    print("🌱 soil-crop-recommender-ui - Crop Recommendation Model Training")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    print("📂 Loading dataset...")
    
    df = pd.read_csv(INPUT_FILE)
    print(f"   ✓ Loaded {len(df):,} records from {INPUT_FILE.name}")
    
    # Drop rows with missing labels
    initial_count = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"   ⚠ Dropped {dropped:,} rows with missing labels")
    
    print(f"   ✓ Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Data Cleaning & Categorization
    # -------------------------------------------------------------------------
    print("🧹 Cleaning and categorizing data...")
    
    # Remove NO_CROP - not useful for recommendations
    before_no_crop = len(df)
    df = df[df[TARGET_COLUMN] != 'NO_CROP']
    print(f"   ✓ Removed {before_no_crop - len(df):,} NO_CROP records")
    
    # Filter out rare crops (< MIN_SAMPLES_PER_CROP samples)
    crop_counts = df[TARGET_COLUMN].value_counts()
    rare_crops = crop_counts[crop_counts < MIN_SAMPLES_PER_CROP].index.tolist()
    before_rare = len(df)
    df = df[~df[TARGET_COLUMN].isin(rare_crops)]
    print(f"   ✓ Removed {before_rare - len(df):,} records from {len(rare_crops)} rare crops")
    print(f"     (Crops with < {MIN_SAMPLES_PER_CROP} samples: {', '.join(rare_crops)})")
    
    # Map individual crops to categories
    df['Category'] = df[TARGET_COLUMN].map(CROP_CATEGORIES)
    
    # Handle any unmapped crops
    unmapped = df[df['Category'].isna()][TARGET_COLUMN].unique()
    if len(unmapped) > 0:
        print(f"   ⚠ Unmapped crops set to OTHER: {unmapped}")
        df['Category'] = df['Category'].fillna('OTHER')
    
    # Show category distribution
    print()
    print("📊 Category Distribution:")
    print("-" * 50)
    cat_counts = df['Category'].value_counts()
    for cat, count in cat_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        cat_name = CATEGORY_NAMES.get(cat, cat)
        print(f"   {cat:20s}: {count:4,} ({pct:5.1f}%) {bar}")
    
    print(f"\n   ✓ Final dataset: {len(df):,} records, {df['Category'].nunique()} categories")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Feature Selection
    # -------------------------------------------------------------------------
    print("🔬 Preparing features...")
    
    X = df[FEATURE_COLUMNS]
    y = df['Category']  # Use categories instead of individual crops
    
    n_classes = y.nunique()
    print(f"   ✓ Features (X): {FEATURE_COLUMNS}")
    print(f"   ✓ Target (y): Category ({n_classes} classes)")
    print()
    
    # Quick feature summary
    print("📊 Feature Statistics:")
    print("-" * 50)
    for col in FEATURE_COLUMNS:
        print(f"   {col:6s}: min={X[col].min():8.2f}, max={X[col].max():8.2f}, mean={X[col].mean():8.2f}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Train/Test Split
    # -------------------------------------------------------------------------
    print("✂️  Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution
    )
    
    print(f"   ✓ Training set: {len(X_train):,} samples ({100*(1-TEST_SIZE):.0f}%)")
    print(f"   ✓ Test set:     {len(X_test):,} samples ({100*TEST_SIZE:.0f}%)")
    print()
    
    # -------------------------------------------------------------------------
    # Step 5: Model Training
    # -------------------------------------------------------------------------
    print("🧠 Training Random Forest Classifier...")
    print(f"   Parameters: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")
    
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        n_jobs=-1,  # Use all CPU cores
        random_state=RANDOM_STATE,
        class_weight='balanced'  # Handle imbalanced classes
    )
    
    model.fit(X_train, y_train)
    print("   ✓ Training complete!")
    print()
    
    # -------------------------------------------------------------------------
    # Step 5: Evaluation
    # -------------------------------------------------------------------------
    print("📈 Evaluating model performance...")
    print("-" * 60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Overall accuracy (Top-1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 TOP-1 ACCURACY: {accuracy:.2%}")
    
    # Top-5 Accuracy
    top_5_correct = 0
    classes = model.classes_
    
    for i, true_label in enumerate(y_test):
        # Get indices of top 5 predictions (highest probabilities)
        top_5_indices = np.argsort(y_proba[i])[-5:][::-1]
        top_5_classes = [classes[idx] for idx in top_5_indices]
        
        if true_label in top_5_classes:
            top_5_correct += 1
    
    top_5_accuracy = top_5_correct / len(y_test)
    print(f"🎯 TOP-5 ACCURACY: {top_5_accuracy:.2%}")
    print()
    
    # Classification report
    print("📋 CLASSIFICATION REPORT:")
    print("-" * 60)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    
    # -------------------------------------------------------------------------
    # Step 6: Feature Importance
    # -------------------------------------------------------------------------
    print("🔍 FEATURE IMPORTANCE:")
    print("-" * 40)
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    for _, row in importance_df.iterrows():
        pct = row['Importance'] * 100
        bar = "█" * int(pct / 2)
        print(f"   {row['Feature']:6s}: {pct:5.1f}% {bar}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 8: Save Model
    # -------------------------------------------------------------------------
    print("💾 Saving model...")
    
    # Save model with metadata
    model_data = {
        'model': model,
        'features': FEATURE_COLUMNS,
        'classes': list(model.classes_),
        'category_names': CATEGORY_NAMES,
        'accuracy': accuracy,
        'top_5_accuracy': top_5_accuracy,
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH
    }
    
    joblib.dump(model_data, MODEL_FILE)
    print(f"   ✓ Model saved to: {MODEL_FILE}")
    print()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("✅ Model saved successfully!")
    print("=" * 60)
    print()
    print(f"   📁 Model file:     {MODEL_FILE.name}")
    print(f"   🎯 Top-1 Accuracy: {accuracy:.2%}")
    print(f"   🎯 Top-5 Accuracy: {top_5_accuracy:.2%}")
    print(f"   🌾 Categories:     {n_classes} crop categories")
    print(f"   🔬 Features:       {', '.join(FEATURE_COLUMNS)}")
    print()
    
    return model, accuracy, top_5_accuracy


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        model, accuracy, top_5_accuracy = train_model()
    except FileNotFoundError as e:
        print(f"❌ Error: Dataset not found!")
        print(f"   {e}")
        print("\n   Please run 1_build_global_dataset.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
