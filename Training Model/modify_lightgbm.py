import json
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check which source file exists
if os.path.exists(os.path.join(script_dir, 'lightgbm_new.ipynb')):
    source_notebook = os.path.join(script_dir, 'lightgbm_new.ipynb')
    print("Using lightgbm_new.ipynb as source")
else:
    source_notebook = os.path.join(script_dir, 'lightgbm.ipynb')
    print("Using lightgbm.ipynb as source")

output_notebook = os.path.join(script_dir, 'lightgbm.ipynb')

print(f"Reading from: {source_notebook}")
print(f"Writing to: {output_notebook}")

# Read the original notebook
with open(source_notebook, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"\nTotal cells in original notebook: {len(nb['cells'])}")

# Track modifications
modifications = []

# Find and modify specific cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    
    source_text = ''.join(cell.get('source', []))
    
    # 1. Change dataset path
    if 'IAQ Baqubah Teaching Hospital' in source_text and 'pd.read_csv' in source_text:
        cell['source'] = [
            'file_path = "cleaned_dataset.csv"\n',
            'df = pd.read_csv(file_path)\n',
            '\n',
            '# Display basic info\n',
            'print(df.info())\n',
            'df.head()'
        ]
        cell['outputs'] = []
        cell['execution_count'] = None
        modifications.append(f"Cell {i}: Changed dataset to cleaned_dataset.csv")
    
    # 2. Remove eCO2 handling, add timestamp sorting
    elif 'eCO2 missing values' in source_text or ('eCO2' in source_text and 'corr' in source_text and 'interpolate' in source_text):
        cell['source'] = [
            '# Sort by timestamp and set as index\n',
            'df = df.sort_values(\'ts\')\n',
            'df.set_index(\'ts\', inplace=True)\n',
            '\n',
            '# Handle any remaining missing values with forward fill\n',
            'df = df.ffill().bfill()\n',
            '\n',
            'print("Missing values after handling:")\n',
            'print(df.isnull().sum())'
        ]
        cell['outputs'] = []
        cell['execution_count'] = None
        modifications.append(f"Cell {i}: Removed eCO2 handling, added timestamp sorting")
    
    # 3. Update feature selection
    elif "features = ['CO2'" in source_text and 'TVOC' in source_text:
        cell['source'] = [
            '# Define base features (excluding eCO2 which was dropped from cleaned dataset)\n',
            'base_features = [\'CO2\', \'TVOC\', \'PM10\', \'Air Quality\', \'CO\', \'O3\', \'Temp\', \'Hum\', \'LDR\']\n',
            'target = \'PM2.5\'\n',
            '\n',
            'print(f"Base features: {base_features}")\n',
            'print(f"Target: {target}")'
        ]
        cell['outputs'] = []
        cell['execution_count'] = None
        modifications.append(f"Cell {i}: Updated feature selection to exclude eCO2")
    
    # 4. Replace train_test_split with time-based split
    elif 'train_test_split' in source_text:
        cell['source'] = [
            '# Time-based train/test split (80/20)\n',
            '# This preserves temporal order for time-series data\n',
            'split_idx = int(len(df_temporal) * 0.8)\n',
            '\n',
            '# Prepare features and target\n',
            'feature_cols = [col for col in df_temporal.columns if col != target]\n',
            'X = df_temporal[feature_cols]\n',
            'y = df_temporal[target]\n',
            '\n',
            '# Split data\n',
            'X_train = X.iloc[:split_idx]\n',
            'X_test = X.iloc[split_idx:]\n',
            'y_train = y.iloc[:split_idx]\n',
            'y_test = y.iloc[split_idx:]\n',
            '\n',
            'print(f"Training set size: {len(X_train)}")\n',
            'print(f"Test set size: {len(X_test)}")\n',
            'print(f"Number of features: {X_train.shape[1]}")'
        ]
        cell['outputs'] = []
        cell['execution_count'] = None
        modifications.append(f"Cell {i}: Changed to time-based split")
    
    # 5. Update model training (LGBMRegressor instead of XGBRegressor)
    elif 'LGBMRegressor' in source_text and 'fit' in source_text:
        cell['source'] = [
            '# Initialize LightGBM Regressor\n',
            'model = lgb.LGBMRegressor(\n',
            '    objective=\'regression\',\n',
            '    n_estimators=100,\n',
            '    learning_rate=0.1,\n',
            '    max_depth=5,\n',
            '    random_state=42\n',
            ')\n',
            '\n',
            '# Train model\n',
            'model.fit(X_train, y_train)\n',
            '\n',
            'print("Model training complete.")'
        ]
        cell['outputs'] = []
        cell['execution_count'] = None
        modifications.append(f"Cell {i}: Updated model training")
    
    # 6. Add MAE to evaluation
    elif 'mean_squared_error' in source_text and 'r2_score' in source_text and 'predict' in source_text:
        cell['source'] = [
            '# Make predictions\n',
            'y_pred = model.predict(X_test)\n',
            '\n',
            '# Evaluate the model\n',
            'from sklearn.metrics import mean_absolute_error\n',
            '\n',
            'mae = mean_absolute_error(y_test, y_pred)\n',
            'mse = mean_squared_error(y_test, y_pred)\n',
            'rmse = np.sqrt(mse)\n',
            'r2 = r2_score(y_test, y_pred)\n',
            '\n',
            'print(f"Mean Absolute Error (MAE): {mae:.4f}")\n',
            'print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")\n',
            'print(f"R-squared (R2): {r2:.4f}")\n',
            '\n',
            '# Feature Importance\n',
            'lgb.plot_importance(model, max_num_features=20)\n',
            'plt.tight_layout()\n',
            'plt.show()'
        ]
        cell['outputs'] = []
        cell['execution_count'] = None
        modifications.append(f"Cell {i}: Added MAE metric")

# Insert temporal feature engineering cells
# Find the position after feature selection
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and any('Model Training' in line or '4.' in line for line in cell.get('source', [])):
        insert_idx = i
        break

if insert_idx:
    # Insert markdown header
    feature_eng_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3.5 Temporal Feature Engineering\n", "\n", "Create lag features, rolling statistics, and time-based features for time-series forecasting."]
    }
    
    # Insert feature engineering code
    feature_eng_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create temporal features for time-series modeling\n",
            "def create_temporal_features(df, target_col, base_features):\n",
            "    \"\"\"\n",
            "    Create lag features, rolling statistics, and time-based features\n",
            "    for time-series forecasting with tree-based models.\n",
            "    \"\"\"\n",
            "    df_features = df.copy()\n",
            "    \n",
            "    # Time-based features\n",
            "    df_features['hour'] = df_features.index.hour\n",
            "    df_features['day_of_week'] = df_features.index.dayofweek\n",
            "    df_features['month'] = df_features.index.month\n",
            "    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)\n",
            "    \n",
            "    # Lag features for target variable (PM2.5)\n",
            "    for lag in [1, 2, 3, 6, 12, 24]:\n",
            "        df_features[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)\n",
            "    \n",
            "    # Lag features for key variables\n",
            "    key_vars = ['CO2', 'TVOC', 'PM10']\n",
            "    for var in key_vars:\n",
            "        if var in df.columns:\n",
            "            for lag in [1, 3, 6]:\n",
            "                df_features[f'{var}_lag_{lag}'] = df[var].shift(lag)\n",
            "    \n",
            "    # Rolling statistics for target\n",
            "    for window in [3, 6, 12, 24]:\n",
            "        df_features[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()\n",
            "    \n",
            "    for window in [6, 12, 24]:\n",
            "        df_features[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()\n",
            "    \n",
            "    for window in [12, 24]:\n",
            "        df_features[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()\n",
            "        df_features[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()\n",
            "    \n",
            "    # Drop rows with NaN values created by lag/rolling features\n",
            "    df_features = df_features.dropna()\n",
            "    \n",
            "    return df_features\n",
            "\n",
            "# Create temporal features\n",
            "df_temporal = create_temporal_features(df, target, base_features)\n",
            "\n",
            "print(f\"Original dataset shape: {df.shape}\")\n",
            "print(f\"Dataset shape after temporal features: {df_temporal.shape}\")\n",
            "print(f\"\\nNew features created: {df_temporal.shape[1] - df.shape[1]}\")\n",
            "print(f\"\\nFirst few temporal feature names:\")\n",
            "print([col for col in df_temporal.columns if col not in df.columns][:10])"
        ]
    }
    
    nb['cells'].insert(insert_idx, feature_eng_header)
    nb['cells'].insert(insert_idx + 1, feature_eng_code)
    modifications.append(f"Inserted temporal feature engineering at position {insert_idx}")

# Save the modified notebook
with open(output_notebook, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("\n" + "="*60)
print("LightGBM notebook successfully modified!")
print("="*60)
print(f"\nTotal cells in modified notebook: {len(nb['cells'])}")
print(f"\nModifications made ({len(modifications)}):")
for mod in modifications:
    print(f"  ✓ {mod}")

print("\n" + "="*60)
print("Summary of changes:")
print("="*60)
print("1. Changed dataset to cleaned_dataset.csv")
print("2. Removed eCO2 handling (column already dropped)")
print("3. Added timestamp sorting and indexing")
print("4. Updated feature selection to exclude eCO2")
print("5. Added temporal feature engineering:")
print("   - Lag features (1, 2, 3, 6, 12, 24 hours)")
print("   - Rolling statistics (mean, std, min, max)")
print("   - Time-based features (hour, day_of_week, month, is_weekend)")
print("6. Changed to time-based train/test split (80/20)")
print("7. Updated model training")
print("8. Added MAE to evaluation metrics")
print("\nOutput saved to: lightgbm.ipynb")
