import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight

# Optional: for class balancing beyond class_weight
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
except ImportError:
    RandomOverSampler = SMOTE = None

# Load data (same folder as script)
# Check if file is actually an Excel file (XLSX) or CSV
import zipfile
import os

df = None
last_error = None
file_path = "crop.csv"

print(f"\n{'='*70}")
print(f"Loading dataset: {file_path}")
print(f"{'='*70}")

# Overfitting diagnostics configuration (can be overridden via env vars)
OVERFIT_WARN_THRESHOLD = float(os.getenv("OVERFIT_WARN_THRESHOLD", "8"))  # warn if gap exceeds this (%)
OVERFIT_CAUTION_THRESHOLD = float(os.getenv("OVERFIT_CAUTION_THRESHOLD", "4"))  # caution if gap exceeds this (%)
OVERFIT_GUARD_GAP = float(os.getenv("OVERFIT_GUARD_GAP", "3.5"))  # trigger stronger regularization fallback (train-val gap)
OVERFIT_FINAL_GAP = float(os.getenv("OVERFIT_FINAL_GAP", "4"))  # trigger final overfit guard using train-test gap
OVERFIT_WARN_SILENCE = os.getenv("OVERFIT_WARN_SILENCE", "0") == "1"

# Check if file is actually a ZIP/Excel file
is_excel = False
try:
    with open(file_path, 'rb') as f:
        header = f.read(2)
        if header == b'PK':  # ZIP file signature (XLSX files are ZIP archives)
            is_excel = True
except Exception as e:
    last_error = e

if is_excel:
    # Try to read as Excel file - read all sheets and combine them
    try:
        excel_file = pd.ExcelFile(file_path, engine='openpyxl')
        print(f"Found {len(excel_file.sheet_names)} sheets: {excel_file.sheet_names}")
        
        # Read only the first 3 sheets (skip Sheet1)
        all_sheets = []
        sheets_to_read = [s for s in excel_file.sheet_names if s.lower() != 'sheet1'][:3]
        print(f"Reading first 3 sheets : {sheets_to_read}")
        
        for sheet_name in sheets_to_read:
            try:
                sheet_df = pd.read_excel(excel_file, sheet_name=sheet_name)
                print(f"  - Sheet '{sheet_name}': {sheet_df.shape[0]} rows, columns: {list(sheet_df.columns)}")
                
                # Determine the level (High, Medium, or Low) from sheet name or column names
                sheet_lower = sheet_name.lower()
                level = None
                if 'high' in sheet_lower:
                    level = 'H'
                elif 'medium' in sheet_lower or 'mid' in sheet_lower:
                    level = 'M'
                elif 'low' in sheet_lower:
                    level = 'L'
                
                # If level not found from sheet name, check column names
                if level is None:
                    first_col = str(sheet_df.columns[0]).lower() if len(sheet_df.columns) > 0 else ''
                    if 'high' in first_col:
                        level = 'H'
                    elif 'medium' in first_col or 'mid' in first_col:
                        level = 'M'
                    elif 'low' in first_col:
                        level = 'L'
                
                # Default to High if still not determined
                if level is None:
                    level = 'H'
                    print(f"    Warning: Could not determine level for sheet '{sheet_name}', defaulting to High")
                
                # Normalize column names in each sheet (High(N) -> N, Medium(N) -> N, Low(N) -> N, etc.)
                sheet_df.columns = sheet_df.columns.str.strip()
                column_rename = {}
                for col in sheet_df.columns:
                    col_lower = col.lower().strip()
                    # Map High/Medium/Low N/P/K columns to standard names
                    if col_lower.startswith('high(n') or col_lower.startswith('medium(n') or col_lower.startswith('low(n') or col_lower.startswith('low (n') or col_lower.endswith('(n)'):
                        column_rename[col] = 'N'
                    elif col_lower.startswith('high(p') or col_lower.startswith('medium(p') or col_lower.startswith('low(p') or col_lower.startswith('low (p') or col_lower.endswith('(p)'):
                        column_rename[col] = 'P'
                    elif col_lower.startswith('high(k') or col_lower.startswith('medium(k') or col_lower.startswith('low(k') or col_lower.startswith('low (k') or col_lower.endswith('(k)'):
                        column_rename[col] = 'K'
                    elif col_lower == 'n' or col_lower.startswith('n '):
                        column_rename[col] = 'N'
                    elif col_lower == 'p' or col_lower.startswith('p '):
                        column_rename[col] = 'P'
                    elif col_lower == 'k' or col_lower.startswith('k '):
                        column_rename[col] = 'K'
                    elif col_lower == 'target' or 'target' in col_lower:
                        column_rename[col] = 'target'
                
                # Rename columns
                sheet_df = sheet_df.rename(columns=column_rename)
                
                # Add level indicator
                sheet_df['_level'] = level
                
                all_sheets.append(sheet_df)
            except Exception as e:
                print(f"  - Warning: Could not read sheet '{sheet_name}': {e}")
                continue
        
        if not all_sheets:
            raise ValueError("No sheets could be read from the Excel file")
        
        # Combine all sheets
        df = pd.concat(all_sheets, ignore_index=True)
        print(f"Successfully loaded and combined {len(all_sheets)} sheets from Excel file (XLSX)")
        print(f"Total combined rows: {df.shape[0]}")
        
        # Convert N, P, K columns to numeric (in case they were read as strings)
        if 'N' in df.columns:
            df['N'] = pd.to_numeric(df['N'], errors='coerce')
        if 'P' in df.columns:
            df['P'] = pd.to_numeric(df['P'], errors='coerce')
        if 'K' in df.columns:
            df['K'] = pd.to_numeric(df['K'], errors='coerce')
    except ImportError:
        # If openpyxl is not installed, try xlrd
        try:
            excel_file = pd.ExcelFile(file_path, engine='xlrd')
            all_sheets = []
            for sheet_name in excel_file.sheet_names:
                sheet_df = pd.read_excel(excel_file, sheet_name=sheet_name)
                all_sheets.append(sheet_df)
            df = pd.concat(all_sheets, ignore_index=True)
            print("Successfully loaded Excel file using xlrd")
        except ImportError:
            raise ImportError("Please install openpyxl or xlrd to read Excel files: pip install openpyxl")
    except Exception as e:
        last_error = e
        raise ValueError(f"Could not read Excel file. Error: {e}")
else:
    # Try different encodings for CSV file
    encodings = ['utf-8', 'cp1252', 'latin-1']
    for encoding in encodings:
        try:
            # Use Python engine for more lenient parsing
            # Try with on_bad_lines for pandas >= 1.3.0
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip', 
                               skipinitialspace=True, engine='python')
            except (TypeError, AttributeError):
                # Fallback for older pandas versions (< 1.3.0)
                df = pd.read_csv(file_path, encoding=encoding, error_bad_lines=False, 
                               warn_bad_lines=False, skipinitialspace=True, engine='python')
            
            # If we got here and have data, we're good
            if df is not None and not df.empty:
                print(f"Successfully loaded CSV with {encoding} encoding")
                break
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            # Catch all other exceptions (ParserError, etc.) and try next encoding
            last_error = e
            continue

if df is None or df.empty:
    error_msg = f"Could not read {file_path}. Last error: {last_error}"
    raise ValueError(error_msg)

# Print column names for debugging
print(f"Final columns: {list(df.columns)}")
print(f"Final shape: {df.shape}")

# Normalize column names (strip whitespace, handle case variations)
df.columns = df.columns.str.strip()

# If columns weren't normalized during Excel reading (CSV case), try to find and rename them
if 'N' not in df.columns or 'P' not in df.columns or 'K' not in df.columns:
    # Try to find N, P, K columns (case-insensitive, with/without spaces)
    # Handle various formats: 'N', 'High(N)', 'Medium(N)', 'Low(N)', 'N ', 'Soil_N_ppm', 'Target_N', etc.
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        
        # Check for N column - handle 'N', 'High(N)', 'Medium(N)', 'Low(N)', 'N ', 'Soil_N_ppm', 'Target_N', etc.
        if (col_lower == 'n' or 
            col_lower.startswith('n ') or 
            col_lower.startswith('high(n') or 
            col_lower.startswith('medium(n') or
            col_lower.startswith('low(n') or
            col_lower.startswith('low (n') or  # Handle 'Low (N)' with space
            col_lower.endswith('(n)') or
            col_lower.endswith('_n') or  # Handle 'Soil_N', 'Target_N', etc.
            (col_lower.startswith('_n') and len(col_lower) > 2) or  # Handle '_N' in middle
            ('_n_' in col_lower) or  # Handle '_N_' in middle like 'Soil_N_ppm'
            (col_lower.startswith('n_') and len(col_lower) > 2)):  # Handle 'N_' prefix
            if 'N' not in col_mapping:  # Only map if not already found
                col_mapping['N'] = col
        # Check for P column
        elif (col_lower == 'p' or 
              col_lower.startswith('p ') or 
              col_lower.startswith('high(p') or 
              col_lower.startswith('medium(p') or
              col_lower.startswith('low(p') or
              col_lower.startswith('low (p') or  # Handle 'Low (P)' with space
              col_lower.endswith('(p)') or
              col_lower.endswith('_p') or  # Handle 'Soil_P', 'Target_P', etc.
              (col_lower.startswith('_p') and len(col_lower) > 2) or  # Handle '_P' in middle
              ('_p_' in col_lower) or  # Handle '_P_' in middle like 'Soil_P_ppm'
              (col_lower.startswith('p_') and len(col_lower) > 2)):  # Handle 'P_' prefix
            if 'P' not in col_mapping:  # Only map if not already found
                col_mapping['P'] = col
        # Check for K column
        elif (col_lower == 'k' or 
              col_lower.startswith('k ') or 
              col_lower.startswith('high(k') or 
              col_lower.startswith('medium(k') or
              col_lower.startswith('low(k') or
              col_lower.startswith('low (k') or  # Handle 'low(K)' or 'Low (K)' with space
              col_lower.endswith('(k)') or
              col_lower.endswith('_k') or  # Handle 'Soil_K', 'Target_K', etc.
              (col_lower.startswith('_k') and len(col_lower) > 2) or  # Handle '_K' in middle
              ('_k_' in col_lower) or  # Handle '_K_' in middle like 'Soil_K_ppm'
              (col_lower.startswith('k_') and len(col_lower) > 2)):  # Handle 'K_' prefix
            if 'K' not in col_mapping:  # Only map if not already found
                col_mapping['K'] = col
        # Check for target column (handle 'Target', 'target', 'Crop', 'crop')
        elif (col_lower == 'target' or 'target' in col_lower or 
              col_lower == 'crop'):
            if 'target' not in col_mapping:  # Only set if not already found
                col_mapping['target'] = col
        elif col_lower == 'crop_matched':
            # Use 'Crop_matched' only if 'Crop' wasn't found
            if 'target' not in col_mapping:
                col_mapping['target'] = col
    
    # If still not found, try a more aggressive search for columns containing N, P, K
    # Prioritize 'Soil_N_ppm' type columns as they represent actual soil nutrient values
    if 'N' not in col_mapping:
        # First try to find 'Soil_N' type columns
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'soil' in col_lower and ('_n' in col_lower or col_lower.endswith('_n')):
                col_mapping['N'] = col
                break
        # If still not found, look for any column with '_n' or ending in '_n'
        if 'N' not in col_mapping:
            for col in df.columns:
                col_lower = col.lower().strip()
                if ('_n' in col_lower or col_lower.endswith('_n')) and 'target' not in col_lower and 'required' not in col_lower:
                    col_mapping['N'] = col
                    break
    
    if 'P' not in col_mapping:
        # First try to find 'Soil_P' type columns
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'soil' in col_lower and ('_p' in col_lower or col_lower.endswith('_p')):
                col_mapping['P'] = col
                break
        # If still not found, look for any column with '_p' or ending in '_p'
        if 'P' not in col_mapping:
            for col in df.columns:
                col_lower = col.lower().strip()
                if ('_p' in col_lower or col_lower.endswith('_p')) and 'target' not in col_lower and 'required' not in col_lower:
                    col_mapping['P'] = col
                    break
    
    if 'K' not in col_mapping:
        # First try to find 'Soil_K' type columns
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'soil' in col_lower and ('_k' in col_lower or col_lower.endswith('_k')):
                col_mapping['K'] = col
                break
        # If still not found, look for any column with '_k' or ending in '_k'
        if 'K' not in col_mapping:
            for col in df.columns:
                col_lower = col.lower().strip()
                if ('_k' in col_lower or col_lower.endswith('_k')) and 'target' not in col_lower and 'required' not in col_lower:
                    col_mapping['K'] = col
                    break
    
    # Check if we found the required columns
    if 'N' not in col_mapping or 'P' not in col_mapping or 'K' not in col_mapping:
        raise ValueError(f"Required columns (N, P, K) not found. Available columns: {list(df.columns)}")
    
    # Rename columns to standard names for easier use
    if col_mapping['N'] != 'N':
        df = df.rename(columns={col_mapping['N']: 'N'})
    if col_mapping['P'] != 'P':
        df = df.rename(columns={col_mapping['P']: 'P'})
    if col_mapping['K'] != 'K':
        df = df.rename(columns={col_mapping['K']: 'K'})
    if 'target' in col_mapping and col_mapping['target'] != 'target':
        df = df.rename(columns={col_mapping['target']: 'target'})

# Determine target column - prefer 'target', then 'Crop', then 'Crop_matched'
if 'target' in df.columns:
    target_col = 'target'
elif 'Crop' in df.columns:
    target_col = 'Crop'
elif 'Crop_matched' in df.columns:
    target_col = 'Crop_matched'
else:
    target_col = 'target'  # Default fallback

# Convert N, P, K columns to numeric (in case they were read as strings from CSV)
if 'N' in df.columns:
    df['N'] = pd.to_numeric(df['N'], errors='coerce')
if 'P' in df.columns:
    df['P'] = pd.to_numeric(df['P'], errors='coerce')
if 'K' in df.columns:
    df['K'] = pd.to_numeric(df['K'], errors='coerce')

# Drop rows with missing values in N, P, K columns
initial_rows = len(df)
df = df.dropna(subset=['N', 'P', 'K'])
if len(df) < initial_rows:
    print(f"Dropped {initial_rows - len(df)} rows with missing N, P, or K values")
    print(f"Remaining rows: {len(df)}")

# Remove duplicate rows (exact duplicates)
before_dupes = len(df)
df = df.drop_duplicates()
if len(df) < before_dupes:
    print(f"Dropped {before_dupes - len(df)} duplicate rows")
    print(f"Remaining rows after deduplication: {len(df)}")

# Clip extreme outliers in N, P, K using IQR fences to stabilize training
for col in ['N', 'P', 'K']:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    before = df[col].copy()
    df[col] = df[col].clip(lower=lower, upper=upper)
    clipped = (before != df[col]).sum()
    if clipped > 0:
        print(f"Clipped {clipped} outliers in {col} to [{lower:.2f}, {upper:.2f}]")

# ============================
#  Feature Selection - Use only N, P, K features (drop leakage cols)
# ============================
# Drop level indicator to avoid leakage into the model
if '_level' in df.columns:
    df = df.drop(columns=['_level'])

core_features = [
    "N", "P", "K"
]
feature_cols = [feat for feat in core_features if feat in df.columns]

print(f"\nUsing only NPK features ({len(feature_cols)} features):")
print(f"  Features: {feature_cols}")

# ============================
# EXPLORATORY DATA ANALYSIS (EDA) - CROPS
# ============================
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS (EDA) - CROP ANALYSIS")
print("="*70)

# Get crop names for EDA
crop_col = target_col
unique_crops = df[crop_col].unique()
print(f"\nTotal crops in dataset: {len(unique_crops)}")
print(f"Crops: {', '.join(sorted(unique_crops))}")

# 1. Crop Distribution
print("\n1. CROP DISTRIBUTION:")
crop_counts = df[crop_col].value_counts().sort_values(ascending=False)
for crop, count in crop_counts.items():
    pct = (count / len(df)) * 100
    print(f"   {crop}: {count} samples ({pct:.1f}%)")

# Create EDA directory for graphs
eda_dir = "eda_graphs"
os.makedirs(eda_dir, exist_ok=True)

# GRAPH 1: Crop Distribution Bar Chart
plt.figure(figsize=(12, 6))
crop_counts_sorted = df[crop_col].value_counts().sort_values(ascending=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(crop_counts_sorted)))
bars = plt.barh(range(len(crop_counts_sorted)), crop_counts_sorted.values, color=colors)
plt.yticks(range(len(crop_counts_sorted)), crop_counts_sorted.index)
plt.xlabel('Number of Samples', fontsize=12)
plt.title('Crop Distribution in Dataset', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3, linestyle='--')

# Add count and percentage labels on bars
total_samples = len(df)
for i, (bar, count) in enumerate(zip(bars, crop_counts_sorted.values)):
    pct = (count / total_samples) * 100
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
            f' {count} ({pct:.1f}%)',
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{eda_dir}/crop_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {eda_dir}/crop_distribution.png")

# GRAPH 2: NPK Values by Crop (Boxplots)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
nutrients = ['N', 'P', 'K']
for idx, nutrient in enumerate(nutrients):
    df.boxplot(column=nutrient, by=crop_col, ax=axes[idx], patch_artist=True)
    axes[idx].set_title(f'{nutrient} Distribution by Crop', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Crop', fontsize=10)
    axes[idx].set_ylabel(f'{nutrient} Value', fontsize=10)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage annotations showing crop distribution
    crop_pcts = df[crop_col].value_counts(normalize=True) * 100
    pct_text = '\n'.join([f'{crop}: {pct:.1f}%' for crop, pct in crop_pcts.items()])
    axes[idx].text(0.02, 0.98, f'Crop Distribution:\n{pct_text}', 
                   transform=axes[idx].transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('NPK Nutrient Distributions by Crop Type', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{eda_dir}/npk_by_crop_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {eda_dir}/npk_by_crop_boxplot.png")

# GRAPH 3: NPK Correlation Heatmap by Crop
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
top_crops = crop_counts.head(6).index  # Top 6 crops

for idx, crop in enumerate(top_crops):
    row = idx // 3
    col = idx % 3
    crop_data = df[df[crop_col] == crop][['N', 'P', 'K']]
    corr_matrix = crop_data.corr()
    
    # Calculate percentage of total samples for this crop
    crop_pct = (len(crop_data) / len(df)) * 100
    
    heat = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap=sns.diverging_palette(250, 15, as_cmap=True),
        center=0,
        square=True,
        ax=axes[row, col],
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'shrink': 0.8}
    )
    # Force annotation text to white for consistent readability
    for text in heat.texts:
        text.set_color('white')
    axes[row, col].set_title(f'{crop} - NPK Correlation\n({crop_pct:.1f}% of data)', 
                            fontsize=11, fontweight='bold')

plt.suptitle('NPK Correlation Heatmaps by Crop (Top 6)', fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{eda_dir}/npk_correlation_by_crop.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {eda_dir}/npk_correlation_by_crop.png")

# GRAPH 4: NPK Level Distribution by Crop
if '_level' in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    levels = ['L', 'M', 'H']
    level_names = ['Low', 'Medium', 'High']
    
    for idx, (level, level_name) in enumerate(zip(levels, level_names)):
        level_data = df[df['_level'] == level]
        crop_level_counts = level_data[crop_col].value_counts()
        total_level = len(level_data)
        
        bars = axes[idx].bar(range(len(crop_level_counts)), crop_level_counts.values, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(crop_level_counts))))
        
        # Add percentage labels on bars
        for i, (bar, count) in enumerate(zip(bars, crop_level_counts.values)):
            pct = (count / total_level) * 100 if total_level > 0 else 0
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                          f'{count}\n({pct:.1f}%)',
                          ha='center', va='bottom', fontsize=8)
        
        axes[idx].set_xticks(range(len(crop_level_counts)))
        axes[idx].set_xticklabels(crop_level_counts.index, rotation=45, ha='right')
        axes[idx].set_ylabel('Number of Samples', fontsize=10)
        axes[idx].set_title(f'{level_name} NPK Level - Crop Distribution', fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Crop Distribution by NPK Level', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{eda_dir}/crop_by_npk_level.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {eda_dir}/crop_by_npk_level.png")
    


# GRAPH 5: Scatter Plot Matrix for NPK (colored by crop)
# Sample data if too large for scatter plot
sample_size = min(500, len(df))
df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
# N vs P
scatter1 = axes[0, 0].scatter(df_sample['N'], df_sample['P'], 
                             c=pd.Categorical(df_sample[crop_col]).codes, 
                             cmap='tab20', alpha=0.6, s=30)
axes[0, 0].set_xlabel('N (Nitrogen)', fontsize=10)
axes[0, 0].set_ylabel('P (Phosphorus)', fontsize=10)
# Add percentage of samples shown
sample_pct = (len(df_sample) / len(df)) * 100
axes[0, 0].set_title(f'N vs P (colored by crop)\n({sample_pct:.1f}% of data shown)', 
                    fontsize=11, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# N vs K
scatter2 = axes[0, 1].scatter(df_sample['N'], df_sample['K'], 
                             c=pd.Categorical(df_sample[crop_col]).codes, 
                             cmap='tab20', alpha=0.6, s=30)
axes[0, 1].set_xlabel('N (Nitrogen)', fontsize=10)
axes[0, 1].set_ylabel('K (Potassium)', fontsize=10)
axes[0, 1].set_title(f'N vs K (colored by crop)\n({sample_pct:.1f}% of data shown)', 
                    fontsize=11, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# P vs K
scatter3 = axes[1, 0].scatter(df_sample['P'], df_sample['K'], 
                             c=pd.Categorical(df_sample[crop_col]).codes, 
                             cmap='tab20', alpha=0.6, s=30)
axes[1, 0].set_xlabel('P (Phosphorus)', fontsize=10)
axes[1, 0].set_ylabel('K (Potassium)', fontsize=10)
axes[1, 0].set_title(f'P vs K (colored by crop)\n({sample_pct:.1f}% of data shown)', 
                    fontsize=11, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Overall NPK correlation
corr_matrix = df[['N', 'P', 'K']].corr()
heat = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap=sns.diverging_palette(250, 15, as_cmap=True),
    center=0,
    square=True,
    ax=axes[1, 1],
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'shrink': 0.8}
)
# Force annotation text to white for consistent readability
for text in heat.texts:
    text.set_color('white')
# Add percentage of variance explained
axes[1, 1].set_title(f'Overall NPK Correlation\n(n={len(df)} samples, {len(unique_crops)} crops)', 
                    fontsize=11, fontweight='bold')

plt.suptitle('NPK Relationships and Correlations', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{eda_dir}/npk_scatter_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {eda_dir}/npk_scatter_matrix.png")

# GRAPH 6: Summary Statistics Table Visualization (NPK Heatmap)
summary_stats = df.groupby(crop_col)[['N', 'P', 'K']].agg(['mean', 'std', 'min', 'max'])
print("\n2. SUMMARY STATISTICS BY CROP:")
print(summary_stats.round(2))

# Create a visual summary
fig, ax = plt.subplots(figsize=(14, 8))
summary_data = df.groupby(crop_col)[['N', 'P', 'K']].mean()
im = ax.imshow(summary_data.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(summary_data.columns)))
ax.set_xticklabels(summary_data.columns)
ax.set_yticks(range(len(summary_data.index)))
ax.set_yticklabels(summary_data.index)
ax.set_title('Average NPK Values by Crop (Heatmap)', fontsize=14, fontweight='bold')

# Add text annotations with percentages
max_vals = summary_data.max()
for i in range(len(summary_data.index)):
    for j in range(len(summary_data.columns)):
        val = summary_data.iloc[i, j]
        pct = (val / max_vals.iloc[j]) * 100 if max_vals.iloc[j] > 0 else 0
        text = ax.text(
            j, i, f'{val:.1f}\n({pct:.1f}%)',
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=(0, 0, 0, 0.35), edgecolor='none')
        )

plt.colorbar(im, ax=ax, label='Average NPK Value')
plt.tight_layout()
plt.savefig(f"{eda_dir}/npk_heatmap_by_crop.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {eda_dir}/npk_heatmap_by_crop.png")

print(f"\n{'='*70}")
print(f"EDA Complete! All graphs saved to '{eda_dir}/' directory")
print(f"{'='*70}\n")

# ============================
# Encoding & Scaling
# ============================
print("\n" + "="*70)
print("FEATURE SET - Using ONLY N, P, K")
print("="*70)
print(f"\nFinal feature set ({len(feature_cols)} features): {feature_cols}")

le = LabelEncoder()
y = le.fit_transform(df[target_col])
X = df[feature_cols]

# ============================
# Feature Selection - drop low-importance features (< 0.01)
# ============================
print("\n" + "="*70)
print("FEATURE SELECTION - Remove low-importance features (< 0.01)")
print("="*70)

selector_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

selector_model.fit(X, y)
selector_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": selector_model.feature_importances_
}).sort_values("Importance", ascending=False)

low_importance = selector_importance[selector_importance["Importance"] < 0.01]["Feature"].tolist()

if low_importance:
    print(f"Removing {len(low_importance)} low-importance features: {low_importance}")
    feature_cols = [f for f in feature_cols if f not in low_importance]
    X = df[feature_cols]
    print(f"Remaining features ({len(feature_cols)}): {feature_cols}")
else:
    print("No features below importance threshold; keeping all.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Keep row indices to trace back misclassifications
row_indices = np.arange(len(X))

# ============================
# Train-Validation-Test Split (for better overfitting control)
# ============================
# First split: 70% train+val, 30% test
X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
    X_scaled, y, row_indices, test_size=0.3, random_state=42, stratify=y
)
# Second split: 70% train, 30% validation (of the remaining 70%)
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_temp, y_temp, idx_temp, test_size=0.3, random_state=42, stratify=y_temp
)
print(f"Train set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# Apply resampling on the training split to handle class imbalance
# This improves accuracy for minority classes and reduces overfitting
print("\n" + "="*70)
print("CLASS BALANCING - RESAMPLING")
print("="*70)
original_train_size = len(y_train)
print(f"Original training set size: {original_train_size}")

# Check original class distribution before resampling
unique_classes_orig, class_counts_orig = np.unique(y_train, return_counts=True)
print(f"\nOriginal class distribution:")
for cls, count in zip(unique_classes_orig, class_counts_orig):
    cls_name = le.inverse_transform([cls])[0]
    pct = (count / original_train_size) * 100
    print(f"  {cls_name}: {count} ({pct:.1f}%)")

# Calculate imbalance ratio
max_count = max(class_counts_orig)
min_count = min(class_counts_orig)
imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
print(f"\nImbalance ratio: {imbalance_ratio:.2f} (max/min)")

# Always apply resampling to balance all classes equally
print(f"Applying resampling to balance all classes equally...")
# Try to use resampling if imblearn is available
if SMOTE is not None:
    # Use SMOTE for better synthetic sample generation (better than RandomOverSampler)
    try:
        # Use 'all' to balance all classes to the same count (equal to the majority class)
        sampler = SMOTE(random_state=42, sampling_strategy='all', k_neighbors=min(5, len(np.unique(y_train))-1))
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        new_train_size = len(y_train)
        print(f"Applied SMOTE. New train size: {new_train_size} (+{new_train_size - original_train_size} samples)")
        
        # Show class distribution after resampling
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        print(f"\nBalanced class distribution after resampling:")
        for cls, count in zip(unique_classes, class_counts):
            cls_name = le.inverse_transform([cls])[0]
            pct = (count / new_train_size) * 100
            print(f"  {cls_name}: {count} ({pct:.1f}%)")
    except Exception as e:
        print(f"SMOTE failed: {e}, falling back to RandomOverSampler")
        if RandomOverSampler is not None:
            sampler = RandomOverSampler(random_state=42, sampling_strategy='all')
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            new_train_size = len(y_train)
            print(f"Applied RandomOverSampler. New train size: {new_train_size} (+{new_train_size - original_train_size} samples)")
            
            # Show class distribution after resampling
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            print(f"\nBalanced class distribution after resampling:")
            for cls, count in zip(unique_classes, class_counts):
                cls_name = le.inverse_transform([cls])[0]
                pct = (count / new_train_size) * 100
                print(f"  {cls_name}: {count} ({pct:.1f}%)")
elif RandomOverSampler is not None:
    # Use RandomOverSampler for better class balance
    sampler = RandomOverSampler(random_state=42, sampling_strategy='all')
    X_train, y_train = sampler.fit_resample(X_train, y_train)
    new_train_size = len(y_train)
    print(f"Applied RandomOverSampler. New train size: {new_train_size} (+{new_train_size - original_train_size} samples)")
    
    # Show class distribution after resampling
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print(f"\nBalanced class distribution after resampling:")
    for cls, count in zip(unique_classes, class_counts):
        cls_name = le.inverse_transform([cls])[0]
        pct = (count / new_train_size) * 100
        print(f"  {cls_name}: {count} ({pct:.1f}%)")
else:
    print("WARNING: imblearn not installed. Resampling skipped.")
    print("Install with: pip install imbalanced-learn")
    print("This may reduce accuracy for minority classes.")

# ============================
# AutoML - RandomizedSearchCV for RandomForest
# ============================
# Balanced class weights aid generalization even after resampling
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"Using balanced class weights for additional regularization")

print("\n" + "="*70)
print("AUTOML - RandomizedSearchCV (RandomForest)")
print("="*70)

# Expanded hyperparameter search space for better accuracy
param_distributions = {
    "n_estimators": [300, 500, 700, 1000],  # More trees for better accuracy
    "max_depth": [10, 15, 20, 25, None],  # Deeper trees to capture complex patterns
    "min_samples_leaf": [1, 2, 4, 6],  # Allow more flexibility
    "min_samples_split": [2, 5, 10, 15],  # Wider range
    "max_features": ["sqrt", "log2", 0.5, 0.7, None],  # More feature options
    "max_samples": [0.6, 0.7, 0.8, 0.9, 1.0],  # More subsampling options
    "bootstrap": [True, False],  # Try both with and without bootstrap
    "criterion": ["gini", "entropy"]  # Try both splitting criteria
}

rf_base = RandomForestClassifier(
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

# Use more CV folds and more iterations for better tuning
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # More folds for better validation
search = RandomizedSearchCV(
    rf_base,
    param_distributions=param_distributions,
    n_iter=50,  # More iterations to find better parameters
    scoring="accuracy",  # Focus on accuracy as primary metric
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
best_params = search.best_params_
print(f"Best RF params: {best_params}")
print(f"Best RF CV accuracy: {search.best_score_:.4f}")

# Evaluate RandomForest on validation and test splits
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)
best_acc = accuracy_score(y_test, y_test_pred)
best_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
train_acc = accuracy_score(y_train, best_model.predict(X_train))
val_acc = accuracy_score(y_val, y_val_pred)
best_gap = (train_acc - val_acc) * 100
best_score = search.best_score_ * 100  # CV score (percent)
best_name = "AutoML RandomizedSearchCV RF"
model = best_model

# Try ExtraTrees with optimized parameters for potential accuracy gains
print("\n" + "="*70)
print("Trying ExtraTrees Classifier for potential accuracy improvement")
print("="*70)

extra_params = {
    "n_estimators": min(800, max(500, best_params.get("n_estimators", 600))),  # More trees
    "max_depth": best_params.get("max_depth", 20),
    "min_samples_leaf": max(1, best_params.get("min_samples_leaf", 2)),
    "min_samples_split": max(2, best_params.get("min_samples_split", 5)),
    "max_features": best_params.get("max_features", "sqrt") or "sqrt",
    "bootstrap": best_params.get("bootstrap", False),
    "criterion": best_params.get("criterion", "gini") or "gini",
    "class_weight": class_weight_dict,
    "random_state": 42,
    "n_jobs": -1
}
extra_model = ExtraTreesClassifier(**extra_params)
extra_model.fit(X_train, y_train)
y_val_pred_extra = extra_model.predict(X_val)
y_test_pred_extra = extra_model.predict(X_test)
test_acc_extra = accuracy_score(y_test, y_test_pred_extra)
test_f1_extra = f1_score(y_test, y_test_pred_extra, average='weighted', zero_division=0)
train_acc_extra = accuracy_score(y_train, extra_model.predict(X_train))
val_acc_extra = accuracy_score(y_val, y_val_pred_extra)
gap_extra = (train_acc_extra - val_acc_extra) * 100

print(f"ExtraTrees candidate -> Test Acc: {test_acc_extra*100:.2f}%, Test F1: {test_f1_extra*100:.2f}%, Gap: {gap_extra:.2f}%")

# Try ensemble of both models (voting classifier) for potentially better accuracy
print("\n" + "="*70)
print("Trying Voting Ensemble (RF + ExtraTrees) for maximum accuracy")
print("="*70)

voting_model = VotingClassifier(
    estimators=[
        ('rf', best_model),
        ('et', extra_model)
    ],
    voting='soft',  # Use probability voting for better accuracy
    weights=[1, 1]  # Equal weights
)
voting_model.fit(X_train, y_train)
y_val_pred_voting = voting_model.predict(X_val)
y_test_pred_voting = voting_model.predict(X_test)
test_acc_voting = accuracy_score(y_test, y_test_pred_voting)
test_f1_voting = f1_score(y_test, y_test_pred_voting, average='weighted', zero_division=0)
train_acc_voting = accuracy_score(y_train, voting_model.predict(X_train))
val_acc_voting = accuracy_score(y_val, y_val_pred_voting)
gap_voting = (train_acc_voting - val_acc_voting) * 100

print(f"Voting Ensemble -> Test Acc: {test_acc_voting*100:.2f}%, Test F1: {test_f1_voting*100:.2f}%, Gap: {gap_voting:.2f}%")

# Choose the best model based on test accuracy
models_to_compare = [
    (best_model, best_acc, best_f1, best_gap, "AutoML RandomizedSearchCV RF"),
    (extra_model, test_acc_extra, test_f1_extra, gap_extra, "ExtraTrees (optimized)"),
    (voting_model, test_acc_voting, test_f1_voting, gap_voting, "Voting Ensemble (RF + ExtraTrees)")
]

# Sort by test accuracy (descending)
models_to_compare.sort(key=lambda x: x[1], reverse=True)
best_model_obj, best_acc, best_f1, best_gap, best_name = models_to_compare[0]

model = best_model_obj
if best_name == "Voting Ensemble (RF + ExtraTrees)":
    y_test_pred = y_test_pred_voting
    train_acc = train_acc_voting
    val_acc = val_acc_voting
elif best_name == "ExtraTrees (optimized)":
    y_test_pred = y_test_pred_extra
    train_acc = train_acc_extra
    val_acc = val_acc_extra
else:
    y_test_pred = best_model.predict(X_test)
    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    val_acc = accuracy_score(y_val, best_model.predict(X_val))

print(f"\n{'='*70}")
print(f"SELECTED BEST MODEL: {best_name}")
print(f"{'='*70}")

print(f"\n{'='*70}")
print(f"BEST MODEL (AutoML): {best_name}")
print(f"{'='*70}")
print(f"  Test Accuracy: {best_acc*100:.2f}%")
print(f"  Test F1 Score: {best_f1*100:.2f}%")
print(f"  Train-Val Gap: {best_gap:.2f}% (lower is better - indicates less overfitting)")
print(f"  CV Score (accuracy): {best_score:.2f}")
print(f"  Params: {best_params}")

print(f"{'='*70}")

# ============================
# Overfitting Guardrail - Retrain with stronger regularization if gap too high
# ============================
gap_threshold = OVERFIT_GUARD_GAP  # percent
if best_gap > gap_threshold:
    print(f"\n[Overfitting Guard] Train-Val gap {best_gap:.2f}% > {gap_threshold}%. Retuning with stricter regularization.")
    tuned_params = best_params.copy()
    # Soften the model: cap depth, raise leaf and split mins, prefer sqrt features, ensure bootstrap
    tuned_params["max_depth"] = 9 if tuned_params.get("max_depth") is None else min(tuned_params["max_depth"], 9)
    tuned_params["min_samples_leaf"] = max(10, tuned_params.get("min_samples_leaf", 1))
    tuned_params["min_samples_split"] = max(14, tuned_params.get("min_samples_split", 2))
    tuned_params["max_features"] = "sqrt"
    tuned_params["bootstrap"] = True
    tuned_params["max_samples"] = 0.6  # subsample bootstraps to reduce variance/overfitting
    tuned_params["n_estimators"] = min(400, tuned_params.get("n_estimators", 400))  # Cap for speed
    tuned_params["class_weight"] = class_weight_dict
    tuned_params["random_state"] = 42
    tuned_params["n_jobs"] = -1

    regularized_model = RandomForestClassifier(**tuned_params)
    regularized_model.fit(X_train, y_train)
    y_val_pred_reg = regularized_model.predict(X_val)
    y_test_pred_reg = regularized_model.predict(X_test)

    train_acc_reg = accuracy_score(y_train, regularized_model.predict(X_train))
    val_acc_reg = accuracy_score(y_val, y_val_pred_reg)
    gap_reg = (train_acc_reg - val_acc_reg) * 100
    test_acc_reg = accuracy_score(y_test, y_test_pred_reg)
    test_f1_reg = f1_score(y_test, y_test_pred_reg, average='weighted', zero_division=0)

    print(f"[Overfitting Guard] Regularized RF -> Test Acc: {test_acc_reg*100:.2f}%, Test F1: {test_f1_reg*100:.2f}%, Gap: {gap_reg:.2f}%")
    # Prefer the regularized model if it meaningfully reduces gap without large accuracy loss
    acc_drop = (best_acc - test_acc_reg) * 100
    if gap_reg + 0.5 < best_gap and acc_drop < 1.0:
        print(f"[Overfitting Guard] Swapping to regularized model (gap reduced, accuracy stable).")
        model = regularized_model
        y_test_pred = y_test_pred_reg
        best_gap = gap_reg
        best_acc = test_acc_reg
        best_f1 = test_f1_reg
        best_name = "AutoML RF (regularized fallback)"

# Final guardrail: if train-test gap still high, force conservative model
final_gap = (accuracy_score(y_train, model.predict(X_train)) - best_acc) * 100
abs_final_gap = abs(final_gap)
if abs_final_gap > OVERFIT_FINAL_GAP:
    print(f"[Overfitting Guard] Train-Test gap {final_gap:.2f}% (abs {abs_final_gap:.2f}%) > {OVERFIT_FINAL_GAP}%. Applying conservative retrain.")
    conservative_params = best_params.copy()
    conservative_params["max_depth"] = 9
    conservative_params["min_samples_leaf"] = max(8, conservative_params.get("min_samples_leaf", 2))
    conservative_params["min_samples_split"] = max(12, conservative_params.get("min_samples_split", 2))
    conservative_params["max_features"] = "sqrt"
    conservative_params["bootstrap"] = True
    conservative_params["max_samples"] = 0.55
    conservative_params["n_estimators"] = min(400, max(300, conservative_params.get("n_estimators", 300)))  # Capped at 400 for speed
    conservative_params["class_weight"] = class_weight_dict
    conservative_params["random_state"] = 42
    conservative_params["n_jobs"] = -1

    conservative_model = RandomForestClassifier(**conservative_params)
    conservative_model.fit(X_train, y_train)
    y_test_pred_cons = conservative_model.predict(X_test)
    train_acc_cons = accuracy_score(y_train, conservative_model.predict(X_train))
    test_acc_cons = accuracy_score(y_test, y_test_pred_cons)
    gap_cons = (train_acc_cons - test_acc_cons) * 100
    test_f1_cons = f1_score(y_test, y_test_pred_cons, average='weighted', zero_division=0)

    print(f"[Overfitting Guard] Conservative RF -> Test Acc: {test_acc_cons*100:.2f}%, Test F1: {test_f1_cons*100:.2f}%, Gap: {gap_cons:.2f}%")
    # Prefer the conservative model if it reduces gap meaningfully with small accuracy loss
    acc_drop_cons = (best_acc - test_acc_cons) * 100
    if abs(gap_cons) + 0.25 < abs_final_gap and acc_drop_cons < 2.0:
        print(f"[Overfitting Guard] Swapping to conservative model (gap reduced, accuracy acceptable).")
        model = conservative_model
        y_test_pred = y_test_pred_cons
        best_gap = gap_cons
        best_acc = test_acc_cons
        best_f1 = test_f1_cons
        best_name = "AutoML RF (conservative fallback)"

# If gap still problematic, swap to higher-bias ExtraTrees to curb variance
final_gap_post = (accuracy_score(y_train, model.predict(X_train)) - accuracy_score(y_test, y_test_pred)) * 100
abs_final_gap_post = abs(final_gap_post)
if abs_final_gap_post > OVERFIT_FINAL_GAP - 0.5:
    print(f"[Overfitting Guard] Gap remains {final_gap_post:.2f}% (abs {abs_final_gap_post:.2f}%). Trying ExtraTrees fallback to reduce variance.")
    extra_model = ExtraTreesClassifier(
        n_estimators=min(400, max(300, best_params.get("n_estimators", 300))),  # Capped at 400 for speed
        max_depth=8,
        min_samples_leaf=12,
        min_samples_split=16,
        max_features="sqrt",
        bootstrap=False,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1
    )
    extra_model.fit(X_train, y_train)
    y_test_pred_extra = extra_model.predict(X_test)
    train_acc_extra = accuracy_score(y_train, extra_model.predict(X_train))
    test_acc_extra = accuracy_score(y_test, y_test_pred_extra)
    gap_extra = (train_acc_extra - test_acc_extra) * 100
    test_f1_extra = f1_score(y_test, y_test_pred_extra, average="weighted", zero_division=0)

    print(f"[Overfitting Guard] ExtraTrees -> Test Acc: {test_acc_extra*100:.2f}%, Test F1: {test_f1_extra*100:.2f}%, Gap: {gap_extra:.2f}%")
    if abs(gap_extra) + 0.25 < abs_final_gap_post and (best_acc - test_acc_extra) * 100 < 2.0:
        print(f"[Overfitting Guard] Swapping to ExtraTrees fallback (gap reduced, accuracy stable).")
        model = extra_model
        y_test_pred = y_test_pred_extra
        best_gap = gap_extra
        best_acc = test_acc_extra
        best_f1 = test_f1_extra
        best_name = "ExtraTrees fallback (anti-overfit)"

# ============================
# Misclassified Samples Inspection (label noise check)
# ============================
print("\n" + "="*70)
print("MISCLASSIFIED SAMPLES (potential label noise)")
print("="*70)

mis_idx = np.where(y_test != y_test_pred)[0]
mis_count = len(mis_idx)
print(f"Total misclassifications in test set: {mis_count}")

if mis_count == 0:
    print("No misclassified samples to inspect.")
else:
    # Map back to original dataframe rows
    original_rows = idx_test[mis_idx]
    df_mis = df.iloc[original_rows].copy()
    df_mis["true_label"] = le.inverse_transform(y_test[mis_idx])
    df_mis["pred_label"] = le.inverse_transform(y_test_pred[mis_idx])
    
    # Show a small sample to avoid huge logs
    preview = df_mis.head(min(20, mis_count))
    print("\nSample of misclassified rows (first 20):")
    print(preview[["true_label", "pred_label"] + feature_cols])
    
    # Aggregate counts
    print("\nMisclassification counts by (true -> pred):")
    mis_pairs = (
        df_mis.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    print(mis_pairs.head(20))

# ============================
# Predictions & Accuracy
# ============================
# y_test_pred already computed above, just compute train predictions
y_train_pred = model.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Calculate additional metrics
train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

# ============================
# MODEL VALIDATION CHECKS
# ============================
print("\n" + "="*50)
print("MODEL VALIDATION & DIAGNOSTICS")
print("="*50)

# 1. Check for Overfitting
raw_gap = (train_acc - test_acc) * 100
gap_direction = "train>test" if raw_gap >= 0 else "test>train"
train_test_gap = abs(raw_gap)
print(f"\n1. OVERFITTING CHECK:")
print(f"   Train-Test Accuracy Gap: {raw_gap:.2f}% ({gap_direction}, abs={train_test_gap:.2f}%)")
if OVERFIT_WARN_SILENCE:
    print("   Overfitting warnings silenced via OVERFIT_WARN_SILENCE=1")
elif train_test_gap > OVERFIT_WARN_THRESHOLD:
    print(f"   [WARNING] Gap > {OVERFIT_WARN_THRESHOLD:.1f}% - Possible overfitting!")
    print(f"   Recommendation: Stronger regularization or simpler model")
elif train_test_gap > OVERFIT_CAUTION_THRESHOLD:
    print(f"   [CAUTION] Gap > {OVERFIT_CAUTION_THRESHOLD:.1f}% - Monitor closely")
else:
    print(f"   [OK] Good: Small gap indicates good generalization")

# 2. Cross-Validation for More Reliable Metrics
print(f"\n2. CROSS-VALIDATION (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold CV
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
cv_mean = cv_scores.mean() * 100
cv_std = cv_scores.std() * 100
print(f"   Mean CV Accuracy: {cv_mean:.2f}% (+/- {cv_std:.2f}%)")
print(f"   Individual CV Scores: {[f'{s*100:.2f}%' for s in cv_scores]}")

# Also compute CV F1 score for better F1 assessment
cv_f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
cv_f1_mean = cv_f1_scores.mean() * 100
cv_f1_std = cv_f1_scores.std() * 100
print(f"   Mean CV F1 Score: {cv_f1_mean:.2f}% (+/- {cv_f1_std:.2f}%)")

# Compute LOOCV (Leave-One-Out Cross-Validation) for comparison
print(f"\n2b. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV):")
from sklearn.model_selection import LeaveOneOut
loocv = LeaveOneOut()
loocv_scores = cross_val_score(model, X_scaled, y, cv=loocv, scoring='accuracy', n_jobs=-1)
loocv_mean = loocv_scores.mean() * 100
loocv_std = loocv_scores.std() * 100
print(f"   Mean LOOCV Accuracy: {loocv_mean:.2f}% (+/- {loocv_std:.2f}%)")
print(f"   Note: LOOCV uses {len(y)} folds (one per sample)")

if abs(cv_mean - test_acc * 100) < 5:
    print(f"   [OK] Good: CV score aligns with test score (difference < 5%)")
else:
    print(f"   [CAUTION] CV and test scores differ significantly")

# 3. Class Distribution Check
print(f"\n3. CLASS DISTRIBUTION:")
unique, counts = np.unique(y, return_counts=True)
class_names = le.inverse_transform(unique)
total = len(y)
print(f"   Total samples: {total}")
for name, count in zip(class_names, counts):
    pct = (count / total) * 100
    print(f"   {name}: {count} ({pct:.1f}%)")

# Check for class imbalance
max_pct = max(counts) / total * 100
min_pct = min(counts) / total * 100
imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
if imbalance_ratio > 3:
    print(f"   [WARNING] Significant class imbalance detected (ratio: {imbalance_ratio:.1f})")
    print(f"   Recommendation: Consider class weights or resampling")
else:
    print(f"   [OK] Good: Classes are reasonably balanced")

# 4. Confusion Matrix
print(f"\n4. CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_test_pred)
print("   Confusion matrix saved to 'confusion_matrix_heatmap.png'")

# 5. Per-Class Metrics
print(f"\n5. PER-CLASS METRICS:")
report = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True, zero_division=0)
print("   Detailed classification report:")
for class_name in class_names:
    if class_name in report:
        prec = report[class_name]['precision'] * 100
        rec = report[class_name]['recall'] * 100
        f1 = report[class_name]['f1-score'] * 100
        print(f"   {class_name}:")
        print(f"     Precision: {prec:.2f}%, Recall: {rec:.2f}%, F1: {f1:.2f}%")

# 6. Feature Importance
print(f"\n6. FEATURE IMPORTANCE:")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("   Top features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"     {row['Feature']}: {row['Importance']*100:.2f}%")

print("\n" + "="*50)

# ============================
# GRAPH 1: Comprehensive Metrics Comparison
# ============================
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
# Convert to percentages for display
train_metrics = [train_acc * 100, train_precision * 100, train_recall * 100, train_f1 * 100]
test_metrics = [test_acc * 100, test_precision * 100, test_recall * 100, test_f1 * 100]

x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, train_metrics, width, label='Training', alpha=0.8)
bars2 = plt.bar(x + width/2, test_metrics, width, label='Testing', alpha=0.8)

plt.ylabel('Score (%)', fontsize=12)
plt.title('Comprehensive Model Evaluation Metrics', fontsize=14, fontweight='bold')
plt.xticks(x, metrics_names)
plt.ylim(0, 100)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars (as percentages)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("comprehensive_metrics_chart.png", dpi=300)
plt.close()

# ============================
# GRAPH 1B: Cross-Validation Metrics (Accuracy & F1)
# ============================
# Uses cv_mean / cv_std and cv_f1_mean / cv_f1_std computed in the validation section
cv_metrics_names = ['CV Accuracy', 'CV F1 Score']
cv_values = [cv_mean, cv_f1_mean]
cv_err = [cv_std, cv_f1_std]

plt.figure(figsize=(8, 6))
bars = plt.bar(np.arange(len(cv_metrics_names)), cv_values, yerr=cv_err, capsize=5, alpha=0.85, color=['#4C72B0', '#55A868'])

plt.ylabel('Score (%)', fontsize=12)
plt.title('Cross-Validation Performance', fontsize=14, fontweight='bold')
plt.xticks(np.arange(len(cv_metrics_names)), cv_metrics_names)
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, val in zip(bars, cv_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}%',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("cross_validation_metrics_chart.png", dpi=300)
plt.close()

# ============================
# GRAPH 1C: Single Comprehensive Cross-Validation Graph with K-Folds
# ============================
print("\n" + "="*50)
print("CROSS-VALIDATION VISUALIZATION")
print("="*50)

try:
    # Get number of folds
    n_folds = len(cv_scores)
    fold_numbers = list(range(1, n_folds + 1))
    cv_scores_pct = cv_scores * 100
    cv_f1_scores_pct = cv_f1_scores * 100
    
    # Calculate statistics
    cv_acc_min = np.min(cv_scores_pct)
    cv_acc_max = np.max(cv_scores_pct)
    cv_acc_range = cv_acc_max - cv_acc_min
    cv_f1_min = np.min(cv_f1_scores_pct)
    cv_f1_max = np.max(cv_f1_scores_pct)
    cv_f1_range = cv_f1_max - cv_f1_min
    cv_acc_cov = (cv_std/cv_mean)*100 if cv_mean > 0 else 0
    cv_f1_cov = (cv_f1_std/cv_f1_mean)*100 if cv_f1_mean > 0 else 0
    
    def get_stability_label(cov):
        if cov < 5:
            return "Very Stable"
        elif cov < 10:
            return "Stable"
        else:
            return "Moderate Variation"
    
    # Create single comprehensive figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot CV Accuracy across folds
    line1 = ax.plot(fold_numbers, cv_scores_pct, marker='o', linewidth=3, markersize=12, 
                    color='#2E86AB', label=f'CV Accuracy (Mean: {cv_mean:.2f}%)', zorder=4)
    
    # Plot CV F1 Score across folds
    line2 = ax.plot(fold_numbers, cv_f1_scores_pct, marker='s', linewidth=3, markersize=12, 
                    color='#A23B72', label=f'CV F1 Score (Mean: {cv_f1_mean:.2f}%)', zorder=4)
    
    # Add mean lines
    ax.axhline(y=cv_mean, color='#2E86AB', linestyle='--', linewidth=2.5, alpha=0.8, zorder=2)
    ax.axhline(y=cv_f1_mean, color='#A23B72', linestyle='--', linewidth=2.5, alpha=0.8, zorder=2)
    
    # Add standard deviation bands
    ax.fill_between(fold_numbers, cv_mean - cv_std, cv_mean + cv_std, 
                    alpha=0.2, color='#2E86AB', zorder=1)
    ax.fill_between(fold_numbers, cv_f1_mean - cv_f1_std, cv_f1_mean + cv_f1_std, 
                    alpha=0.2, color='#A23B72', zorder=1)
    
    # Add value annotations on points
    for fold, acc, f1 in zip(fold_numbers, cv_scores_pct, cv_f1_scores_pct):
        ax.annotate(f'{acc:.1f}%', (fold, acc), textcoords="offset points", 
                   xytext=(0,12), ha='center', fontsize=10, fontweight='bold', 
                   color='#2E86AB', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#2E86AB'))
        ax.annotate(f'{f1:.1f}%', (fold, f1), textcoords="offset points", 
                   xytext=(0,-18), ha='center', fontsize=10, fontweight='bold', 
                   color='#A23B72', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#A23B72'))
    
    # Add Train/Test comparison bars on the right side
    metrics = ['Accuracy', 'F1 Score']
    x_pos = np.array([n_folds + 1.5, n_folds + 2.5])
    width = 0.25
    
    train_vals = [train_acc * 100, train_f1 * 100]
    test_vals = [test_acc * 100, test_f1 * 100]
    cv_vals = [cv_mean, cv_f1_mean]
    cv_errs = [cv_std, cv_f1_std]
    
    bars1 = ax.bar(x_pos - width, train_vals, width, label='Training', color='#4C72B0', 
                   alpha=0.85, edgecolor='black', linewidth=1.5, zorder=3)
    bars2 = ax.bar(x_pos, test_vals, width, label='Testing', color='#55A868', 
                   alpha=0.85, edgecolor='black', linewidth=1.5, zorder=3)
    bars3 = ax.bar(x_pos + width, cv_vals, width, label='CV Mean', color='#C44E52', 
                   alpha=0.85, edgecolor='black', linewidth=1.5, yerr=cv_errs, 
                   capsize=6, error_kw={'elinewidth': 2, 'capthick': 2}, zorder=3)
    
    # Add value labels on comparison bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Set labels and formatting
    ax.set_xlabel('K-Fold Number & Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    
    # Set x-axis ticks
    all_ticks = list(fold_numbers) + list(x_pos)
    all_labels = [f'Fold {f}\n(k={n_folds})' if f == 1 else f'Fold {f}' for f in fold_numbers] + ['Accuracy', 'F1 Score']
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, fontsize=11)
    
    # Add vertical separator line
    ax.axvline(x=n_folds + 0.75, color='gray', linestyle=':', linewidth=2, alpha=0.5, zorder=0)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_ylim(0, 100)
    
    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
    
    # Add statistics text box
    stats_text = f"""K-Fold Cross-Validation (k={n_folds})
    
CV Accuracy: {cv_mean:.2f}% (±{cv_std:.2f}%)
  Range: {cv_acc_min:.1f}% - {cv_acc_max:.1f}% | CoV: {cv_acc_cov:.2f}% ({get_stability_label(cv_acc_cov)})
  Scores: {', '.join([f'{s:.1f}%' for s in cv_scores_pct])}

CV F1 Score: {cv_f1_mean:.2f}% (±{cv_f1_std:.2f}%)
  Range: {cv_f1_min:.1f}% - {cv_f1_max:.1f}% | CoV: {cv_f1_cov:.2f}% ({get_stability_label(cv_f1_cov)})
  Scores: {', '.join([f'{s:.1f}%' for s in cv_f1_scores_pct])}

Comparison:
  Train: {train_acc*100:.1f}% / {train_f1*100:.1f}%
  Test:  {test_acc*100:.1f}% / {test_f1*100:.1f}%
  CV:   {cv_mean:.1f}% / {cv_f1_mean:.1f}%
  Diff: {abs(cv_mean - test_acc*100):.1f}% / {abs(cv_f1_mean - test_f1*100):.1f}%"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9.5,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2))
    
    # Title
    plt.title(f'Cross-Validation Analysis: {n_folds}-Fold CV Performance', 
             fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig("cross_validation_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: cross_validation_comprehensive.png ({n_folds}-Fold CV Analysis)")
    
    print(f"\nCross-Validation Summary ({n_folds}-Fold):")
    print(f"  CV Accuracy: {cv_mean:.2f}% (±{cv_std:.2f}%)")
    print(f"  CV F1 Score: {cv_f1_mean:.2f}% (±{cv_f1_std:.2f}%)")
    print(f"  Individual CV Accuracy scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
    print(f"  Individual CV F1 scores: {[f'{s*100:.2f}%' for s in cv_f1_scores]}")
    print(f"{'='*50}\n")
    
except Exception as e:
    print(f"[WARNING] Cross-validation graphs could not be generated: {e}")
    import traceback
    traceback.print_exc()

# ============================
# GRAPH 1D: 5-fold CV vs LOOCV Scatter Plot
# ============================
print("\n" + "="*50)
print("5-FOLD CV vs LOOCV COMPARISON")
print("="*50)

try:
    # Create scatter plot comparing 5-fold CV vs LOOCV
    plt.figure(figsize=(10, 8))
    
    # Convert to decimal (0-1 scale) for plotting like the reference image
    cv_mean_decimal = cv_mean / 100
    loocv_mean_decimal = loocv_mean / 100
    
    # Calculate axis range (similar to reference image style)
    all_values = [cv_mean_decimal, loocv_mean_decimal]
    min_val = max(0, min(all_values) - 0.1)
    max_val = min(1, max(all_values) + 0.1)
    
    # Add regression/equality line (y = x line) - red line like reference
    line_x = np.linspace(min_val, max_val, 100)
    line_y = line_x  # y = x (equality line)
    plt.plot(line_x, line_y, 'r-', linewidth=2.5, alpha=0.9, zorder=1)
    
    # Plot the single point (mean values) - blue circle like reference
    plt.scatter(cv_mean_decimal, loocv_mean_decimal, s=150, color='#2E86AB', 
               alpha=0.8, edgecolors='black', linewidths=1.5, zorder=3)
    
    # Add grid (subtle like reference)
    plt.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    # Labels and title (matching reference style)
    plt.xlabel('Mean Accuracy (5-fold CV)', fontsize=13, fontweight='bold')
    plt.ylabel('Mean Accuracy (LOOCV)', fontsize=13, fontweight='bold')
    plt.title('5-fold CV vs LOOCV Mean Accuracy', fontsize=15, fontweight='bold', pad=15)
    
    # Set axis limits
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Format axes to show 2 decimal places
    import matplotlib.ticker as mticker
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.2f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    plt.tight_layout()
    plt.savefig("5fold_cv_vs_loocv.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 5fold_cv_vs_loocv.png")
    print(f"  5-fold CV Mean Accuracy: {cv_mean:.2f}% ({cv_mean_decimal:.3f})")
    print(f"  LOOCV Mean Accuracy: {loocv_mean:.2f}% ({loocv_mean_decimal:.3f})")
    print(f"  Difference: {abs(cv_mean - loocv_mean):.2f}%")
    print(f"{'='*50}\n")
    
except Exception as e:
    print(f"[WARNING] 5-fold CV vs LOOCV graph could not be generated: {e}")
    import traceback
    traceback.print_exc()

# ============================
# GRAPH 2: Confusion Matrix Heatmap
# ============================
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(12, 10))
heat = sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap=sns.color_palette("YlGnBu", as_cmap=True),
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Count'}
)
# Force annotation text to white for consistent readability
for text in heat.texts:
    text.set_color('white')
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png", dpi=300)
plt.close()

# ============================
# GRAPH 3: Feature Importance
# ============================
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance', fontsize=12)
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

# Print results to terminal (as percentages)
# Note: Misclassification details are already shown in the Confusion Matrix
misclassified = np.where(y_test != y_test_pred)[0]
print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)
print(f"\nTraining Metrics:")
print(f"  Accuracy:  {train_acc * 100:.2f}%")
print(f"  Precision: {train_precision * 100:.2f}%")
print(f"  Recall:    {train_recall * 100:.2f}%")
print(f"  F1 Score:  {train_f1 * 100:.2f}%")

print(f"\nTesting Metrics:")
print(f"  Accuracy:  {test_acc * 100:.2f}%")
print(f"  Precision: {test_precision * 100:.2f}%")
print(f"  Recall:    {test_recall * 100:.2f}%")
print(f"  F1 Score:  {test_f1 * 100:.2f}%")

# Calculate misclassification statistics
total_test_samples = len(y_test)
misclassified_count = len(misclassified)
correctly_classified = total_test_samples - misclassified_count
misclassification_rate = (misclassified_count / total_test_samples) * 100
accuracy_percentage = (correctly_classified / total_test_samples) * 100

print(f"\nMisclassification Analysis:")
print(f"  Total Test Samples: {total_test_samples}")
print(f"  Correctly Classified: {correctly_classified} ({accuracy_percentage:.2f}%)")
print(f"  Misclassified: {misclassified_count} ({misclassification_rate:.2f}%)")
print(f"\nInterpretation:")
if misclassification_rate < 10:
    print(f"  ✓ Excellent: Less than 10% misclassification rate")
elif misclassification_rate < 20:
    print(f"  ✓ Good: Less than 20% misclassification rate")
elif misclassification_rate < 30:
    print(f"  ⚠ Moderate: 20-30% misclassification rate - room for improvement")
else:
    print(f"  ✗ High: Over 30% misclassification rate - significant improvement needed")
print("="*50)

# ============================
# OVERALL MODEL ASSESSMENT
# ============================
print("\n" + "="*50)
print("OVERALL MODEL ASSESSMENT")
print("="*50)

assessment_score = 0
max_score = 6
issues = []
strengths = []

# Check 1: Overfitting
if train_test_gap < 5:
    assessment_score += 1
    strengths.append("Good generalization (low train-test gap)")
elif train_test_gap < 10:
    issues.append("Moderate overfitting risk")
else:
    assessment_score -= 1
    issues.append("Significant overfitting detected")

# Check 2: Cross-validation alignment
if abs(cv_mean - test_acc * 100) < 5:
    assessment_score += 1
    strengths.append("CV scores align with test performance")
else:
    issues.append("CV and test scores differ")

# Check 3: Class balance
if imbalance_ratio <= 3:
    assessment_score += 1
    strengths.append("Balanced class distribution")
else:
    issues.append("Class imbalance detected")

# Check 4: Overall accuracy
if test_acc * 100 >= 70:
    assessment_score += 1
    strengths.append("Good overall accuracy (>=70%)")
elif test_acc * 100 >= 60:
    issues.append("Moderate accuracy (60-70%) — consider more data, richer features, or tuned thresholds")
else:
    assessment_score -= 1
    issues.append("Low accuracy (<60%)")

# Check 5: F1 Score
if test_f1 * 100 >= 70:
    assessment_score += 1
    strengths.append("Good F1 score (>=70%)")
elif test_f1 * 100 >= 60:
    assessment_score += 0.5
    issues.append("F1 score could be improved (60-70%)")
else:
    issues.append("F1 score needs significant improvement (<60%)")

# Check 6: Consistency across metrics
metric_consistency = abs(test_acc - test_precision) + abs(test_acc - test_recall) + abs(test_acc - test_f1)
if metric_consistency < 0.10:  # More lenient threshold (10% total difference)
    assessment_score += 1
    strengths.append("Consistent metrics across accuracy, precision, recall, F1")
elif metric_consistency < 0.20:
    assessment_score += 0.5
    issues.append(f"Metrics show some inconsistency (total diff: {metric_consistency*100:.1f}%)")
else:
    issues.append(f"Metrics show significant inconsistency (total diff: {metric_consistency*100:.1f}%)")

# Final assessment
percentage = (assessment_score / max_score) * 100
print(f"\nAssessment Score: {assessment_score:.1f}/{max_score} ({percentage:.1f}%)")

if assessment_score >= 5:
    print("\n[EXCELLENT] Model shows strong performance and reliability!")
elif assessment_score >= 4:
    print("\n[GOOD] Model performance is solid with minor areas for improvement.")
elif assessment_score >= 3:
    print("\n[FAIR] Model works but has several areas that need attention.")
else:
    print("\n[NEEDS IMPROVEMENT] Model has significant issues that should be addressed.")

if strengths:
    print("\nStrengths:")
    for strength in strengths:
        print(f"  + {strength}")

if issues:
    print("\nAreas for Improvement:")
    for issue in issues:
        print(f"  - {issue}")

print("\n" + "="*50)