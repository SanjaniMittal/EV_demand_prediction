"""
Dataset Inspection Script
Analyzes data/Electric_Vehicle_Population_Data.csv for columns, shapes, data types, missing values, and distributions.
"""

import pandas as pd
import numpy as np

def inspect():
    csv_path = 'data/Electric_Vehicle_Population_Data.csv'
    df = pd.read_csv(csv_path)
    
    print("=== DATASET OVERVIEW ===")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {df.shape[1]}")
    
    print("\n=== COLUMNS & DATA TYPES ===")
    for col, dtype in df.dtypes.items():
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f" - {col} ({dtype}): {null_count:,} missing ({null_pct:.2f}%)")
        
    print("\n=== SAMPLE RECORDS ===")
    print(df[['VIN (1-10)', 'County', 'City', 'State', 'Model Year', 'Make', 'Model', 'Electric Vehicle Type', 'Electric Range']].head(3))
    
    print("\n=== MODEL YEARS (Tail 15) ===")
    years = df['Model Year'].value_counts().sort_index()
    print(years.tail(15))
    
    print("\n=== ELECTRIC VEHICLE TYPES ===")
    ev_types = df['Electric Vehicle Type'].value_counts()
    for ev_type, count in ev_types.items():
        print(f" - {ev_type}: {count:,} ({count/len(df)*100:.1f}%)")
        
    print("\n=== CAFV ELIGIBILITY ===")
    cafv = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts()
    for cat, count in cafv.items():
        print(f" - {cat}: {count:,} ({count/len(df)*100:.1f}%)")
        
    print("\n=== TOP 10 MAKES ===")
    print(df['Make'].value_counts().head(10))
    
    print("\n=== TOP 10 MODELS ===")
    print(df.groupby(['Make', 'Model']).size().sort_values(ascending=False).head(10))
    
    print("\n=== TOP 10 COUNTIES ===")
    print(df['County'].value_counts().head(10))
    
    print("\n=== ELECTRIC RANGE DISTRIBUTION ===")
    print(df['Electric Range'].describe())
    zero_range = (df['Electric Range'] == 0).sum()
    print(f"Zero Range Count: {zero_range:,} ({zero_range/len(df)*100:.1f}%)")
    nonzero_range = df[df['Electric Range'] > 0]['Electric Range']
    print(f"Non-zero Range Median: {nonzero_range.median():.1f}, Mean: {nonzero_range.mean():.1f}, Max: {nonzero_range.max():.1f}")
    
    print("\n=== BASE MSRP SUMMARY ===")
    print(df['Base MSRP'].describe())
    zero_msrp = (df['Base MSRP'] == 0).sum()
    print(f"Zero MSRP Count: {zero_msrp:,} ({zero_msrp/len(df)*100:.1f}%)")

if __name__ == '__main__':
    inspect()
