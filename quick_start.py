"""
Quick Start Script for Netflix ML Pipeline
Runs initial data quality check and cleaning
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.quality_check import DataQualityChecker
from src.data.cleaning import DataCleaner


def main():
    print("=" * 70)
    print(" " * 15 + "NETFLIX ML PIPELINE - QUICK START")
    print("=" * 70)
    print()
    
    # Step 1: Data Quality Assessment
    print("STEP 1: Running Data Quality Assessment...")
    print("-" * 70)
    
    try:
        checker = DataQualityChecker(
            data_path="data/raw/netflix1.csv",
            output_dir="reports/data_quality"
        )
        report = checker.run_full_assessment()
        checker.save_report()
        print("\n✓ Data quality assessment completed successfully!")
    except Exception as e:
        print(f"\n✗ Error in quality assessment: {e}")
        return
    
    print("\n" + "=" * 70)
    
    # Step 2: Data Cleaning
    print("\nSTEP 2: Running Data Cleaning Pipeline...")
    print("-" * 70)
    
    try:
        cleaner = DataCleaner(
            input_path="data/raw/netflix1.csv",
            output_dir="data/processed"
        )
        cleaned_df = cleaner.run_full_cleaning()
        cleaner.save_cleaned_data()
        print("\n✓ Data cleaning completed successfully!")
    except Exception as e:
        print(f"\n✗ Error in data cleaning: {e}")
        return
    
    print("\n" + "=" * 70)
    print(" " * 20 + "QUICK START COMPLETE!")
    print("=" * 70)
    print()
    print("📊 Next Steps:")
    print("  1. Review quality report: reports/data_quality/")
    print("  2. Check cleaned data: data/processed/netflix_cleaned.csv")
    print("  3. Review changelog: logs/transformation_changelog.json")
    print("  4. Open Jupyter notebook: notebooks/01_data_quality.ipynb")
    print()
    print("📖 For full implementation guide, see: IMPLEMENTATION_GUIDE.md")
    print()


if __name__ == "__main__":
    main()
