#!/usr/bin/env python3
"""
Collect and aggregate ablation results from steering coefficient and layer experiments.
Computes average scores across all eval sets and generates summary statistics.
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path


def parse_filename(filename):
    """
    Parse the filename to extract trait, layer, and coefficient.
    Expected format: {trait}_layer{layer}_coef{coef}.csv
    """
    pattern = r"(.+)_layer(\d+)_coef([\d.]+)\.csv"
    match = re.match(pattern, filename)
    if match:
        trait = match.group(1)
        layer = int(match.group(2))
        coef = float(match.group(3))
        return trait, layer, coef
    return None, None, None

def parse_filename_baseline(filename):
    """
    Parse the filename for baseline
    Expected format: ${trait_target}.csv"
    """
    pattern = r"(.+)\.csv"
    match = re.match(pattern, filename)
    if match:
        trait_target = match.group(1)
        return None, trait_target
    return None, None

def load_results(results_dir, ablation_type=None):
    """
    Load all result CSV files from the results directory.
    Returns a list of dictionaries with parsed metadata and scores.
    """
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return results
    
    csv_files = list(results_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {results_dir}")
    
    for csv_file in csv_files:
        if ablation_type == "question_type":
            trait, trait_target = parse_filename_baseline(csv_file.name)
        else:
            trait, coef = parse_filename_baseline(csv_file.name)
        
        print('trait parsed:', trait)
        
        
        try:
            df = pd.read_csv(csv_file)
            
            # Get the trait score column (should match the trait name)
            
            # Added to handle question_type ablation
            # The median is added to consider the skewness of the distribution
            if ablation_type == "question_type":
                if trait_target in df.columns:
                    trait_scores = df[trait_target].dropna()
                    trait_mean = trait_scores.mean() if len(trait_scores) > 0 else None
                    trait_std = trait_scores.std() if len(trait_scores) > 0 else None
                    trait_median = trait_scores.median() if len(trait_scores) > 0 else None

            else:
                if trait in df.columns:
                    trait_scores = df[trait].dropna()
                    trait_mean = trait_scores.mean() if len(trait_scores) > 0 else None
                    trait_std = trait_scores.std() if len(trait_scores) > 0 else None
                    trait_median = trait_scores.median() if len(trait_scores) > 0 else None
                else:
                    # Try to find a matching column
                    trait_col = None
                    for col in df.columns:
                        if trait.replace("_", "") in col.replace("_", "").lower():
                            trait_col = col
                            break
                    
                    if trait_col:
                        trait_scores = df[trait_col].dropna()
                        trait_mean = trait_scores.mean() if len(trait_scores) > 0 else None
                        trait_std = trait_scores.std() if len(trait_scores) > 0 else None
                        trait_median = trait_scores.median() if len(trait_scores) > 0 else None
                    else:
                        print(f"  Warning: Could not find trait column for {trait} in {csv_file.name}")
                        trait_mean = None
                        trait_std = None
                        trait_median = None

            # Get coherence score if available
            if "coherence" in df.columns:
                coherence_scores = df["coherence"].dropna()
                coherence_mean = coherence_scores.mean() if len(coherence_scores) > 0 else None
                coherence_std = coherence_scores.std() if len(coherence_scores) > 0 else None
                coherence_median = coherence_scores.median() if len(coherence_scores) > 0 else None
            else:
                coherence_mean = None
                coherence_std = None
                coherence_median = None

            if ablation_type == "question_type":
                results.append({
                    "trait_source": trait,
                    "trait_target": trait_target,
                    "trait_score_mean": trait_mean,
                    "trait_score_std": trait_std,
                    "trait_score_median": trait_median,
                    "coherence_mean": coherence_mean,
                    "coherence_std": coherence_std,
                    "coherence_median": coherence_median,
                    "n_samples": len(df),
                    "file": csv_file.name
                })
            else:
                results.append({
                    "trait": trait,
                    "layer": layer,
                    "coef": coef,
                    "trait_score_mean": trait_mean,
                    "trait_score_std": trait_std,
                    "trait_score_median": trait_median,
                    "coherence_mean": coherence_mean,
                    "coherence_std": coherence_std,
                    "coherence_median": coherence_median,
                    "n_samples": len(df),
                    "file": csv_file.name
                })
            
        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")
    
    return results


def compute_aggregated_scores(results_df):
    """
    Compute aggregated scores across all eval sets for each (layer, coef) combination.
    Returns a DataFrame with averaged scores.
    """
    # Group by layer and coefficient
    grouped = results_df.groupby(["layer", "coef"]).agg({
        "trait_score_mean": ["mean", "std", "count"],
        "coherence_mean": ["mean", "std"],
    }).reset_index()
    
    # Flatten column names
    grouped.columns = [
        "layer", "coef",
        "avg_trait_score", "trait_score_across_sets_std", "n_eval_sets",
        "avg_coherence", "coherence_across_sets_std"
    ]
    
    return grouped


def create_pivot_table(results_df, value_col="trait_score_mean"):
    """
    Create a pivot table for easy visualization of results.
    Rows: layers, Columns: coefficients, Values: averaged scores across traits
    """
    # First average across traits for each (layer, coef)
    avg_df = results_df.groupby(["layer", "coef"])[value_col].mean().reset_index()
    
    # Create pivot table
    pivot = avg_df.pivot(index="layer", columns="coef", values=value_col)
    
    return pivot


def main():
    parser = argparse.ArgumentParser(
        description="Collect and aggregate ablation results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing result CSV files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for aggregated results (default: {results_dir}/ablation_summary.csv)"
    )
    parser.add_argument(
        "--print_summary",
        action="store_true",
        default=True,
        help="Print summary statistics to console"
    )
    
    args = parser.parse_args()
    
    # Set default output file
    if args.output_file is None:
        args.output_file = os.path.join(args.results_dir, "ablation_summary.csv")
    
    #read last dir of results_dir
    ablation_type = os.path.basename(os.path.normpath(args.results_dir))
    ablation_type = 'question_type'
    
    print("=" * 70)
    print("Collecting Ablation Results")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print()
    
    if ablation_type == "question_type":
        print("Detected Question Type Ablation Results")
        results = load_results(args.results_dir, ablation_type)
    else:
        results = load_results(args.results_dir)
    
    if not results:
        print("No results found. Exiting.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    print()
    print("=" * 70)
    print("Per-Eval-Set Results")
    print("=" * 70)
    print()
    print(results_df.to_string(index=False))
    
    
    if ablation_type != "question_type":
        # Compute aggregated scores
        print()
        print("=" * 70)
        print("Aggregated Results (Averaged Across Eval Sets)")
        print("=" * 70)
        print()
        
        aggregated_df = compute_aggregated_scores(results_df)
        print(aggregated_df.to_string(index=False))
        
        # Create pivot table for visualization
        print()
        print("=" * 70)
        print("Pivot Table: Average Trait Score (Rows=Layers, Cols=Coefficients)")
        print("=" * 70)
        print()
        
        pivot = create_pivot_table(results_df, "trait_score_mean")
        print(pivot.to_string())
        
        # Find best configuration
        print()
        print("=" * 70)
        print("Best Configurations")
        print("=" * 70)
        print()
        
        # Best for maximizing trait score (e.g., for steering towards evil)
        best_max_idx = aggregated_df["avg_trait_score"].idxmax()
        best_max = aggregated_df.loc[best_max_idx]
        print(f"Highest trait score: layer={int(best_max['layer'])}, coef={best_max['coef']:.1f}")
        print(f"  Avg trait score: {best_max['avg_trait_score']:.2f}")
        print(f"  Avg coherence: {best_max['avg_coherence']:.2f}")
        
        # Best for minimizing trait score (e.g., for steering away from evil)
        best_min_idx = aggregated_df["avg_trait_score"].idxmin()
        best_min = aggregated_df.loc[best_min_idx]
        print()
        print(f"Lowest trait score: layer={int(best_min['layer'])}, coef={best_min['coef']:.1f}")
        print(f"  Avg trait score: {best_min['avg_trait_score']:.2f}")
        print(f"  Avg coherence: {best_min['avg_coherence']:.2f}")
    
    # Save results
    print()
    print("=" * 70)
    print("Saving Results")
    print("=" * 70)
    print()
    
    # Save detailed results
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.mkdir(os.path.dirname(args.output_file))
    results_df.to_csv(args.output_file, index=False)
    print(f"Detailed results saved to: {args.output_file}")
    
    if ablation_type != "question_type":
        # Save aggregated results
        agg_output = args.output_file.replace(".csv", "_aggregated.csv")
        aggregated_df.to_csv(agg_output, index=False)
        print(f"Aggregated results saved to: {agg_output}")

        # Save pivot table
        pivot_output = args.output_file.replace(".csv", "_pivot.csv")
        pivot.to_csv(pivot_output)
        print(f"Pivot table saved to: {pivot_output}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()

