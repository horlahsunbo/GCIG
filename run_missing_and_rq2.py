#!/usr/bin/env python
"""
Run missing methods and RQ2 for all 7 methods.

Part 1: Run only the 3 missing methods for RQ1 (CV)
        - gcf_ig, hardt, agarwal
        - Uses same seed (42) to ensure identical fold splits

Part 2: Run RQ2 (Simplex Analysis) for ALL 7 methods
        - unconstrained, dir, hardt, agarwal, lagrangian, adversarial, gcf_ig
        - This will include the 4 methods from the original run + 3 new ones
"""

import sys
sys.path.insert(0, '/home/n88r684/gcf_ig')

from comprehensive_experiments import ComprehensiveExperiment

def main():
    print("="*80)
    print("RUNNING MISSING METHODS + RQ2 FOR ALL METHODS")
    print("="*80)
    
    # Create experiment with SAME settings as original
    experiment = ComprehensiveExperiment(
        output_dir='all_results/full_experiment',
        random_seed=42  # CRITICAL: Same seed as original!
    )
    
    # =======================================================================
    # PART 1: RQ1 CV - Missing Methods Only
    # =======================================================================
    print("\n" + "="*80)
    print("PART 1: RQ1 K-FOLD CV - MISSING METHODS ONLY")
    print("="*80)
    print("Methods: gcf_ig, hardt, agarwal")
    print("Datasets: german, compas, adult, bank")
    print("Random seed: 42 (same as original run)")
    print("="*80)
    
    # Mark as partial run to avoid overwriting existing files
    experiment.partial_run_suffix = '_MISSING'
    
    print("\nStarting K-Fold CV for missing methods...")
    
    # Run only the 3 missing methods
    df_folds, df_summary, df_ci = experiment.run_rq1_with_kfold_cv(
        datasets=['german', 'compas', 'adult', 'bank'],
        methods=[ 'hardt', 'agarwal'],  # ONLY missing methods
        n_folds=5
    )
    
    print("\n✅ Part 1 Complete: Missing methods CV done")
    print("\nGenerated files:")
    print("  - rq1_cv_metrics_by_fold_MISSING.csv")
    print("  - rq1_cv_summary_MISSING.csv")
    print("  - rq1_cv_confidence_intervals_MISSING.csv")
    
    # =======================================================================
    # PART 2: RQ2 Simplex Analysis - ALL 7 Methods
    # =======================================================================
    print("\n" + "="*80)
    print("PART 2: RQ2 SIMPLEX ANALYSIS - ALL 7 METHODS")
    print("="*80)
    print("Methods: ALL 7 (unconstrained, dir, hardt, agarwal, lagrangian, adversarial, gcf_ig)")
    print("Dataset: german (standard for RQ2)")
    print("="*80)
    
    # Remove suffix for RQ2 (use standard filenames)
    delattr(experiment, 'partial_run_suffix')
    
    print("\nRunning simplex analysis for all methods...")
    
    # Run RQ2 with ALL 7 methods
    experiment.run_rq2_simplex_analysis(
        dataset_name='german',
        # RQ2 will automatically train all 7 methods
    )
    
    print("\n✅ Part 2 Complete: Simplex analysis done for all 7 methods")
    print("\nGenerated files:")
    print("  - simplex/simplex_auc_statistical_parity_gcf_ig.png")
    print("  - simplex/simplex_auc_equalized_odds_gcf_ig.png")
    print("  - simplex/simplex_f1_statistical_parity_gcf_ig.png")
    print("  - simplex/coverage_summary.json")
    
    # =======================================================================
    # Summary
    # =======================================================================
    print("\n" + "="*80)
    print("ALL TASKS COMPLETE!")
    print("="*80)
    print("\n📊 What was generated:")
    print("\n1. RQ1 CV Results (3 missing methods):")
    print("   - rq1_cv_metrics_by_fold_MISSING.csv")
    print("   - rq1_cv_summary_MISSING.csv")
    print("   - rq1_cv_confidence_intervals_MISSING.csv")
    print("\n2. RQ2 Simplex Analysis (ALL 7 methods):")
    print("   - 3 simplex plots showing preference space coverage")
    print("   - Coverage summary with percentages")
    print("\n📋 Next steps:")
    print("1. Run merge_results.py to combine RQ1 results")
    print("2. Check simplex plots in simplex/ directory")
    print("3. Your RQ2 is now complete for all 7 methods!")
    print("="*80)

if __name__ == '__main__':
    main()
