#!/usr/bin/env python3
"""
Quick test to verify all professional visualizations work correctly
"""

import os
import subprocess

def main():
    print("🧪 Testing Professional Visualization System")
    print("=" * 50)
    
    # Check if training data exists
    if os.path.exists('training_outputs/training_progress.json'):
        print("✅ Training data found")
        
        # Test professional visualization
        print("\n🎨 Testing professional visualization...")
        try:
            result = subprocess.run(['python', 'professional_visualization.py'], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ Professional visualization test PASSED")
                
                # Check output files
                expected_files = [
                    'analysis_results/professional_learning_curves.png',
                    'analysis_results/professional_performance_metrics.png',
                    'analysis_results/professional_3d_trajectory.png',
                    'analysis_results/professional_dashboard.png'
                ]
                
                print("\n📊 Checking output files:")
                all_files_exist = True
                for file_path in expected_files:
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        print(f"   ✅ {os.path.basename(file_path)} ({file_size:,} bytes)")
                    else:
                        print(f"   ❌ {os.path.basename(file_path)} - MISSING")
                        all_files_exist = False
                
                if all_files_exist:
                    print("\n🎉 ALL TESTS PASSED!")
                    print("📁 Professional visualizations are ready in analysis_results/")
                else:
                    print("\n⚠️  Some output files are missing")
                    
            else:
                print("❌ Professional visualization test FAILED")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Visualization test TIMED OUT (>60s)")
        except Exception as e:
            print(f"❌ Visualization test ERROR: {e}")
    
    else:
        print("❌ Training data not found")
        print("   Run training first: python examples/simple_train.py")

if __name__ == "__main__":
    main()