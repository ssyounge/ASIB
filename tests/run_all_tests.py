#!/usr/bin/env python3
"""Run all tests in the tests directory"""

import sys
import subprocess
import time
from pathlib import Path

def run_test_file(test_file):
    """Run a single test file"""
    print(f"\n🧪 Running {test_file}...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, test_file
        ], 
        capture_output=True, 
        text=True, 
        cwd=Path(__file__).parent.parent,
        env={"PYTHONPATH": str(Path(__file__).parent.parent)}
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_file} - PASSED ({duration:.2f}s)")
            if result.stdout.strip():
                print(result.stdout)
            return True
        else:
            print(f"❌ {test_file} - FAILED ({duration:.2f}s)")
            if result.stdout.strip():
                print("STDOUT:")
                print(result.stdout)
            if result.stderr.strip():
                print("STDERR:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ {test_file} - ERROR: {e}")
        return False

def main():
    """Run all test files"""
    tests_dir = Path(__file__).parent
    test_files = sorted(tests_dir.glob("test_*.py"))
    
    print("🚀 Running All Tests")
    print("=" * 60)
    print(f"Found {len(test_files)} test files")
    
    results = []
    total_start = time.time()
    
    for test_file in test_files:
        success = run_test_file(test_file)
        results.append((test_file.name, success))
    
    total_end = time.time()
    total_duration = total_end - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    print(f"⏱️  Total time: {total_duration:.2f}s")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 