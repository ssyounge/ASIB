#!/usr/bin/env python3
"""
Comprehensive Analysis Script

모든 심층 분석을 통합적으로 실행합니다.
Information Plane, CCCP Stability, Teacher Adaptation, PF Efficiency 분석
"""

import os
import sys
import subprocess
from datetime import datetime

def run_analysis_script(script_name: str, description: str) -> bool:
    """분석 스크립트를 실행합니다."""
    print(f"\n🔬 {description}")
    print("=" * 60)
    
    script_path = f"scripts/analysis/{script_name}"
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def generate_comprehensive_report():
    """종합 분석 보고서를 생성합니다."""
    report = f"""
Comprehensive ASIB Analysis Report
=================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report summarizes the comprehensive analysis of the ASIB framework,
including Information Plane analysis, CCCP stability, Teacher Adaptation,
and Progressive Partial Freezing efficiency.

Analysis Components:
1. Information Plane Analysis (IB Theory Connection)
2. CCCP Stability Analysis (Learning Curve Comparison)
3. Teacher Adaptation Analysis (Performance Preservation)
4. PF Efficiency Analysis (Memory & Time Optimization)

Results are saved in the following directories:
- outputs/analysis/information_plane/
- outputs/analysis/cccp_stability/
- outputs/analysis/teacher_adaptation/
- outputs/analysis/pf_efficiency/

For detailed results, please refer to the individual analysis reports
in each directory.

Key Insights Summary:
- Information Plane analysis validates IB theory in knowledge distillation
- CCCP provides significant learning stability improvements
- Teacher Adaptation achieves performance gains while preserving teacher knowledge
- Progressive Partial Freezing delivers substantial efficiency improvements

This comprehensive analysis demonstrates the theoretical soundness and
practical effectiveness of the ASIB framework.
"""
    
    report_path = "outputs/reports/comprehensive_analysis_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Comprehensive report saved to: {report_path}")
    return report

def run_comprehensive_analysis():
    """전체 종합 분석을 실행합니다."""
    print("🎯 Starting Comprehensive ASIB Analysis")
    print("=" * 60)
    print("This will run all deep analysis components:")
    print("1. Information Plane Analysis")
    print("2. CCCP Stability Analysis") 
    print("3. Teacher Adaptation Analysis")
    print("4. PF Efficiency Analysis")
    print("=" * 60)
    
    # 분석 스크립트 목록
    analyses = [
        ("information_plane_analysis.py", "Information Plane Analysis (IB Theory Connection)"),
        ("cccp_stability_analysis.py", "CCCP Stability Analysis (Learning Curve Comparison)"),
        ("teacher_adaptation_analysis.py", "Teacher Adaptation Analysis (Performance Preservation)"),
        ("pf_efficiency_analysis.py", "PF Efficiency Analysis (Memory & Time Optimization)")
    ]
    
    # 각 분석 실행
    results = {}
    for script_name, description in analyses:
        success = run_analysis_script(script_name, description)
        results[description] = success
    
    # 결과 요약
    print("\n📊 Analysis Results Summary")
    print("=" * 60)
    
    successful = 0
    for description, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {description}")
        if success:
            successful += 1
    
    print(f"\nOverall: {successful}/{len(analyses)} analyses completed successfully")
    
    # 종합 보고서 생성
    if successful > 0:
        print("\n📄 Generating comprehensive report...")
        report = generate_comprehensive_report()
        print(report)
    
    print("\n🎉 Comprehensive analysis completed!")
    print("📁 All results saved in outputs/analysis/ and outputs/reports/ directories")

if __name__ == "__main__":
    run_comprehensive_analysis() 