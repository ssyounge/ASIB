# Reports Documentation

이 디렉토리에는 ASIB 프레임워크의 실험 계획과 분석 보고서들이 포함되어 있습니다.

## 📁 파일 목록

### **EXPERIMENT_PLAN.md**
- **용도**: 전체 실험 계획 문서
- **내용**: Phase 1, 2, 3 실험 계획과 실행 방법
- **상태**: 최종 버전

### **ABLATION_STUDY_REPORT.md**
- **용도**: 초기 Ablation Study 보고서
- **내용**: 실험 1, 2, 3의 하이퍼파라미터와 실험 방법
- **상태**: 초기 버전 (참고용)

### **IMPROVED_ABLATION_STUDY_REPORT.md**
- **용도**: 개선된 Ablation Study 보고서
- **내용**: 피드백 반영한 완전한 5단계 실험 계획
- **상태**: 최종 버전 (권장)

## 🔄 버전 관리

- **초기 버전**: `ABLATION_STUDY_REPORT.md` (기본 3단계 실험)
- **개선 버전**: `IMPROVED_ABLATION_STUDY_REPORT.md` (완전한 5단계 실험)
- **최종 계획**: `EXPERIMENT_PLAN.md` (전체 3단계 실험 계획)

## 📊 사용 권장사항

실험을 시작하기 전에 다음 순서로 문서를 참조하세요:

1. **EXPERIMENT_PLAN.md**: 전체 실험 계획 파악
2. **IMPROVED_ABLATION_STUDY_REPORT.md**: Phase 1 상세 계획
3. **ABLATION_STUDY_REPORT.md**: 초기 아이디어 참고

## 🎯 주요 개선사항

### **IMPROVED_ABLATION_STUDY_REPORT.md의 핵심 개선사항**

1. **완전한 Ablation Study**: 5단계 실험 (기존 3단계 → 5단계)
2. **β 민감도 분석**: Information Plane 분석 포함
3. **정량적 분석**: 통계적 유의성과 정량적 지표
4. **심층 분석 도구**: 4개 전용 분석 스크립트
5. **리뷰어 방어**: 최상위권 학회 요구사항 충족

이 문서들을 통해 체계적이고 완전한 실험을 수행할 수 있습니다. 