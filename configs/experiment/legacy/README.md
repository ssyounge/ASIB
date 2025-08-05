# Legacy Experiment Configs

이 디렉토리에는 초기 1회성 실험을 위한 config 파일들이 보관되어 있습니다.

## 파일 설명

### `res152_convnext_effi.yaml`
- **용도**: ResNet152 + ConvNeXt-L + EfficientNet-L2 조합 실험
- **학생 모델**: ResNet152
- **교사 모델**: ConvNeXt-L, EfficientNet-L2
- **상태**: 1회성 실험 완료

### `res152_effi_l2.yaml`
- **용도**: ResNet152 + EfficientNet-L2 조합 실험
- **학생 모델**: ResNet152
- **교사 모델**: ResNet152, EfficientNet-L2
- **상태**: 1회성 실험 완료

### `_template.yaml`
- **용도**: 실험 config 템플릿
- **상태**: 참고용으로 보관

## 새로운 실험 계획

이제 체계적인 3단계 실험 계획에 따라 새로운 config 파일들이 사용됩니다:

- **Phase 1**: Ablation Study (`ablation_*.yaml`)
- **Phase 2**: SOTA Comparison (`sota_*.yaml`)
- **Phase 3**: Overlap Analysis (`overlap_*.yaml`)

자세한 내용은 `EXPERIMENT_PLAN.md`를 참조하세요. 