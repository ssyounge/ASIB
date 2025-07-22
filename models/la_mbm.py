"""
LightweightAttnMBM  (DEPRECATED)
================================
2025-07-22 기준 연구 라인에서 더 이상 사용하지 않습니다.
  • 새 실험은 모두 Information-Bottleneck MBM(ib_mbm) 으로 대체
  • 코드를 참조하려고 import 하면 즉시 RuntimeError 를 발생시켜
    잘못된 설정을 초기에 알아차릴 수 있도록 했습니다.
"""

class LightweightAttnMBM:
    """Deprecated placeholder for backward compatibility."""
    def __init__(self, *_, **__):
        raise RuntimeError(
            "LightweightAttnMBM is deprecated. "
            "config 파일에서 `mbm_type: ib_mbm` 으로 변경하세요."
        )

    def forward(self, *_):
        # 호출이 일어날 가능성은 거의 없지만, 혹시 모를 실행 도중
        # 에러 메시지가 더 명확하게 보이도록 한 번 더 막아 둡니다.
        raise RuntimeError("LA-MBM forward called - this module is deprecated.")
