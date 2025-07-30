# models/common/registry.py

"""Minimal registry – **자동 스캔은 사용하지 않습니다**.
   - ``@register("key")`` 로만 수동 등록
   - 또는 ``configs/registry_key.yaml`` 에 명시된 모듈만 import 합니다.
"""

from pathlib import Path
from importlib import import_module
import re
import yaml

MODEL_REGISTRY: dict[str, type] = {}


def register(key: str):
    """Decorator for manual registration."""

    def _wrap(cls):
        if key in MODEL_REGISTRY:
            # 동일 클래스면 조용히 패스, 아니면 에러
            if MODEL_REGISTRY[key] is cls:
                return cls
            raise KeyError(f"[registry] duplicate key: {key}")
        MODEL_REGISTRY[key] = cls
        return cls

    return _wrap


# ---------------------------------------------------------
# 내부 유틸: 서브모듈 import + BaseKDModel 자동 등록
#   ‐ 직접 호출 금지, 아래 ensure_scanned() 에서만 사용
# ---------------------------------------------------------
def _import_submodules(pkg_root: str):
    pkg = import_module(pkg_root)
    root = Path(pkg.__file__).parent
    for p in root.glob("**/*.py"):
        if p.name.startswith("_"):
            continue
        rel = p.with_suffix("").relative_to(root)
        mod = f"{pkg_root}.{rel.as_posix().replace('/', '.')}"
        import_module(mod)


# ---------------------------------------------------------
# ❶  *자동* alias 생성 **전면 제거**
#     – BaseKDModel 서브클래스도 “데코레이터로 등록한 키”만 유지
# ---------------------------------------------------------
def _auto_register():
    """No-op: alias 자동 생성 로직을 완전히 비활성화."""
    return


# ---------------------------------------------------------
# ❷  registry_key.yaml  (허용 키 필터)
# ---------------------------------------------------------
import yaml, pkg_resources, json
_CFG_PATH = pkg_resources.resource_filename(__name__, "../../configs/registry_key.yaml")
if Path(_CFG_PATH).is_file():
    with open(_CFG_PATH, "r") as f:
        _key_cfg = yaml.safe_load(f)
    _ALLOW_KEYS = set(_key_cfg.get("student_keys", []) + _key_cfg.get("teacher_keys", []))
else:
    _ALLOW_KEYS = set()

# ---------------------------------------------------------------------------
# 호출 여부 플래그와 헬퍼
_SCANNED = False


def ensure_scanned(*, slim: bool = False):
    """첫 호출 시 students / teachers 서브모듈 import → 자동 등록"""
    global _SCANNED
    if _SCANNED:
        return

    # ------------------------------------------------------------------
    #  A) 1차:  registry_key.yaml 의 key → 대응 모듈만 import
    # ------------------------------------------------------------------
    def _deduce_module(key: str) -> list[str]:
        """
        key → import 대상 후보 목록

        - <base>_pretrain_student  →  models.students.<base>_student
        - <base>_scratch_student   →            "
        - <base>_student           →  그대로
        - <base>_teacher           →  그대로
        - 접미사 없으면 두 군데 모두 시도
        """
        if key.endswith("_teacher"):
            return [f"models.teachers.{key}"]

        if key.endswith("_student"):
            # pretrain/scratch → student 파일
            base = key.rsplit("_", 2)[0]   # 'resnet152'
            return [
                f"models.students.{base}_student",   # 표준 파일
                f"models.students.{key}",            # 혹시 동일 파일명도 있을 때
            ]

        # 접미사 없는 경우
        return [
            f"models.students.{key}_student",
            f"models.teachers.{key}_teacher",
        ]

    import logging

    for k in _ALLOW_KEYS:
        ok = False
        for m in _deduce_module(k):
            try:
                import_module(m)
                ok = True          # 한 번 성공하면 더 시도 X
                break
            except ModuleNotFoundError:
                continue
        if not ok:                 # 후보 전부 실패했을 때만 경고
            logging.warning("[registry] import failed for key '%s'", k)

    _auto_register()        # ← 지금은 빈 함수 (alias X)

    # ------------------------------------------------------------------
    #  B) 2차: allow-list 필터  (scan 후에도 key 누락시 제거)
    # ------------------------------------------------------------------
    if _ALLOW_KEYS:
        # 1-A) 먼저 허용-키에 없는 기존 엔트리는 제거
        for k in list(MODEL_REGISTRY):
            if k not in _ALLOW_KEYS:
                MODEL_REGISTRY.pop(k)

        # 1-B) 허용-키인데 아직 없는 경우 ↔ 자동 alias 보강
        def _try_alias(key: str, *candidates: str):
            for c in candidates:
                if c in MODEL_REGISTRY:
                    MODEL_REGISTRY[key] = MODEL_REGISTRY[c]
                    break

        for k in _ALLOW_KEYS:
            if k in MODEL_REGISTRY:
                continue

            # ── Student 키 패턴:  base_(pretrain|scratch)
            m = re.match(r"(.+?)_(pretrain|scratch)$", k)
            if m:
                base = m.group(1)
                _try_alias(
                    k,
                    f"{base}_student",
                    base,
                    f"{base}student",
                )
                continue

            # ── Teacher 키:  base
            base = k
            _try_alias(
                k,
                f"{base}_teacher",
                base,
                f"{base}teacher",
            )

    _SCANNED = True

