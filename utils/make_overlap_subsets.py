# utils/make_overlap_subsets.py

import random, json
def split_classes(n_cls=100, seed=42):
    random.seed(seed)
    cls = list(range(n_cls)); random.shuffle(cls)
    return cls

def make_pairs(overlap_pct: int, n_cls=100):
    m = int(n_cls * overlap_pct / 100)     # 겹치는 클래스 수
    base = split_classes(n_cls)
    common = base[:m]                      # 공유 부분
    rest   = base[m:]
    half   = (n_cls - m) // 2
    t1 = sorted(common + rest[:half])
    t2 = sorted(common + rest[half:half*2])
    return {"overlap": overlap_pct, "T1": t1, "T2": t2}

# 예시 출력
for p in [0,10,20,30]:
    print(make_pairs(p))
