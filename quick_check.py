import torch, importlib
from models.common.base_wrapper import MODEL_REGISTRY

x = torch.rand(2, 3, 32, 32)
for name, cls in MODEL_REGISTRY.items():
    try:
        m = cls(pretrained=False, num_classes=10, small_input=True, cfg={})
        m.eval()
        m(x)
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
