#!/usr/bin/env python3
"""
미리 Teacher 출력(logits·z)을 저장해 두면 이후 KD 단계에서
teacher forward 시간이 ≒ 0 이 됩니다.
"""
import torch, tqdm, argparse, os, yaml
from data.cifar100 import get_cifar100_loaders
from utils.model_factory import create_teacher_by_name

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True, help='원본 YAML (teachers 정보 읽기)')
parser.add_argument('--out', required=True, help='output *.pt')
args = parser.parse_args()

cfg = yaml.safe_load(open(args.cfg))
device = 'cuda'

# ---------- teachers ----------
t1 = create_teacher_by_name('resnet152', 100, pretrained=False).to(device)
t1.load_state_dict(torch.load(cfg['teacher1_ckpt'], map_location='cpu'))
t2 = create_teacher_by_name('efficientnet_b2', 100, pretrained=False).to(device)
t2.load_state_dict(torch.load(cfg['teacher2_ckpt'], map_location='cpu'))
t1.eval(); t2.eval()

train_loader, _ = get_cifar100_loaders(
    root=cfg.get('dataset_root', './data'),
    batch_size=256,
    num_workers=4,
    randaug_N=0, randaug_M=0,
    persistent_train=False, persistent_test=False,
)

cache = {'logits': [], 'z': []}
with torch.no_grad():
    for x, _ in tqdm.tqdm(train_loader):
        x = x.to(device)
        z1, log1 = t1(x)['z'], t1(x)['logits']
        z2, log2 = t2(x)['z'], t2(x)['logits']
        cache['logits'].append((log1.cpu(), log2.cpu()))
        cache['z'].append((z1.cpu(), z2.cpu()))
for k in cache:
    cache[k] = torch.cat([torch.cat(t, dim=0) for t in cache[k]])
torch.save(cache, args.out)
print('✓ saved', args.out)
