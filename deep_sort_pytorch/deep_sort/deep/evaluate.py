import torch

features = torch.load("features.pth")
qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]

scores = qf.mm(gf.t())
res = scores.topk(5, dim=1)[1][:, 0]
top1correct = gl[res].eq(ql).sum().item()

print(f"Acc top1:{top1correct / ql.size(0):.3f}")
