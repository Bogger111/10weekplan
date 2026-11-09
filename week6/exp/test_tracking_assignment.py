import torch
from multihmr2.tracker import FeatPelvisTracker

tracker = FeatPelvisTracker()

# 模拟已有历史 Track：ID 5 和 ID 8
historical_ids = torch.tensor([5, 8])

# 当前三个检测与历史 Track 的相似度
sim = torch.tensor([
    [0.90, 0.88],
    [0.89, 0.10],
])

# 模拟已有 ID 0~8 均已分配过
tracker.next_human_id = 9

assigned_ids, is_new = tracker._compute_assignment(
    sim,
    historical_ids,
)

print("Assigned IDs:", assigned_ids.tolist())
print("New identities:", is_new.tolist())
print("Next available ID:", tracker.next_human_id)