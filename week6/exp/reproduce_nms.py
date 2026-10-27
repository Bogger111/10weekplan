import torch

conf = torch.tensor([
    0.80, 0.95, 0.70, 0.60, 0.30, 0.20
])

pelvis = torch.tensor([
    [0.00, 0.0, 2.0],
    [0.10, 0.0, 2.0],
    [1.00, 0.0, 2.0],
    [1.10, 0.0, 2.0],
    [2.00, 0.0, 2.0],
    [3.00, 0.0, 2.0]
])

CONF_THRESH = 0.5
NMS_THRESH = 0.25

def filter_queries(conf, threshold):
    mask = conf > threshold
    indices = torch.where(mask)[0]

    return indices

def nms_pelvis(pelvis, conf, threshold):
    order = conf.argsort(descending=True)
    suppressed = torch.zeros(
        len(conf), dtype=torch.bool, device=conf.device
    )
    keep = []
    for i in order:
        i = i.item()
        if suppressed[i]:
            continue

        keep.append(i)

        dist = torch.norm(pelvis - pelvis[i], dim=1)

        suppressed |= (dist < threshold)

        suppressed[i] = True

    return torch.tensor(
        keep, dtype=torch.long, device=conf.device
    )



import torch


def run_test(name, pelvis, conf, threshold, expected):
    try:
        result = nms_pelvis(pelvis, conf, threshold)

        # 转换为 Python list，便于比较
        result = torch.as_tensor(result).tolist()

        assert result == expected, (
            f"Expected {expected}, got {result}"
        )

        print(f"[PASS] {name}")

    except Exception as e:
        print(f"[FAIL] {name}: {e}")


# Test 1: 正常 NMS
# Q0/Q1 接近，Q2/Q3 接近
pelvis = torch.tensor([
    [0.0, 0.0, 2.0],
    [0.1, 0.0, 2.0],
    [1.0, 0.0, 2.0],
    [1.1, 0.0, 2.0],
])

conf = torch.tensor([0.80, 0.95, 0.70, 0.60])

run_test(
    "Normal NMS",
    pelvis, conf, 0.25,
    expected=[1, 2]
)


# Test 2: 只有一个候选
run_test(
    "Single Person",
    torch.tensor([[0.0, 0.0, 2.0]]),
    torch.tensor([0.9]),
    0.25,
    expected=[0]
)


# Test 3: 空输入
run_test(
    "Empty Input",
    torch.empty((0, 3)),
    torch.empty(0),
    0.25,
    expected=[]
)


# Test 4: 所有候选距离很远
run_test(
    "No Suppression",
    torch.tensor([
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [2.0, 0.0, 2.0],
    ]),
    torch.tensor([0.9, 0.8, 0.7]),
    0.25,
    expected=[0, 1, 2]
)


# Test 5: 距离刚好等于阈值
# 源码使用 < 而不是 <=
run_test(
    "Threshold Boundary",
    torch.tensor([
        [0.0, 0.0, 2.0],
        [0.25, 0.0, 2.0],
    ]),
    torch.tensor([0.9, 0.8]),
    0.25,
    expected=[0, 1]
)


# Test 6: 链式距离反例
# A-B = 0.1
# B-C = 0.1
# A-C = 0.2
# threshold = 0.15
run_test(
    "Greedy Chain",
    torch.tensor([
        [0.0, 0.0, 2.0],
        [0.1, 0.0, 2.0],
        [0.2, 0.0, 2.0],
    ]),
    torch.tensor([0.95, 0.80, 0.70]),
    0.15,
    expected=[0, 2]
)

conf = torch.tensor([
    0.80, 0.95, 0.70, 0.60, 0.30, 0.20
])

pelvis = torch.tensor([
    [0.00, 0.0, 2.0],
    [0.10, 0.0, 2.0],
    [1.00, 0.0, 2.0],
    [1.10, 0.0, 2.0],
    [2.00, 0.0, 2.0],
    [3.00, 0.0, 2.0]
])



conf = torch.tensor([
    0.80, 0.95, 0.70, 0.60, 0.30, 0.20
])

pelvis = torch.tensor([
    [0.00, 0.0, 2.0],
    [0.10, 0.0, 2.0],
    [1.00, 0.0, 2.0],
    [1.10, 0.0, 2.0],
    [2.00, 0.0, 2.0],
    [3.00, 0.0, 2.0]
])

indices = filter_queries(conf, 0.5)

filtered_conf = conf[indices]
filtered_pelvis = pelvis[indices]

keep_local = nms_pelvis(
    filtered_pelvis,
    filtered_conf,
    0.25
)

keep_original = indices[keep_local]

print("After confidence:", indices.tolist())
print("After NMS (local):", keep_local.tolist())
print("Final query indices:", keep_original.tolist())