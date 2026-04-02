import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import random

# =========================
# Utils
# =========================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def to_blocks(x, grid_size):
    return x.view(grid_size, grid_size)

def overlap_score(pred_set, gt_set):
    return len(pred_set & gt_set) / len(gt_set)

# =========================
# Model
# =========================
class SimpleMLP(torch.nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_hidden=False):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        if return_hidden:
            return out, h
        return out

# =========================
# Perturbation
# =========================
def inject_perturbation(h, grid_size, magnitude=1.0, k=4):
    h = h.clone()
    indices = torch.randperm(h.numel())[:k]
    gt_set = set()

    for idx in indices:
        h[idx] += magnitude
        i = idx // grid_size
        j = idx % grid_size
        gt_set.add((i.item(), j.item()))

    return h, gt_set

# =========================
# A0: Full search
# =========================
def search_A0(h_clean, h_perturbed, grid_size, k=4):
    cost = 0
    trace = []

    diffs = []
    for i in range(grid_size):
        for j in range(grid_size):
            cost += 1
            trace.append(("scan", (i, j)))

            idx = i * grid_size + j
            diff = torch.abs(h_clean[idx] - h_perturbed[idx]).item()
            diffs.append(((i, j), diff))

    diffs.sort(key=lambda x: -x[1])
    pred_set = set(coord for coord, _ in diffs[:k])

    return pred_set, cost, grid_size * grid_size, trace

# =========================
# A1: Guided search
# =========================
def search_A1(h_clean, h_perturbed, grid_size, k=4):
    cost = 0
    trace = []

    row_scores = []
    for i in range(grid_size):
        cost += 1
        trace.append(("anchor_row", i))

        row_diff = 0
        for j in range(grid_size):
            idx = i * grid_size + j
            row_diff += torch.abs(h_clean[idx] - h_perturbed[idx]).item()

        row_scores.append((i, row_diff))

    row_scores.sort(key=lambda x: -x[1])

    checked = []
    expanded_rows = 0

    for row_idx, _ in row_scores:
        expanded_rows += 1

        for j in range(grid_size):
            cost += 1
            trace.append(("expand", (row_idx, j)))

            idx = row_idx * grid_size + j
            diff = torch.abs(h_clean[idx] - h_perturbed[idx]).item()
            checked.append(((row_idx, j), diff))

        if len(checked) >= k and expanded_rows >= 2:
            break

    checked.sort(key=lambda x: -x[1])
    pred_set = set(coord for coord, _ in checked[:k])

    return pred_set, cost, expanded_rows * grid_size, trace

# =========================
# A2: Structured computation
# =========================
def search_A2(h_clean, h_perturbed, grid_size, k=4):
    cost = 0
    trace = []

    cost += 1
    trace.append(("anchor", None))

    diff = torch.abs(h_clean - h_perturbed)

    cost += 1
    topk = torch.topk(diff, k=k).indices.tolist()

    pred_set = set()
    for idx in topk:
        i = idx // grid_size
        j = idx % grid_size
        pred_set.add((i, j))
        trace.append(("decode", (i, j)))

    return pred_set, cost, k, trace

# =========================
# Experiment
# =========================
def run_experiment():
    class Cfg:
        seed = 0
        hidden_dim = 36
        num_classes = 10
        lr = 1e-3
        epochs = 1
        device = "cpu"
        grid_size = 6
        target_k = 4
        samples = 10
        perturb_magnitude = 1.0

    cfg = Cfg()
    set_seed(cfg.seed)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    model = SimpleMLP(cfg.hidden_dim, cfg.num_classes)
    model.eval()

    collected = 0

    for x, y in dataset:
        if collected >= cfg.samples:
            break

        x = x.unsqueeze(0)
        out, h = model(x, return_hidden=True)

        h_pert, gt_set = inject_perturbation(h[0], cfg.grid_size, k=cfg.target_k)

        pred_A0, cost_A0, _, trace_A0 = search_A0(h[0], h_pert, cfg.grid_size, k=cfg.target_k)
        pred_A1, cost_A1, _, trace_A1 = search_A1(h[0], h_pert, cfg.grid_size, k=cfg.target_k)
        pred_A2, cost_A2, _, trace_A2 = search_A2(h[0], h_pert, cfg.grid_size, k=cfg.target_k)

        print("\n=== SAMPLE ===")
        print("A0 trace len:", len(trace_A0))
        print("A1 trace len:", len(trace_A1))
        print("A2 trace:", trace_A2)

        collected += 1

# =========================
# Main
# =========================
if __name__ == "__main__":
    run_experiment()