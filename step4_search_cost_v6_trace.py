
# -*- coding: utf-8 -*-
Çdef run_experiment(cfg):
    set_seed(cfg.seed)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    model = SimpleMLP(cfg.hidden_dim, cfg.num_classes).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # quick training
    model.train()
    for epoch in range(cfg.epochs):
        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            out = model(x)
            loss = F.cross_entropy(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    results = {
        "A0": [],
        "A1": [],
        "A2": []
    }

    collected = 0

    for x, y in test_dataset:
        if collected >= cfg.samples:
            break

        x = x.unsqueeze(0).to(cfg.device)
        y = torch.tensor([y]).to(cfg.device)

        with torch.no_grad():
            out_clean, h_clean = model(x, return_hidden=True)
            pred_clean = out_clean.argmax(dim=1)

        h_pert, gt_set = inject_perturbation(
            h_clean[0],
            cfg.grid_size,
            magnitude=cfg.perturb_magnitude,
            k=cfg.target_k
        )

        with torch.no_grad():
            out_pert = model.fc2(torch.relu(h_pert.unsqueeze(0)))
            pred_pert = out_pert.argmax(dim=1)

        # only keep error samples
        if pred_clean.item() == pred_pert.item():
            continue

        collected += 1

        pred_A0, cost_A0, radius_A0 = search_A0(
            h_clean[0], h_pert, cfg.grid_size, k=cfg.target_k
        )
        pred_A1, cost_A1, radius_A1 = search_A1(
            h_clean[0], h_pert, cfg.grid_size, k=cfg.target_k
        )
        pred_A2, cost_A2, radius_A2, trace_A2 = search_A2(
            h_clean[0], h_pert, cfg.grid_size, k=cfg.target_k
        )

        print("A2 trace:", trace_A2)

        results["A0"].append({
            "exact": int(pred_A0 == gt_set),
            "overlap": overlap_score(pred_A0, gt_set),
            "cost": cost_A0,
            "radius": radius_A0
        })

        results["A1"].append({
            "exact": int(pred_A1 == gt_set),
            "overlap": overlap_score(pred_A1, gt_set),
            "cost": cost_A1,
            "radius": radius_A1
        })

        results["A2"].append({
            "exact": int(pred_A2 == gt_set),
            "overlap": overlap_score(pred_A2, gt_set),
            "cost": cost_A2,
            "radius": radius_A2,
            "trace": trace_A2
        })

        print(f"[{collected}/{cfg.samples}] collected")

    return results