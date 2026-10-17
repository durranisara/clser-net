import yaml, torch, argparse, copy
from torch.utils.data import DataLoader
from models.clser_net import CLSERNet
from training.trainer import run_epoch
from training.transfer import (BLOCK_ORDER, L, set_trainable,
                               build_optimizer)
from data.datasets import (EmotionDataset, collate_pad, parse_emodb,
                           parse_urdu, FeatureExtractor)
from utils.seed import set_seed

def main(cfg_path, scenario_name):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg["seed"]); dev = cfg["device"]
    sc = next(s for s in cfg["scenarios"] if s["name"] == scenario_name)
    fx = FeatureExtractor(**{k: cfg["features"][k] for k in
                             ["sample_rate","n_mfcc","n_mels","win_ms","hop_ms"]})
    parser = {"emodb": parse_emodb, "urdu": parse_urdu}
    items = parser[sc["target"]](cfg["data"][f"{sc['target']}_root"])
    ds = EmotionDataset(items, fx, cfg["features"]["sample_rate"])
    n = len(ds); tr, va = int(0.7*n), int(0.15*n)
    g = torch.Generator().manual_seed(cfg["seed"])
    tr_ds, va_ds, _ = torch.utils.data.random_split(ds, [tr, va, n-tr-va], generator=g)
    tl = DataLoader(tr_ds, batch_size=cfg["training"]["batch_size"],
                    shuffle=True, collate_fn=collate_pad)
    vl = DataLoader(va_ds, batch_size=cfg["training"]["batch_size"],
                    shuffle=False, collate_fn=collate_pad)

    model = CLSERNet(d_h=cfg["model"]["d_h"],
                     num_emotions=cfg["model"]["num_emotions"],
                     dropout=cfg["model"]["dropout"],
                     temperature=cfg["model"]["temperature"]).to(dev)
    model.load_state_dict(torch.load(f"ckpt_pretrain_{sc['source']}.pt"))

    eta_b = cfg["training"]["eta_base"]; gam = cfg["training"]["gamma"]
    E = cfg["training"]["unfreeze_interval"]
    U0 = cfg["training"]["initial_trainable"]

    # Phase 2: freeze FE and CLATE, train EPRA and classifier
    trainable = set(BLOCK_ORDER[-U0:])
    set_trainable(model, trainable)
    opt = build_optimizer(model, eta_b, gam, trainable)

    best, best_state = -1.0, None
    U = U0
    for ep in range(1, cfg["training"]["adapt_epochs"] + 1):
        # Phase 3: progressive unfreezing every E epochs
        if ep > 1 and (ep - 1) % E == 0 and U < L:
            U += 1
            trainable = set(BLOCK_ORDER[-U:])
            set_trainable(model, trainable)
            opt = build_optimizer(model, eta_b, gam, trainable)
            print(f"[ep {ep}] unfroze -> {sorted(trainable)}")

        tr = run_epoch(model, tl, dev, opt,
                       lambda_attr=cfg["loss"]["lambda_attr"],
                       lambda_sep=cfg["loss"]["lambda_sep"],
                       margin=cfg["loss"]["margin"])
        va = run_epoch(model, vl, dev, None,
                       lambda_attr=cfg["loss"]["lambda_attr"],
                       lambda_sep=cfg["loss"]["lambda_sep"],
                       margin=cfg["loss"]["margin"])
        print(f"ep {ep:03d} | train {tr['accuracy']:.4f} | val {va['accuracy']:.4f}")
        if va["accuracy"] > best:
            best, best_state = va["accuracy"], copy.deepcopy(model.state_dict())
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"ckpt_adapted_{sc['name']}.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scenario", required=True, choices=["EN_DE","EN_UR","RAV_DE","RAV_UR"])
    a = ap.parse_args()
    main(a.config, a.scenario)
