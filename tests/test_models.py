import numpy as np
import torch
from torchdiffeq import odeint

from cnode.data.dataset import LongitudinalDataset, Normalizer, collate, load_subjects
from cnode.models import MODEL_NAMES, build_model
from cnode.models.losses import metadata_weighted_infonce
from cnode.train import run_cv


def _batch(synthetic_dir, n=6):
    subjects, feats, vols = load_subjects(synthetic_dir)
    subjects = Normalizer().fit(subjects).transform(subjects)
    ds = LongitudinalDataset(subjects[:n])
    return collate([ds[i] for i in range(n)]), len(feats), len(vols) + 2


def test_all_models_forward_and_predict(synthetic_dir, tiny_cfg):
    batch, D, Dc = _batch(synthetic_dir)
    for name in MODEL_NAMES:
        model = build_model(name, D, Dc, tiny_cfg["model"])
        out = model(batch)
        assert torch.isfinite(out["loss"])
        if name != "persistence":
            out["loss"].backward()
        pred = model.predict(batch)
        assert pred.shape == batch["X"].shape


def test_predict_does_not_use_targets(synthetic_dir, tiny_cfg):
    """Forecasts must be invariant to the (held-out) target values."""
    batch, D, Dc = _batch(synthetic_dir)
    corrupted = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    corrupted["X"][batch["target"]] = 1e3
    for name in MODEL_NAMES:
        torch.manual_seed(0)
        model = build_model(name, D, Dc, tiny_cfg["model"]).eval()
        a, b = model.predict(batch), model.predict(corrupted)
        m = batch["target"]
        assert torch.allclose(a[m], b[m], atol=1e-5), name


def test_batched_aligned_ode_matches_per_subject_solve(synthetic_dir, tiny_cfg):
    """Solving in shared elapsed time == solving each subject on its own aligned timeline (Eq. 7)."""
    batch, D, Dc = _batch(synthetic_dir, n=4)
    cfg = dict(tiny_cfg["model"])
    cfg["cnode"] = {**cfg["cnode"], "ode_method": "dopri5"}
    model = build_model("cnode", D, Dc, cfg).eval()
    with torch.no_grad():
        for p in model.align.parameters():
            p.normal_(0, 0.3)
        tau, gamma = model.alignment(batch)
        z0 = torch.randn(4, cfg["cnode"]["latent_dim"])
        z = model.integrate(z0, batch["times"], tau, gamma)
        g = model.odefunc.g
        for b in range(4):
            n = int(batch["valid"][b].sum())
            t_al = model.align_times(batch["times"], tau, gamma)[b, :n]
            ref = odeint(lambda t, y: g(torch.cat([y, t.expand(1, 1)], -1)), z0[b:b + 1], t_al,
                         rtol=1e-6, atol=1e-7)
            assert torch.allclose(z[b, :n], ref[:, 0], atol=1e-3)


def test_contrastive_loss_moves_aligned_times():
    torch.manual_seed(0)
    x = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    sid = torch.tensor([0, 1, 2, 3])
    cond = torch.ones(4, 3)
    t = torch.tensor([0.0, 2.0, 0.1, 2.1], requires_grad=True)
    loss = metadata_weighted_infonce(x, t, cond, sid, delta=0.8, temperature=1.0)
    loss.backward()
    # Positives (0,1) and (2,3) should be pulled together.
    assert t.grad[0] < 0 < t.grad[1]
    assert t.grad[2] < 0 < t.grad[3]


def test_cv_smoke(synthetic_dir, tiny_cfg, tmp_path):
    summary = run_cv(str(synthetic_dir), str(tmp_path / "run"), ["cnode", "gru", "persistence"], tiny_cfg)
    assert set(summary["model"]) == {"cnode", "gru", "persistence"}
    assert np.all(np.isfinite(summary["mse_mean"]))
    assert (tmp_path / "run" / "results_table.md").exists()
