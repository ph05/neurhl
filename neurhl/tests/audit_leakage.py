"""NeurHL-2 — exhaustive leakage audit of every S1 model input.

Written after a real leak shipped: the on-ice encoder oriented its for/against
pooling by `home_next = (team[t+1] == home)`, which is the team component of the
target, verified at 1.0000 agreement. Gates E1 and E2 both PASSED with it
present, because a gate that scores the model on the same inputs the model saw
cannot see a leaked input. Only the downstream consumer, which must supply every
input itself, exposed it.

So the lesson is not "fix that one field". It is that realism gates are
structurally blind to leakage and a dedicated audit is required. This module is
that audit, and it runs two independent checks:

  **STATIC** — every model input must be built from the CURRENT index slice, not
  the next. The batcher slices `sl = [a, a+n)` for inputs and `nx = [a+1, a+1+n)`
  for targets; anything reaching the model from `nx` is flagged and must carry an
  explicit, argued exemption.

  **EMPIRICAL** — for the one exempted input (on-ice personnel, which the
  deployment process legitimately supplies before an event resolves), test
  whether it carries information about the next event BEYOND personnel. The
  sharp test is whether the change in on-ice composition between the current
  event and the next predicts the next event's TYPE. If the on-ice feed times a
  penalty's box-time to the penalty event itself, then "one fewer skater next"
  announces "penalty next", and the model would learn to read an announcement it
  will never receive at simulation time.

A third check guards the target side: no target may be reachable from an input by
construction (the `home_next` failure), tested as exact agreement between every
input column and every target column.

Run: uv run --no-project --python 3.12 --with numpy --with torch \
     --with "pandas<3" --with pyarrow python neurhl/tests/audit_leakage.py
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
from train.train_event_sim import (Season, era_vectors, make_batch,  # noqa: E402
                                   device, TRAIN, VAL)

# Inputs allowed to come from the next index, each with the reason it is safe.
NEXT_INDEX_EXEMPT = {
    "on_ctx": ("personnel for the upcoming interval; supplied by the deployment "
                "process (S3) before any event resolves, so it exists at "
                "simulation time. Must still carry no information about WHAT "
                "happens next beyond who is on the ice — checked empirically."),
}


def causality_test(season, eras, dev, game=0, cut=None) -> dict:
    """THE decisive test: corrupt the future, assert the inputs do not move.

    AST parsing tells you which slice a line uses; it cannot tell you whether a
    value derived from those slices reaches the model. This does. A copy of the
    season has every event from index `cut` onward replaced with garbage, the
    batch is rebuilt, and every model input at positions strictly before `cut-1`
    must be bit-identical. Anything that changes is reading the future, whatever
    the source code appears to say.
    """
    import copy
    a0, b0 = int(season.off[game]), int(season.off[game + 1])
    n = b0 - a0
    cut = cut if cut is not None else n // 2

    clean = make_batch([season], [(season, game)], eras, dev)
    dirty_season = copy.copy(season)
    dirty_season.d = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                      for k, v in season.d.items()}
    rng = np.random.default_rng(99)
    for k, v in dirty_season.d.items():
        if not isinstance(v, np.ndarray) or v.shape[:1] != (len(season.d["etype"]),):
            continue
        lo = a0 + cut
        if lo >= b0:
            continue
        blk = v[lo:b0]
        dirty_season.d[k][lo:b0] = rng.permutation(blk) if blk.ndim == 1 \
            else rng.permutation(blk, axis=0)
    dirty = make_batch([dirty_season], [(dirty_season, game)], eras, dev)

    moved, checked = [], []
    horizon = max(cut - 1, 0)
    for k, v in clean.items():
        if k.startswith("tgt_") or k in ("valid", "tgt_valid", "era"):
            continue
        if not torch.is_tensor(v) or v.dim() < 2:
            continue
        cv, dv = v[:, :horizon], dirty[k][:, :horizon]
        same = bool(torch.equal(cv, dv))
        checked.append(k)
        if not same:
            frac = float((cv != dv).float().mean())
            moved.append({"input": k, "frac_changed": round(frac, 5)})
    return {"game": game, "n_events": n, "cut": cut, "horizon": horizon,
            "inputs_checked": sorted(checked), "inputs_that_moved": moved,
            "clean": not moved}


def actor_coverage(season, eras, dev, n_games=120) -> dict:
    """How often is the NEXT event's actor already on the ice NOW?

    With the leak removed the actor head must choose from CURRENT personnel, so
    this is the ceiling on actor accuracy and needs to be stated, not assumed.
    """
    tot = res = 0
    for lo in range(0, n_games, 8):
        picks = [(season, g) for g in range(lo, min(lo + 8, n_games))]
        b = make_batch([season], picks, eras, dev)
        v = b["tgt_valid"]
        t = b["tgt_actor"][v]
        tot += int(v.sum())
        res += int((t >= 0).sum())
    return {"steps": tot, "actor_in_current_onice": res,
            "coverage": round(res / max(tot, 1), 4)}


def target_identity_check(b: dict) -> dict:
    """No input may agree with a target by construction."""
    v = b["tgt_valid"]
    tgts = {k: b[k][v].cpu().numpy() for k in b if k.startswith("tgt_")
            and k != "tgt_valid"}
    ins = {k: b[k][v].cpu().numpy() for k in b
           if not k.startswith("tgt_") and torch.is_tensor(b[k])
           and b[k].dim() == 2 and b[k].shape == v.shape}
    hits = []
    for ik, iv in ins.items():
        for tk, tv in tgts.items():
            if iv.shape != tv.shape:
                continue
            a = float(np.mean(iv.astype(np.float64) == tv.astype(np.float64)))
            if a > 0.98:
                hits.append({"input": ik, "target": tk, "agreement": round(a, 4)})
    # the specific historical failure: an input equal to the target's team
    vocab = json.loads((TENSORS / "seq_vocab.json").read_text())
    inv = {i: k for k, i in vocab["type_team"].items()}
    team_of_tgt = np.array([int(inv[int(t)].split("|")[1])
                            for t in tgts["tgt_tt"]])
    team_hits = []
    for ik, iv in ins.items():
        if iv.dtype.kind not in "if":
            continue
        a = float(np.mean((iv > 0.5) == (team_of_tgt == 1)))
        if max(a, 1 - a) > 0.98:
            team_hits.append({"input": ik, "agreement_with_target_team":
                              round(max(a, 1 - a), 4)})
    return {"identical_pairs": hits, "reveals_target_team": team_hits}


def onice_information_check(season, eras, dev, n_games=200) -> dict:
    """Does the NEXT on-ice set announce the next event TYPE?

    Compares the skater/goalie counts implied by `on_next` against the current
    event's counts. Under a clean feed the difference is a line change and says
    nothing about what happens next; under a leaky one it announces penalties
    (a skater vanishes) or empty-net play.
    """
    picks = [(season, g) for g in range(min(n_games, season.n_games))]
    rows = {"d_sk": [], "d_g": [], "tt": []}
    for lo in range(0, len(picks), 8):
        b = make_batch([season], picks[lo:lo + 8], eras, dev)
        v = b["tgt_valid"]
        on = b["on_ctx"]
        nh = (on[..., 1:7] > 0).sum(-1)
        na = (on[..., 8:14] > 0).sum(-1)
        gh = (on[..., 0] > 0).long()
        ga = (on[..., 7] > 0).long()
        d_sk = (nh - b["n_home"]) + (na - b["n_away"])
        d_g = (gh - b["g_home"]) + (ga - b["g_away"])
        rows["d_sk"].append(d_sk[v].cpu().numpy())
        rows["d_g"].append(d_g[v].cpu().numpy())
        rows["tt"].append(b["tgt_tt"][v].cpu().numpy())
    d = {k: np.concatenate(v) for k, v in rows.items()}

    vocab = json.loads((TENSORS / "seq_vocab.json").read_text())
    inv = {i: k for k, i in vocab["type_team"].items()}
    et = np.array([int(inv[int(t)].split("|")[0]) for t in d["tt"]])

    out = {"n": int(len(et))}
    for name, code in (("penalty", 10), ("goal", 7), ("faceoff", 3),
                       ("stoppage", 15)):
        base = float((et == code).mean())
        by = {}
        for val in sorted(set(d["d_sk"].tolist()))[:9]:
            m = d["d_sk"] == val
            if m.sum() < 200:
                continue
            by[int(val)] = {"n": int(m.sum()),
                            "rate": round(float((et[m] == code).mean()), 5),
                            "lift": round(float((et[m] == code).mean() / max(base, 1e-9)), 3)}
        out[name] = {"base_rate": round(base, 5), "by_delta_skaters": by,
                     "max_lift": round(max((v["lift"] for v in by.values()),
                                           default=1.0), 3)}
    out["delta_skater_dist"] = {int(k): int(v) for k, v in
                                zip(*np.unique(d["d_sk"], return_counts=True))}
    out["delta_goalie_dist"] = {int(k): int(v) for k, v in
                                zip(*np.unique(d["d_g"], return_counts=True))}
    return out


def main():
    dev = device()
    s = Season(VAL)
    eras, _ = era_vectors(TRAIN + [VAL])
    print("=" * 74)
    print("CAUSALITY TEST — corrupt the future, assert the inputs do not move")
    cts = [causality_test(s, eras, dev, game=g) for g in (0, 5, 17)]
    for ct in cts:
        print(f"  game {ct['game']}: {ct['n_events']} events, cut at "
              f"{ct['cut']}, {len(ct['inputs_checked'])} inputs checked -> "
              f"{'CLEAN' if ct['clean'] else 'MOVED: ' + str(ct['inputs_that_moved'])}")
    st = {"causality": cts}
    bad = [m for ct in cts for m in ct["inputs_that_moved"]]
    print(f"  inputs checked: {cts[0]['inputs_checked']}")
    b = make_batch([s], [(s, g) for g in range(24)], eras, dev)

    print("\n" + "=" * 74)
    print("TARGET-IDENTITY AUDIT — does any input equal a target?")
    ti = target_identity_check(b)
    print(f"  inputs identical to a target (>98%): "
          f"{ti['identical_pairs'] or 'NONE'}")
    print(f"  inputs revealing the target's TEAM (>98%): "
          f"{ti['reveals_target_team'] or 'NONE'}")

    print("\n" + "=" * 74)
    print("ACTOR COVERAGE — is the next actor already on the ice now?")
    ac = actor_coverage(s, eras, dev)
    print(f"  {ac['actor_in_current_onice']:,} of {ac['steps']:,} steps "
          f"= {ac['coverage']:.4f}")

    print("\n" + "=" * 74)
    print("EMPIRICAL AUDIT — does on_ctx announce the next event type?")
    oi = onice_information_check(s, eras, dev)
    print(f"  n = {oi['n']:,} held-out steps")
    print(f"  delta-skaters distribution: {oi['delta_skater_dist']}")
    print(f"  delta-goalie   distribution: {oi['delta_goalie_dist']}")
    for name in ("penalty", "goal", "faceoff", "stoppage"):
        r = oi[name]
        print(f"  {name:<9} base {r['base_rate']:.5f}  max lift "
              f"{r['max_lift']:.3f}")
        for k, v in r["by_delta_skaters"].items():
            print(f"      d_sk={k:>+3}  n={v['n']:>7,}  rate {v['rate']:.5f}"
                  f"  lift {v['lift']:.3f}")

    out = {"causality": st, "actor_coverage": ac,
           "target_identity": ti, "onice_information": oi,
           "exempt": {k: v[0] for k, v in NEXT_INDEX_EXEMPT.items()}}
    p = Path(__file__).resolve().parents[1] / "configs" / "leakage_audit.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(f"\n-> {p}")
    clean = (not bad and not ti["identical_pairs"]
             and not ti["reveals_target_team"])
    print(f"\nSTATIC + IDENTITY: {'CLEAN' if clean else 'LEAK PRESENT'}")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
