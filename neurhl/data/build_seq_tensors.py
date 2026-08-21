"""NeurHL-2 — S1 sequence tensors: the stream in the form the model consumes.

One .npz per season holding flat per-event arrays plus game offsets, so a game's
sequence is a slice and the training loop never touches pandas. ~7.5M events over
2008-2026.

Three things are resolved here rather than in the training loop, because getting
them wrong is silent:

  * **The joint (type, team) vocabulary.** The simulator must emit "who does what
    next", not "what happens next" — a shot is only meaningful attached to a
    side. Modelling type and team as independent heads would let the model assign
    probability to combinations that never occur. The vocabulary is built from
    observed pairs, so impossible ones are simply absent.
  * **Actor as a SLOT index, not a player id.** The actor head is a softmax
    restricted to the players actually on the ice (PLAN_NeurHL2 S1), so the
    target must be a position within this event's on-ice set. Storing a raw
    player id would force that lookup into the hot loop and silently drop actors
    who are not on the recorded ice.
  * **Player index is global and stable across seasons**, so an embedding table
    means the same thing in 2009 and 2026.

Vantage safety: RAPM priors are attached per season from `rapm_prior_<V>`, which
is fit on seasons < V only. Nothing here reads a player's future.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/data/build_seq_tensors.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

H_ON = [f"h_on{i}" for i in range(7)]
A_ON = [f"a_on{i}" for i in range(7)]
SEASONS = list(range(2008, 2027))
N_SLOT = 14                      # 7 home (goalie + 6 skaters) + 7 away
VOCAB_PATH = TENSORS / "seq_vocab.json"


def build_vocab(seasons) -> dict:
    """Global player index and the joint (event_type, team) vocabulary."""
    pids, pairs = set(), set()
    for s in seasons:
        p = TENSORS / f"stream_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["event_type", "home_event"] + H_ON + A_ON)
        v = d[H_ON + A_ON].to_numpy(np.int64).ravel()
        pids |= set(int(x) for x in np.unique(v) if x > 0)
        pairs |= set(map(tuple, d[["event_type", "home_event"]]
                         .drop_duplicates().to_numpy().tolist()))
    players = sorted(pids)
    # index 0 is reserved for "empty slot" / unknown player
    pmap = {p: i + 1 for i, p in enumerate(players)}
    tt = sorted(pairs)
    ttmap = {f"{a}|{b}": i for i, (a, b) in enumerate(tt)}
    return {"players": pmap, "type_team": ttmap, "n_players": len(pmap) + 1,
            "n_type_team": len(ttmap)}


def tensorise(se: int, vocab: dict) -> dict:
    d = pd.read_parquet(TENSORS / f"stream_{se}.parquet")
    d = d.sort_values(["game_id", "seq"], kind="stable").reset_index(drop=True)

    gid = d.game_id.to_numpy(np.int64)
    starts = np.flatnonzero(np.concatenate([[True], gid[1:] != gid[:-1]]))
    offsets = np.concatenate([starts, [len(d)]]).astype(np.int64)

    pmap = vocab["players"]
    lut = np.zeros(max(pmap) + 2, np.int32)
    for k, v in pmap.items():
        lut[k] = v
    on = np.concatenate([d[H_ON].to_numpy(np.int64),
                         d[A_ON].to_numpy(np.int64)], axis=1)
    on_idx = lut[np.clip(on, 0, len(lut) - 1)].astype(np.int32)

    ttmap = vocab["type_team"]
    key = (d.event_type.astype(str) + "|" + d.home_event.astype(str))
    tt = key.map(ttmap).fillna(0).to_numpy(np.int16)

    # actor slot: where p1 sits in this event's on-ice set (-1 if not present)
    p1 = d.p1.to_numpy(np.int64)
    actor = np.full(len(d), -1, np.int8)
    for c in range(N_SLOT):
        m = (on[:, c] == p1) & (p1 > 0) & (actor < 0)
        actor[m] = c

    return {
        "offsets": offsets,
        "game_id": d.game_id.to_numpy(np.int64)[starts],
        "game_type": d.game_type.to_numpy(np.int8)[starts],
        "tt": tt,
        "etype": d.event_type.to_numpy(np.int8),
        "team": (d.home_event.to_numpy(np.int8) + 1),        # 0/1/2
        "zone": d.zone.to_numpy(np.int8),
        "stype": d.shot_type.to_numpy(np.int8),
        "strength": d.strength_key.to_numpy(np.int8),
        "score": (d.score_diff.to_numpy(np.int8) + 4),       # 0..8
        "period": d.period.to_numpy(np.int8),
        "dt": d.dt.to_numpy(np.float32),
        "t_rem": d.t_rem.to_numpy(np.float32),
        "xa": d.xa.to_numpy(np.float32),
        "ya": d.ya.to_numpy(np.float32),
        "has_xy": d.has_xy.to_numpy(np.int8),
        "xg": d.xg.to_numpy(np.float32),
        "has_xg": d.has_xg.to_numpy(np.int8),
        "n_for": d.n_for.to_numpy(np.int8),
        "n_against": d.n_against.to_numpy(np.int8),
        "on": on_idx,
        "actor": actor,
    }


def rapm_matrix(se: int, vocab: dict) -> np.ndarray:
    """[n_players+1, 5] prior features, vantage-correct for season `se`."""
    n = vocab["n_players"]
    M = np.zeros((n, 5), np.float32)
    p = TENSORS / f"rapm_prior_{se}.parquet"
    if not p.exists():
        return M
    r = pd.read_parquet(p)
    real = r[~r.is_replacement.astype(bool)]
    repl = r[r.is_replacement.astype(bool)]
    if len(repl):
        M[:] = np.array([repl.cf_off.mean(), repl.cf_def.mean(),
                         repl.g_off.mean(), repl.g_def.mean(),
                         repl.cf_off_se.mean()], np.float32)
    M[0] = 0.0
    pmap = vocab["players"]
    idx = real.player_id.map(pmap)
    ok = idx.notna()
    M[idx[ok].to_numpy(np.int64)] = real.loc[ok, ["cf_off", "cf_def", "g_off",
                                                  "g_def", "cf_off_se"]] \
        .to_numpy(np.float32)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=SEASONS)
    ap.add_argument("--rebuild-vocab", action="store_true")
    args = ap.parse_args()

    if args.rebuild_vocab or not VOCAB_PATH.exists():
        print("building vocabulary ...")
        vocab = build_vocab(SEASONS)
        VOCAB_PATH.write_text(json.dumps(vocab))
    else:
        vocab = json.loads(VOCAB_PATH.read_text())
        vocab["players"] = {int(k): v for k, v in vocab["players"].items()}
    print(f"players {vocab['n_players']:,}   (type,team) classes "
          f"{vocab['n_type_team']}\n")

    print(f"{'yr':>5} {'events':>9} {'games':>6} {'actorOK':>8} {'meanlen':>8} "
          f"{'rapm cov':>9}")
    for se in args.seasons:
        if not (TENSORS / f"stream_{se}.parquet").exists():
            continue
        t = tensorise(se, vocab)
        t["rapm"] = rapm_matrix(se, vocab)
        out = TENSORS / f"seq_{se}.npz"
        np.savez_compressed(out, **t)
        n_ev = len(t["etype"])
        n_g = len(t["offsets"]) - 1
        cov = float((t["rapm"][1:].any(1)).mean())
        print(f"{se:>5} {n_ev:>9,} {n_g:>6} {(t['actor'] >= 0).mean():>8.3f} "
              f"{n_ev / n_g:>8.1f} {cov:>9.3f}")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
