"""NeurHL 1.1 C2b: in-season standings projection (PLAN_NeurHL_1_1).

The frozen NeurHL 1.0 season files are preseason distributions. Once games
are played, the season is partly known, and so is how far each team is from
its preseason strength. This module projects the rest of a season:

  1. Posterior team strength. Each team's shock s (sd sigma0 on the log goal
     rate, the season layer's prior) is updated from the games already
     played. The update is a Laplace approximation to the logistic model
     P(home win) = sigmoid(z_i + k_i (s_home - s_away)): the posterior mode
     and Hessian of a Gaussian-penalised logistic regression. z_i is the
     frozen preseason logit and k_i the engine's rate sensitivity.
  2. Drift. Strength keeps moving after the last game played, as a random
     walk with weekly sd sw, so later games are less certain.
  3. Simulation. The remaining games are simulated with shocks drawn from the
     posterior plus drift. Each game's regulation/overtime split comes from
     its frozen outcome4, and the points already earned are added.

A game's own frozen probability is never changed; only the strength shock
moves.
"""
import numpy as np


def posterior(zp, kp, hp, ap, yp, T, sigma0, iters=30):
    """Mode and covariance of s given played games (logit zp, sensitivity kp,
    team indices hp/ap, home-win outcomes yp)."""
    s = np.zeros(T)
    prec0 = 1.0 / sigma0 ** 2
    for _ in range(iters):
        eta = zp + kp * (s[hp] - s[ap])
        p = 1 / (1 + np.exp(-eta))
        g = np.zeros(T)
        np.add.at(g, hp, kp * (yp - p))
        np.add.at(g, ap, -kp * (yp - p))
        g -= prec0 * s
        w = kp * kp * p * (1 - p)
        H = np.diag(np.full(T, prec0))
        np.add.at(H, (hp, hp), w)
        np.add.at(H, (ap, ap), w)
        np.add.at(H, (hp, ap), -w)
        np.add.at(H, (ap, hp), -w)
        step = np.linalg.solve(H, g)
        s += step
        if np.abs(step).max() < 1e-9:
            break
    eta = zp + kp * (s[hp] - s[ap])
    p = 1 / (1 + np.exp(-eta))
    w = kp * kp * p * (1 - p)
    H = np.diag(np.full(T, prec0))
    np.add.at(H, (hp, hp), w)
    np.add.at(H, (ap, ap), w)
    np.add.at(H, (hp, ap), -w)
    np.add.at(H, (ap, hp), -w)
    return s, np.linalg.inv(H)


def project(zr, kr, hr, ar, o4r, week_r, week_now, pts_now, s_mean, s_cov, sw, sims, seed):
    """Final points per team (sims x T) for the remaining games (logit zr,
    sensitivity kr, team indices hr/ar, frozen outcome4 o4r, week of each game),
    given points already earned and the posterior of s."""
    rng = np.random.default_rng(seed)
    T = len(pts_now)
    L = np.linalg.cholesky(s_cov + 1e-12 * np.eye(T))
    s0 = s_mean[None, :] + rng.standard_normal((sims, T)) @ L.T
    if len(zr) == 0:
        return np.repeat(pts_now[None, :], sims, 0)
    dw = np.maximum(week_r - week_now, 0)
    W = int(dw.max()) + 1
    if sw > 0:
        walk = np.cumsum(rng.normal(0, sw, (sims, W, T)), axis=1)
        walk = np.concatenate([np.zeros((sims, 1, T)), walk[:, :-1]], axis=1)
        S = s0[:, None, :] + walk
        sh, sa = S[:, dw, hr], S[:, dw, ar]
    else:
        sh, sa = s0[:, hr], s0[:, ar]
    z = zr[None, :] + kr[None, :] * (sh - sa)
    hw = rng.random(z.shape) < 1 / (1 + np.exp(-z))
    rh = o4r[:, 0] / np.maximum(o4r[:, 0] + o4r[:, 2], 1e-9)
    ra = o4r[:, 1] / np.maximum(o4r[:, 1] + o4r[:, 3], 1e-9)
    reg = np.where(hw, rng.random(z.shape) < rh[None, :], rng.random(z.shape) < ra[None, :])
    hp_ = np.where(hw, 2, np.where(reg, 0, 1))
    ap_ = np.where(~hw, 2, np.where(reg, 0, 1))
    Hm = np.zeros((len(zr), T))
    Hm[np.arange(len(zr)), hr] = 1
    Am = np.zeros((len(zr), T))
    Am[np.arange(len(zr)), ar] = 1
    return pts_now[None, :] + hp_ @ Hm + ap_ @ Am
