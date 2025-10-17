import torch


# ---------- robust weighting helpers ----------
def mad_scale(x, eps=1e-8):
    med = x.median()
    return (1.4826 * (x - med).abs().median()).clamp_min(eps)


def w_huber(z, delta=1.5):
    # convex; good default
    return torch.minimum(torch.ones_like(z), delta / (z.abs() + 1e-12))


def robust_weights_from_residuals(r, delta=1.5, detach=True):
    sigma = mad_scale(r)
    z = r / sigma
    w = w_huber(z, delta)
    return w.detach() if detach else w, sigma
