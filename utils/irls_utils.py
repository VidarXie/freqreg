import torch


# ---------- robust weighting helpers ----------
def mad_scale(x, eps=1e-8):
    med = x.median()
    return (1.4826 * (x - med).abs().median()).clamp_min(eps)


def w_tukey(z, c=2.5):
    # redescending; outliers → weight 0
    w = torch.zeros_like(z)
    m = z.abs() <= c
    zc = z[m] / c
    w[m] = (1 - zc**2) ** 2
    return w


def w_cauchy(z, c=2.0):
    # smooth heavy-tail
    return 1.0 / (1.0 + (z / c) ** 2)


def w_huber(z, delta=1.5):
    # convex; good default
    return torch.minimum(torch.ones_like(z), delta / (z.abs() + 1e-12))


def robust_weights_from_residuals(r, scheme="tukey", c=2.5, delta=1.5, detach=True):
    sigma = mad_scale(r)
    z = r / sigma
    if scheme == "tukey":
        w = w_tukey(z, c)
    elif scheme == "cauchy":
        w = w_cauchy(z, c)
    elif scheme == "huber":
        w = w_huber(z, delta)
    else:
        raise ValueError("scheme must be tukey|cauchy|huber")
    return w.detach() if detach else w, sigma
