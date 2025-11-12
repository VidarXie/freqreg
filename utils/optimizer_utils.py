import torch


class AdamUniform(torch.optim.Optimizer):
    """
    Variant of Adam with uniform scaling by the second moment.

    Instead of dividing each component by the square root of its second moment,
    we divide all of them by the max.
    """

    def __init__(self, params, lr=0.1, betas=(0.9, 0.999)):
        defaults = dict(lr=lr, betas=betas)
        super(AdamUniform, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamUniform, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data

                g1.mul_(b1).add_(grad, alpha=1 - b1)
                g2.mul_(b2).add_(grad.square(), alpha=1 - b2)
                m1 = g1 / (1 - (b1 ** state["step"]))
                m2 = g2 / (1 - (b2 ** state["step"]))
                # This is the only modification we make to the original Adam algorithm
                gr = m1 / (1e-8 + m2.sqrt().max())
                p.data.sub_(gr, alpha=lr)


class AdamLD(torch.optim.Optimizer):
    """
    Variant of Adam with langvin dynamics
    """

    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), noise_factor=1.0):
        defaults = dict(lr=lr, betas=betas, noise_factor=noise_factor)
        super(AdamLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamLD, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            noise_factor = group["noise_factor"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data

                # Adam momentum and variance calculations
                g1.mul_(b1).add_(grad, alpha=1 - b1)
                g2.mul_(b2).add_(grad.square(), alpha=1 - b2)
                m1 = g1 / (1 - (b1 ** state["step"]))
                m2 = g2 / (1 - (b2 ** state["step"]))
                gr = m1 / (1e-8 + m2.sqrt())

                # --- This is the modified update part ---

                # 1. This is the standard Adam update step
                adam_step = gr.mul(lr)

                # 2. This is the SGLD-style noise term
                noise_std = lr * noise_factor
                noise = torch.randn_like(p.data) * noise_std

                # 3. Apply both: p.data = p.data - adam_step + noise
                p.data.sub_(adam_step)

                if noise_factor > 0:
                    p.data.add_(noise)
