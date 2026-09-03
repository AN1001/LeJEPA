import torch
from lejepa.config import SigregConfig


def char_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Computes the value of the empirical characteristic
    function of X at every t in T, i.e. 𝜙_X(𝑡) for every t in T.

    X: (points) or (B, points)
    t: (t)
    """

    # Turn into batch if single set of points
    if len(x.shape) == 1:
        x = x.unsqueeze(0)

    # Unsqueeze for broadcasting. Multiply every set in batch by every t.
    tx = x.unsqueeze(1) * t.view(1, -1, 1)  # (B, t, num points)

    # Compute characteristic function at every t for every el in batch
    # 𝜙_X(𝑡) = E[e^itx] = 1/N * Σe^itx
    return torch.exp(1j * tx).mean(2)  # (B, t)


def epps_pulley(x, config: SigregConfig = SigregConfig()) -> torch.Tensor:
    """
    Estimates the value of the Epps Pulley statistic:
        N∫|𝜙_X(𝑡) - 𝜙(t)|²w(t)dt
    using the trapezoid rule for target cf: 𝜙(t)=N(0, 1)

    X: (points) or (B, points)
    """
    # Turn into batch if single set of points
    if len(x.shape) == 1:
        x = x.unsqueeze(0)

    # Work out number of points in each set
    n = x.size(1)

    # Range of t values for estimating the integral
    t = torch.linspace(-config.grid_max, config.grid_max, config.grid_points, device=x.device)

    # Work out characteristic functions 𝜙(𝑡) and 𝜙_X(𝑡);
    # target cf happens to be the same as the weight w(t) in LeJEPA
    target_cf = w = torch.exp(-0.5 * t**2)  # 𝜙(𝑡)
    ecf = char_fn(x, t)  # 𝜙_X(𝑡)

    y = (target_cf - ecf).abs().square().mul(w)  #  y = |𝜙_X(𝑡) - 𝜙(t)|²w(t)
    return n * torch.trapezoid(y, t, dim=1)  # EP = N∫ydt


# SIGReg!!!!
def sigreg(z, config: SigregConfig = SigregConfig()) -> torch.Tensor:
    """
    Sketches the discrepancy between the distribution of the points in Z and N(0, 1).

    Projects Z onto [dirs] (e.g. 64) random directions and returns the average of the
    Epps Pulley statistic of each 1D projection against N(0, 1).
    """
    # Enforce fp32 since Epps Pulley requires high
    # accuracy due to small values
    z = z.float()

    # Generate [dirs] random unit vectors, [A], on hypersphere S^(N-1)
    dims = z.size(1)
    a = torch.randn(config.dirs, dims, device=z.device)
    a = a / torch.linalg.vector_norm(a, dim=1, keepdim=True)

    # For each direction project every point in [Z] onto it to get
    # a line of points (which we check is N(0, 1) distributed)
    x = a @ z.T  # ([dirs], num points)

    return epps_pulley(x, config).mean()
