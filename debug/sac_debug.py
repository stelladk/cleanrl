"""Debug metrics for sac_continuous_action.py."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# ---------------
# Compute metrics
# ---------------

def compute_grad_norm(*modules_or_tensors) -> float:
    """L2 norm over all gradients of the given modules / leaf tensors.
    Call right after .backward() and before .step().

    Returns
    -------
    float
        gradient norm
    """
    total = 0.0
    for item in modules_or_tensors:
        params = item.parameters() if isinstance(item, nn.Module) else [item]
        for p in params:
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
    return total ** 0.5


def effective_rank(activations: torch.Tensor, eps: float = 1e-12) -> float:
    """Entropy-based effective rank of an activation matrix.

    Parameters
    ----------
    activations : torch.Tensor
        batch of activation vectors from a layer, shape (B, d)
    eps : float, optional
        numerical floor used to avoid log(0) and division by zero, by default 1e-12

    Returns
    -------
    float
        scalar effective rank in [1, min(B, d)]
    """
    # Center the activations (optional but common — removes mean shift)
    H = activations - activations.mean(dim=0, keepdim=True)

    # Singular values of H are sqrt(eigenvalues) of its Gram matrix. Forming
    # the Gram matrix on the smaller side (B x B if B <= d, else d x d) and
    # taking eigvalsh (symmetric-specialized, values-only) is cheaper than a
    # full SVD of H whenever B != d.
    B, d = H.shape
    gram = H @ H.T if B <= d else H.T @ H
    eigvals = torch.linalg.eigvalsh(gram).clamp(min=0)
    s = eigvals.sqrt()

    # Normalize to probability distribution
    s_sum = s.sum()
    if s_sum < eps:
        return 1.0
    p = s / s_sum

    # Shannon entropy of the singular value distribution, exponentiated.
    # Filter zeros to avoid log(0)
    p_nonzero = p[p > eps]
    entropy = -(p_nonzero * torch.log(p_nonzero)).sum()
    return torch.exp(entropy).item()


def _capture_activations(model: nn.Module, *forward_args) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Run `model(*forward_args)` once and return the outputs of every
    `nn.Linear` and activation (`nn.ReLU`/`nn.SELU`) submodule, each keyed by
    its name from `named_modules()`.

    Architecture-agnostic: discovers layers by type rather than assuming
    fixed names or depth, so it works for SoftQNetwork, Actor, or any other
    module built from `nn.Linear` and `nn.ReLU`/`nn.SELU` instances (rather
    than the functional `F.relu`/`F.selu`) — including the `--selu` variants
    of those networks. Hooking both module types in a single forward pass
    lets `effective_rank_per_layer`, `dead_relu_rate`, `active_neuron_ratio`,
    and `log_layer_activation_stats` share one pass instead of two.

    Parameters
    ----------
    model : nn.Module
        network to probe (called as `model(*forward_args)`)
    *forward_args : torch.Tensor
        positional inputs to `model.forward`

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        (linear_activations, activation_activations), each mapping submodule
        name to its output
    """
    linear_acts: dict[str, torch.Tensor] = {}
    relu_acts: dict[str, torch.Tensor] = {}

    def make_hook(store, name):
        def hook(_module, _input, output):
            store[name] = output.detach()
        return hook

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(make_hook(linear_acts, name)))
        elif isinstance(m, (nn.ReLU, nn.SELU)):
            handles.append(m.register_forward_hook(make_hook(relu_acts, name)))

    try:
        with torch.no_grad():
            model(*forward_args)
    finally:
        for h in handles:
            h.remove()

    return linear_acts, relu_acts


def effective_rank_per_layer(model: nn.Module, *forward_args) -> dict[str, float]:
    """Effective rank of every `nn.Linear` submodule's output in `model`.

    Parameters
    ----------
    model : nn.Module
        network to probe (called as `model(*forward_args)`)
    *forward_args : torch.Tensor
        positional inputs to `model.forward`

    Returns
    -------
    dict[str, float]
        effective rank of each `nn.Linear` submodule's output, keyed by its
        name from `named_modules()`
    """
    linear_acts, _ = _capture_activations(model, *forward_args)
    return {name: effective_rank(act) for name, act in linear_acts.items()}


def dead_relu_rate(model: nn.Module, *forward_args, threshold: float = 0.0) -> dict[str, dict[str, float]]:
    """Fraction of dead units per `nn.ReLU` layer of `model`.

    A unit is "dead" if its post-ReLU output never exceeds `threshold` across
    the batch (for `threshold = 0`, it never fires at all). Only layers that
    use an `nn.ReLU` module (not the functional `F.relu`) are reported.

    Parameters
    ----------
    model : nn.Module
        network to probe (called as `model(*forward_args)`)
    *forward_args : torch.Tensor
        positional inputs to `model.forward`
    threshold : float, optional
        activation level at/below which a unit counts as off, by default 0.0

    Returns
    -------
    dict[str, dict[str, float]]
        per `nn.ReLU` layer: 'dead_unit_fraction' (units that are off for
        the whole batch) and 'zero_activation_fraction' (fraction of all
        (sample, unit) pairs at or below `threshold`)
    """
    _, relu_acts = _capture_activations(model, *forward_args)
    rates = {}
    for name, act in relu_acts.items():
        max_per_unit = act.max(dim=0).values  # shape: (d,)
        rates[name] = {
            "dead_unit_fraction": (max_per_unit <= threshold).float().mean().item(),
            "zero_activation_fraction": (act <= threshold).float().mean().item(),
        }
    return rates


def active_neuron_ratio(model: nn.Module, *forward_args, tau: float = 0.1) -> dict[str, float]:
    """Fraction of "active" (non-dormant) units per activation layer of
    `model`, following the normalized dormant-neuron score of Sokar et al.
    (https://arxiv.org/abs/2302.12902) eq. (1): a unit i in layer l is
    tau-dormant if `s_i = mean_batch(|h_i|) / mean_units(mean_batch(|h|)) <= tau`,
    i.e. its average magnitude is at most a `tau` fraction of its layer's
    average unit magnitude. Normalizing per layer (rather than thresholding
    raw activation values, as `dead_relu_rate` does) makes the threshold
    comparable across layers of different width and activation scale, and
    also catches units that fire but only weakly.

    Parameters
    ----------
    model : nn.Module
        network to probe (called as `model(*forward_args)`)
    *forward_args : torch.Tensor
        positional inputs to `model.forward`
    tau : float, optional
        dormancy threshold on the normalized score, by default 0.1 (`tau=0.0`
        recovers "never fires at all")

    Returns
    -------
    dict[str, float]
        fraction of active units per activation layer, keyed by its name from
        `named_modules()`, plus an 'overall' entry pooling all activation
        layers together
    """
    _, relu_acts = _capture_activations(model, *forward_args)
    ratios = {}
    total_active, total_units = 0.0, 0
    for name, act in relu_acts.items():
        mean_abs_per_unit = act.abs().mean(dim=0)  # shape: (d,)
        layer_avg = mean_abs_per_unit.mean().clamp(min=1e-12)
        is_active = (mean_abs_per_unit / layer_avg) > tau
        ratios[name] = is_active.float().mean().item()
        total_active += is_active.sum().item()
        total_units += is_active.numel()
    if total_units > 0:
        ratios["overall"] = total_active / total_units
    return ratios


def gradient_lipschitz(critic: nn.Module, observations: torch.Tensor, actions: torch.Tensor) -> dict[str, float]:
    """Local Lipschitz estimate of `critic` via its input-gradient norm.

    ||grad_(s,a) Q(s,a)|| bounds how much Q can change for a small input
    perturbation — a single extra forward+backward pass, much cheaper than
    estimating a Lipschitz constant via finite differences.

    Parameters
    ----------
    critic : nn.Module
        SoftQNetwork-like module called as `critic(observations, actions)`
    observations : torch.Tensor
        batch of observations, shape (B, obs_dim)
    actions : torch.Tensor
        batch of actions, shape (B, action_dim)

    Returns
    -------
    dict[str, float]
        'grad_lipschitz_max', 'grad_lipschitz_p99', 'grad_lipschitz_mean' of
        the per-sample input-gradient norm over the batch
    """
    s = observations.clone().requires_grad_(True)
    a = actions.clone().requires_grad_(True)

    q = critic(s, a).sum()
    grad_s, grad_a = torch.autograd.grad(q, [s, a])
    g_norm = torch.cat([grad_s, grad_a], dim=-1).norm(dim=-1)

    return {
        "grad_lipschitz_max": g_norm.max().item(),
        "grad_lipschitz_p99": g_norm.quantile(0.99).item(),
        "grad_lipschitz_mean": g_norm.mean().item(),
    }


def effective_scale_normalized_layers(model: nn.Module) -> dict[str, dict[str, float]]:
    """Effective scale gamma/sigma of every BatchNorm/LayerNorm layer in `model`.

    For BatchNorm this is gamma / sqrt(running_var + eps), the factor actually
    applied to the normalized activation; for LayerNorm it is just gamma,
    since LayerNorm already normalizes per-sample. A shrinking effective scale
    over training indicates the normalization layer is suppressing its input.

    Parameters
    ----------
    model : nn.Module
        network to scan for BatchNorm1d/BatchNorm2d/LayerNorm submodules

    Returns
    -------
    dict[str, dict[str, float]]
        per normalization layer: 'mean_abs', 'max_abs', 'std' of the scale.
        Empty if `model` has no normalization layers.
    """
    scales = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            gamma = module.weight.detach()
            sigma = torch.sqrt(module.running_var + module.eps)
            scale = gamma / sigma
        elif isinstance(module, nn.LayerNorm):
            scale = module.weight.detach()
        else:
            continue

        scales[name] = {
            "mean_abs": scale.abs().mean().item(),
            "max_abs": scale.abs().max().item(),
            "std": scale.std().item(),
        }

    return scales


def weight_norm_per_layer(model: nn.Module) -> dict[str, float]:
    """L2 norm of every `nn.Linear` layer's weight matrix in `model`.

    Pure parameter property — no forward pass needed, so cheap enough to log
    frequently. Tracks how each layer's weight scale evolves over training.

    Parameters
    ----------
    model : nn.Module
        network to scan for `nn.Linear` submodules

    Returns
    -------
    dict[str, float]
        weight-matrix L2 norm of each `nn.Linear` submodule, keyed by its
        name from `named_modules()`
    """
    return {name: m.weight.detach().norm().item() for name, m in model.named_modules() if isinstance(m, nn.Linear)}


def hessian_top_eigenvalue(
    loss_fn: Callable[[], torch.Tensor],
    params: list[torch.Tensor],
    n_iters: int = 10,
    tol: float = 1e-3,
) -> float:
    """Top eigenvalue of the Hessian of `loss_fn()` w.r.t. `params`, via power
    iteration on Hessian-vector products.

    `loss_fn()` is called once to build a `create_graph=True` first-order
    gradient; every iteration then reuses that retained graph for a single
    cheap `Hv = d/dtheta (grad . v)` backward, instead of rerunning the
    forward pass and first-order gradient at every step.

    Parameters
    ----------
    loss_fn : Callable[[], torch.Tensor]
        returns a fresh scalar loss when called (no args — close over data)
    params : list[torch.Tensor]
        parameters to differentiate against
    n_iters : int, optional
        number of power-iteration steps, by default 10
    tol : float, optional
        relative-change stopping tolerance between iterations, by default 1e-3

    Returns
    -------
    float
        estimated top eigenvalue (Rayleigh quotient at the final iterate)
    """
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Random initial direction
    v = [torch.randn_like(p) for p in params]
    v_norm = torch.sqrt(sum((vi * vi).sum() for vi in v))
    v = [vi / v_norm for vi in v]

    eigenvalue = 0.0
    for _ in range(n_iters):
        # Hv = d/dtheta (grad . v), reusing the retained `grads` graph
        grad_dot_v = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
        Hv = [hv.detach() for hv in Hv]

        # Rayleigh quotient
        new_eigenvalue = sum((hv * vi).sum() for hv, vi in zip(Hv, v)).item()

        # Renormalize for the next iteration
        Hv_norm = torch.sqrt(sum((hv * hv).sum() for hv in Hv))
        v = [hv / Hv_norm for hv in Hv]

        if abs(new_eigenvalue - eigenvalue) < tol * (abs(eigenvalue) + 1e-8):
            eigenvalue = new_eigenvalue
            break
        eigenvalue = new_eigenvalue

    return eigenvalue


def normalized_sharpness(
    critic: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    n_iters: int = 10,
) -> dict[str, float]:
    """Scale-invariant sharpness lambda_max(H) / ||theta||^2 of `critic` under `loss_fn`.

    Dividing the top Hessian eigenvalue by the squared parameter norm makes
    the measure invariant to a uniform rescaling of the weights, which would
    otherwise dominate raw sharpness comparisons across training.

    Parameters
    ----------
    critic : nn.Module
        network whose parameters define the Hessian
    loss_fn : Callable[[], torch.Tensor]
        returns a fresh scalar loss when called (no args — close over data)
    n_iters : int, optional
        power-iteration steps passed to `hessian_top_eigenvalue`, by default 10

    Returns
    -------
    dict[str, float]
        'hessian_top_eigenvalue' and 'normalized_sharpness'
    """
    params = [p for p in critic.parameters() if p.requires_grad]
    lambda_max = hessian_top_eigenvalue(loss_fn, params, n_iters)
    theta_sq = sum((p.detach() ** 2).sum() for p in params).item()

    return {
        "hessian_top_eigenvalue": lambda_max,
        "normalized_sharpness": lambda_max / (theta_sq + 1e-8),
    }


def function_complexity(output_grid: torch.Tensor) -> float:
    """Frequency-weighted average of the 2D discrete Fourier spectrum
    magnitude of `output_grid`, per SimBa (https://arxiv.org/abs/2410.09754)
    Appendix A.1/B eq. (10)/(14): c(f) = sum_k |f-hat(k)| . |k| / sum_k |f-hat(k)|.

    The paper's complexity measure is defined for a 1D frequency index k; on a
    2D grid we use the radial frequency magnitude ||(kx, ky)|| as the per-bin
    weight, the natural 2D generalization.

    Parameters
    ----------
    output_grid : torch.Tensor
        (H, W) grid of scalar network outputs over a 2D input slice

    Returns
    -------
    float
        complexity measure c(f); lower means the function is smoother/simpler
    """
    spectrum = torch.fft.fft2(output_grid.double())
    magnitude = spectrum.abs()
    h, w = output_grid.shape
    ky = torch.fft.fftfreq(h, device=output_grid.device) * h
    kx = torch.fft.fftfreq(w, device=output_grid.device) * w
    k_norm = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    return (magnitude * k_norm).sum().item() / magnitude.sum().item()


def sample_function_grid(
    model: nn.Module,
    obs_dim: int,
    action_dim: int,
    grid_size: int = 300,
    input_range: float = 100.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Evaluate `model` over a `grid_size` x `grid_size` grid spanning
    X = [-input_range, input_range]^2, following the Neural Redshift / SimBa
    Appendix B methodology for estimating function complexity.

    SAC critics take a (obs_dim + action_dim)-dimensional input rather than
    the 2D input used in the paper's toy analysis, so the 2D grid coordinates
    are mapped into the critic's actual input space through two fixed random
    orthonormal directions (a random 2D slice through the origin).

    Parameters
    ----------
    model : nn.Module
        critic called as `model(observations, actions)`
    obs_dim : int
        observation dimension (first `obs_dim` columns of the projected input)
    action_dim : int
        action dimension (remaining columns of the projected input)
    grid_size : int, optional
        grid resolution per axis, by default 300 (90,000 points, as in the paper)
    input_range : float, optional
        grid spans [-input_range, input_range] per axis, by default 100.0
    device : torch.device, optional
        device to run the forward pass on, by default cpu

    Returns
    -------
    torch.Tensor
        (grid_size, grid_size) grid of scalar network outputs
    """
    coords = torch.linspace(-input_range, input_range, grid_size, device=device)
    x1, x2 = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack([x1.flatten(), x2.flatten()], dim=1)  # (grid_size^2, 2)

    # Fixed random orthonormal basis spanning a 2D subspace of the input space.
    basis = torch.linalg.qr(torch.randn(obs_dim + action_dim, 2, device=device))[0]
    inputs = grid @ basis.T  # (grid_size^2, obs_dim + action_dim)

    with torch.no_grad():
        outputs = model(inputs[:, :obs_dim], inputs[:, obs_dim:])

    return outputs.view(grid_size, grid_size)


def simplicity_bias_score(
    model_fn: Callable[[], nn.Module],
    obs_dim: int,
    action_dim: int,
    n_init: int = 100,
    grid_size: int = 300,
    input_range: float = 100.0,
    device: torch.device = torch.device("cpu"),
) -> float:
    """SimBa's simplicity bias score s(f), Appendix A.2/B eq. (13)/(15): the
    average, over `n_init` fresh random initializations of an architecture, of
    the inverse function-complexity (`function_complexity`) measured on a
    random 2D slice of the input space (`sample_function_grid`). Higher means
    the architecture is more biased toward simple, low-frequency functions at
    initialization — it is a property of the architecture and its init
    distribution, not of any particular trained network.

    Parameters
    ----------
    model_fn : Callable[[], nn.Module]
        returns a freshly, randomly initialized instance of the architecture
        each call, e.g. `lambda: make_critic(envs, args)`
    obs_dim : int
        observation dimension
    action_dim : int
        action dimension
    n_init : int, optional
        number of random initializations to average over, by default 100 (as in the paper)
    grid_size : int, optional
        forwarded to `sample_function_grid`, by default 300
    input_range : float, optional
        forwarded to `sample_function_grid`, by default 100.0
    device : torch.device, optional
        device to run the forward passes on, by default cpu

    Returns
    -------
    float
        simplicity bias score s(f)
    """
    scores = []
    for _ in range(n_init):
        model = model_fn().to(device)
        model.eval()
        grid = sample_function_grid(model, obs_dim, action_dim, grid_size, input_range, device)
        scores.append(1.0 / (function_complexity(grid) + 1e-12))
    return sum(scores) / len(scores)


def gradient_variance(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    n_batches: int = 8,
) -> dict[str, float]:
    """Variance of per-batch gradients of `model`, estimated from `n_batches`
    independent samples.

    `loss_fn()` should sample a fresh minibatch internally and return its
    scalar loss. A low gradient signal-to-noise ratio (high variance relative
    to the mean gradient) indicates noisy, low-signal updates.

    Leaves `model`'s `.grad` zeroed on return.

    Parameters
    ----------
    model : nn.Module
        network whose parameter gradients are measured
    loss_fn : Callable[[], torch.Tensor]
        samples a fresh minibatch and returns its scalar loss when called
    n_batches : int, optional
        number of independent batches to average over, by default 8

    Returns
    -------
    dict[str, float]
        'grad_mean_norm', 'grad_total_variance', 'grad_variance_to_mean_ratio',
        and 'grad_snr' (mean-gradient norm / total-variance std)
    """
    grads_list = []

    for _ in range(n_batches):
        model.zero_grad()
        loss = loss_fn()
        loss.backward()
        grads_list.append(torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]))

    model.zero_grad()

    G = torch.stack(grads_list)  # (n_batches, n_params)
    mean_g = G.mean(dim=0)
    var_g = G.var(dim=0)

    return {
        "grad_mean_norm": mean_g.norm().item(),
        "grad_total_variance": var_g.sum().item(),
        "grad_variance_to_mean_ratio": var_g.sum().item() / (mean_g.norm().item() ** 2 + 1e-8),
        "grad_snr": mean_g.norm().item() / (var_g.sum().item() ** 0.5 + 1e-8),
    }


# -----------------
# Logging functions
# -----------------

def log_grad_norms(
    writer: SummaryWriter,
    global_step: int,
    qf_grad_norm: float,
    actor_grad_norm: float,
    alpha_grad_norm: Optional[float] = None,
) -> None:
    writer.add_scalar("debug/qf_grad_norm", qf_grad_norm, global_step)
    writer.add_scalar("debug/actor_grad_norm", actor_grad_norm, global_step)
    if alpha_grad_norm is not None:
        writer.add_scalar("debug/alpha_grad_norm", alpha_grad_norm, global_step)


def log_q_stats(
    writer: SummaryWriter,
    global_step: int,
    qf1_a_values: torch.Tensor,
    qf2_a_values: torch.Tensor,
    next_q_value: torch.Tensor,
    prefix: str = "debug",
) -> None:
    writer.add_scalar(f"{prefix}/qf1_values_min", qf1_a_values.min().item(), global_step)
    writer.add_scalar(f"{prefix}/qf1_values_max", qf1_a_values.max().item(), global_step)
    writer.add_scalar(f"{prefix}/qf1_values_std", qf1_a_values.std().item(), global_step)
    writer.add_scalar(f"{prefix}/qf2_values_min", qf2_a_values.min().item(), global_step)
    writer.add_scalar(f"{prefix}/qf2_values_max", qf2_a_values.max().item(), global_step)
    writer.add_scalar(f"{prefix}/qf2_values_std", qf2_a_values.std().item(), global_step)
    writer.add_scalar(f"{prefix}/target_q_mean", next_q_value.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/target_q_std", next_q_value.std().item(), global_step)


def log_layer_activation_stats(
    writer: SummaryWriter,
    global_step: int,
    qf1: nn.Module,
    qf2: nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    prefix: str = "debug",
    dead_threshold: float = 0.0,
    active_tau: float = 0.1,
) -> None:
    """Log effective rank of every `nn.Linear` layer's output, dead-unit
    fractions, and active-neuron ratio (`active_neuron_ratio`) of every
    activation layer, for qf1 and qf2, from one forward pass per network."""
    for name, model in (("qf1", qf1), ("qf2", qf2)):
        linear_acts, relu_acts = _capture_activations(model, observations, actions)
        for layer_name, act in linear_acts.items():
            writer.add_scalar(f"{prefix}/{name}_effective_rank_{layer_name}", effective_rank(act), global_step)
        total_active, total_units = 0.0, 0
        for layer_name, act in relu_acts.items():
            max_per_unit = act.max(dim=0).values
            writer.add_scalar(
                f"{prefix}/{name}_dead_unit_frac_{layer_name}",
                (max_per_unit <= dead_threshold).float().mean().item(),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/{name}_zero_act_frac_{layer_name}",
                (act <= dead_threshold).float().mean().item(),
                global_step,
            )
            mean_abs_per_unit = act.abs().mean(dim=0)
            layer_avg = mean_abs_per_unit.mean().clamp(min=1e-12)
            is_active = (mean_abs_per_unit / layer_avg) > active_tau
            writer.add_scalar(
                f"{prefix}/{name}_active_neuron_ratio_{layer_name}",
                is_active.float().mean().item(),
                global_step,
            )
            total_active += is_active.sum().item()
            total_units += is_active.numel()
        if total_units > 0:
            writer.add_scalar(f"{prefix}/{name}_active_neuron_ratio_overall", total_active / total_units, global_step)


def log_gradient_lipschitz(
    writer: SummaryWriter,
    global_step: int,
    qf1: nn.Module,
    qf2: nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    prefix: str = "debug",
) -> None:
    """Log the input-gradient-norm Lipschitz estimate for qf1 and qf2."""
    for name, critic in (("qf1", qf1), ("qf2", qf2)):
        for stat_name, value in gradient_lipschitz(critic, observations, actions).items():
            writer.add_scalar(f"{prefix}/{name}_{stat_name}", value, global_step)


def log_normalization_scales(
    writer: SummaryWriter,
    global_step: int,
    qf1: nn.Module,
    qf2: nn.Module,
    prefix: str = "debug",
) -> None:
    """Log BatchNorm/LayerNorm effective-scale stats for qf1 and qf2.
    No-op for networks without normalization layers."""
    for name, model in (("qf1", qf1), ("qf2", qf2)):
        for layer_name, stats in effective_scale_normalized_layers(model).items():
            for stat_name, value in stats.items():
                writer.add_scalar(f"{prefix}/{name}_norm_scale_{layer_name}_{stat_name}", value, global_step)


def log_weight_norms(
    writer: SummaryWriter,
    global_step: int,
    qf1: nn.Module,
    qf2: nn.Module,
    prefix: str = "debug",
) -> None:
    """Log per-layer `nn.Linear` weight-matrix norms for qf1 and qf2."""
    for name, model in (("qf1", qf1), ("qf2", qf2)):
        for layer_name, value in weight_norm_per_layer(model).items():
            writer.add_scalar(f"{prefix}/{name}_weight_norm_{layer_name}", value, global_step)


def log_sharpness_stats(
    writer: SummaryWriter,
    global_step: int,
    critic: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    name: str = "qf1",
    n_iters: int = 10,
    prefix: str = "debug",
) -> None:
    """Log `critic`'s normalized sharpness (Hessian top eigenvalue / ||theta||^2)
    under `{prefix}/{name}_*`. Requires extra backward passes through `critic`
    — call at a lower frequency than the other debug metrics."""
    for stat_name, value in normalized_sharpness(critic, loss_fn, n_iters).items():
        writer.add_scalar(f"{prefix}/{name}_{stat_name}", value, global_step)


def log_gradient_variance_stats(
    writer: SummaryWriter,
    global_step: int,
    critic: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    name: str = "qf1",
    n_batches: int = 8,
    prefix: str = "debug",
) -> None:
    """Log `critic`'s per-batch gradient variance/SNR under `{prefix}/{name}_*`.
    Requires `n_batches` extra forward+backward passes through `critic` — call
    at a lower frequency than the other debug metrics."""
    for stat_name, value in gradient_variance(critic, loss_fn, n_batches).items():
        writer.add_scalar(f"{prefix}/{name}_{stat_name}", value, global_step)


def log_action_stats(
    writer: SummaryWriter,
    global_step: int,
    actions: torch.Tensor,
    prefix: str = "debug",
) -> None:
    """Log mean/std of sampled action values, e.g. for tracking the actor's
    action distribution on a fixed set of states."""
    writer.add_scalar(f"{prefix}/action_mean", actions.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/action_std", actions.std().item(), global_step)


def log_log_pi_stats(
    writer: SummaryWriter,
    global_step: int,
    log_pi: torch.Tensor,
    next_state_log_pi: torch.Tensor,
    prefix: str = "debug",
) -> None:
    writer.add_scalar(f"{prefix}/log_pi_mean", log_pi.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/log_pi_std", log_pi.std().item(), global_step)
    writer.add_scalar(f"{prefix}/next_state_log_pi_mean", next_state_log_pi.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/next_state_log_pi_std", next_state_log_pi.std().item(), global_step)


def log_losses(
    writer: SummaryWriter,
    global_step: int,
    qf1_loss: float,
    qf2_loss: float,
    actor_loss: float,
    prefix: str = "debug",
) -> None:
    writer.add_scalar(f"{prefix}/qf1_loss", qf1_loss, global_step)
    writer.add_scalar(f"{prefix}/qf2_loss", qf2_loss, global_step)
    writer.add_scalar(f"{prefix}/qf_loss", (qf1_loss + qf2_loss) / 2.0, global_step)
    writer.add_scalar(f"{prefix}/actor_loss", actor_loss, global_step)


def log_simplicity_bias(
    writer: SummaryWriter,
    global_step: int,
    model_fn: Callable[[], nn.Module],
    obs_dim: int,
    action_dim: int,
    n_init: int = 100,
    grid_size: int = 300,
    input_range: float = 100.0,
    device: torch.device = torch.device("cpu"),
    name: str = "critic",
    prefix: str = "debug",
) -> None:
    """Log the architecture's simplicity bias score (`simplicity_bias_score`).
    This characterizes `model_fn`'s architecture/init distribution rather than
    any particular trained network, so it's typically logged once rather than
    on a training cadence."""
    score = simplicity_bias_score(model_fn, obs_dim, action_dim, n_init, grid_size, input_range, device)
    writer.add_scalar(f"{prefix}/{name}_simplicity_bias_score", score, global_step)


def log_generalization_gap(
    writer: SummaryWriter,
    global_step: int,
    train_qf1_loss: float,
    train_qf2_loss: float,
    train_actor_loss: float,
    holdout_qf1_loss: float,
    holdout_qf2_loss: float,
    holdout_actor_loss: float,
    prefix: str = "holdout",
) -> None:
    """gap = holdout_loss - train_loss; gap_ratio = gap / |train_loss|.
    A large positive gap/ratio indicates overfitting to the training data."""
    losses = {
        "qf1_loss": (train_qf1_loss, holdout_qf1_loss),
        "qf2_loss": (train_qf2_loss, holdout_qf2_loss),
        "qf_loss": ((train_qf1_loss + train_qf2_loss) / 2.0, (holdout_qf1_loss + holdout_qf2_loss) / 2.0),
        "actor_loss": (train_actor_loss, holdout_actor_loss),
    }
    for name, (train_loss, holdout_loss) in losses.items():
        gap = holdout_loss - train_loss
        writer.add_scalar(f"{prefix}/{name}_gap", gap, global_step)
        writer.add_scalar(f"{prefix}/{name}_gap_ratio", gap / (abs(train_loss) + 1e-8), global_step)
