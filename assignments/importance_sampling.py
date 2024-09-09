import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def p_dist(x: np.ndarray):
    return (
        stats.norm.pdf(x, loc=2, scale=1) * 0.3
        + stats.norm.pdf(x, loc=5, scale=2) * 0.4
        + stats.norm.pdf(x, loc=9, scale=1) * 0.3
    )


x_axis = np.linspace(-5, 20, num=1_000)
true_p = p_dist(x_axis)
q_uniform = stats.uniform.pdf(x_axis, loc=0, scale=15)
rng = np.random.default_rng(seed=1)

# Uniform proposal distribution
particles_uniform = rng.uniform(0, 15, size=(1_000,))
weights_uniform = p_dist(particles_uniform) / stats.uniform.pdf(
    particles_uniform, loc=0, scale=15
)
weights_uniform /= weights_uniform.sum()
resampled_uniform = rng.choice(particles_uniform, size=(1_000,), p=weights_uniform)

# Normal proposal distribution
particles_normal = rng.normal(loc=5, scale=4, size=(1_000,))
weights_normal = p_dist(particles_normal) / stats.norm.pdf(
    particles_normal, loc=5, scale=4
)
weights_normal /= weights_normal.sum()
resampled_normal = rng.choice(particles_normal, size=(1_000,), p=weights_normal)


def plot_samples(resamples: dict[str, np.ndarray], sizes=(20, 100, 1_000)):
    _, axs = plt.subplots(
        len(resamples), len(sizes), sharey=True, figsize=(10, 2 * len(sizes))
    )
    cmap = matplotlib.colormaps["Accent"]
    for i, ((name, rs), ax_row) in enumerate(zip(resamples.items(), axs)):
        c = cmap(i)
        for size, ax in zip(sizes, ax_row):
            ax.set_title(f"{name}, {size} samples")
            ax.plot(x_axis, true_p, label="$p(x)$")
            ax.hist(
                rs[:size], bins=20, label="Resampled density", density=True, color=c
            )
            ax.legend()
    plt.tight_layout()
    plt.show()


plot_samples({"Uniform": resampled_uniform, "Normal": resampled_normal})
