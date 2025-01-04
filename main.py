import os
from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.utils import entropy, mi
from src.blahut_arimoto import blahut_arimoto

sns.set_style("darkgrid")
plot_dir = "./plots"
plot_kwargs = {"linewidth": 2.5}
colors = ["#D81C8D", "#5C1CCD"]

def save_fig(fig, name, formats=["png"]):
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(os.path.join(plot_dir, f"{name}.{fmt}"))

def p_w_g(w, g, group_means, group_cov):
    return jax.scipy.stats.multivariate_normal.pdf(w, mean=group_means[g], cov=group_cov)

def p_y_w(w, beta):
    probs = jnp.exp(w / beta)
    return probs / jnp.sum(probs)

def init_supports(n_groups, n_foods, w_min, w_max, w_steps):
    supports = {
        "g": jnp.arange(n_groups),
        "w_i": jnp.linspace(w_min, w_max, w_steps)
    }
    supports["w"] = jnp.array([[w1, w2, w3] for w1 in supports["w_i"] for w2 in supports["w_i"] for w3 in supports["w_i"]])
    supports["x"] = jnp.array([[w1, w2, w3, g] for w1 in supports["w_i"] for w2 in supports["w_i"] for w3 in supports["w_i"] for g in supports["g"]])
    supports["y"] = jnp.arange(n_foods)
    supports["z"] = jnp.arange(len(supports["x"]))

    supports["x_w"] = jnp.array([x[:3] for x in supports["x"]])
    supports["x_g"] = jnp.array([x[3] for x in supports["x"]]).astype(int)

    return supports

def init_constant_dists(supports, group_means, group_cov, choice_beta):
    n_w, n_x, n_y, n_z = len(supports["w"]), len(supports["x"]), len(supports["y"]), len(supports["z"])

    dists = {"p_g": jnp.ones((n_groups)) / n_groups}

    dists["p_w_g"] = jnp.zeros((n_groups, n_w))
    for g in range(n_groups):
        for w_idx, w in enumerate(supports["w"]):
            dists["p_w_g"] = dists["p_w_g"].at[g, w_idx].set(p_w_g(w, g, group_means, group_cov))
    dists["p_w_g"] /= jnp.sum(dists["p_w_g"], axis=1)[:,None]

    dists["p_w"] = dists["p_w_g"].T @ dists["p_g"]
    dists["p_w"] /= jnp.sum(dists["p_w"])

    dists["p_g_w"] = (dists["p_w_g"].T * dists["p_g"]) / dists["p_w"][:, None]
    dists["p_g_w"] /= jnp.sum(dists["p_g_w"], axis=1)[:, None]

    dists["p_x_g"] = jnp.zeros((n_groups, n_x))
    for g in range(n_groups):
        for x_idx, x in enumerate(supports["x"]):
            w1, w2, w3, g_ = x
            if g == g_:
                w = jnp.array([w1, w2, w3])
                dists["p_x_g"] = dists["p_x_g"].at[g, x_idx].set(p_w_g(w, g, group_means, group_cov))
    dists["p_x_g"] /= jnp.sum(dists["p_x_g"], axis=1)[:,None]

    dists["p_x"] = jnp.zeros((n_x))
    for x_idx, x in enumerate(supports["x"]):
        w1, w2, w3, g = x
        w = jnp.array([w1, w2, w3])
        dists["p_x"] = dists["p_x"].at[x_idx].set(p_w_g(w, int(g), group_means, group_cov) * dists["p_g"][int(g)])
    dists["p_x"] /= jnp.sum(dists["p_x"])

    dists["p_z"] = jnp.ones((n_z)) / n_z

    dists["p_y_x"] = jnp.zeros((n_x, n_y))
    for x_idx, x in enumerate(supports["x"]):
        w1, w2, w3, _ = x
        w = jnp.array([w1, w2, w3])
        dists["p_y_x"] = dists["p_y_x"].at[x_idx].set(p_y_w(w, choice_beta))

    dists["p_y_g"] = jnp.zeros((n_groups, n_y))
    for g in range(n_groups):
        for x_idx, x in enumerate(supports["x"]):
            p_x_g = dists["p_x_g"][g, x_idx]
            dists["p_y_g"] = dists["p_y_g"].at[g].add(p_x_g * dists["p_y_x"][x_idx])
    dists["p_y_g"] /= jnp.sum(dists["p_y_g"], axis=1)[:,None]

    # initialise p(z|s) and p(z|x) as uniform
    dists["p_z_x"] = jnp.ones((n_x, n_z)) / n_z

    dists["p_x_z"] = jnp.zeros((n_z, n_x))
    dists["p_xz"] = jnp.zeros((n_x, n_z))

    return dists

@jax.jit
def init_p_y_z(key, supports, dists):
    # initialise p(y|z) with random values
    n_y, n_z = len(supports["y"]), len(supports["z"])
    dists["p_y_z"] = jr.uniform(key, (n_z, n_y))
    dists["p_y_z"] /= jnp.sum(dists["p_y_z"], axis=1)[:, None]
    return dists

def visualise_p_y_gs(p_y_gs):
    data = []
    for sigma, p_y_g in p_y_gs.items():
        for g, p_y in enumerate(p_y_g):
            for f, p in enumerate(p_y):
                data.append({"sigma": sigma, "group": int(g), "food": int(f), "p": float(p)})
    data = pd.DataFrame(data)

    fig, axs = plt.subplots(1, len(p_y_gs), figsize=(14, 3))
    for i, (k, p_y_g) in enumerate(p_y_gs.items()):
        # use a grouped bar chart to show p(y|g) for different groups
        sns.barplot(x="group", y="p", hue="food", data=data[data["sigma"] == k], ax=axs[i], legend=i==len(axs)-1)
        axs[i].set(title=f"sigma={k}", ylim=(0, 1))

    fig.tight_layout()
    plt.show()

# @jax.jit
def compute_mis(supports: Dict[str, jax.Array], dists: Dict[str, jax.Array]):
    ##### STEP 1: COMPUTE I(Z;X) #####
    dists["p_z"] = dists["p_z_x"].T @ dists["p_x"]
    dists["p_z"] /= jnp.sum(dists["p_z"])
    dists["p_xz"] = dists["p_x"][:,None] * dists["p_z_x"]
    dists["p_xz"] /= jnp.sum(dists["p_xz"])
    i_xz = mi(dists["p_x"], dists["p_z"], dists["p_xz"])
    i_xz /= entropy(dists["p_x"]) # normalise by the ceiling H(x)

    ##### STEP 2: COMPUTE I(Z;Y) #####
    dists["p_y"] = dists["p_y_x"].T @ dists["p_x"]
    dists["p_y"] /= jnp.sum(dists["p_y"])
    dists["p_y_z"] = jnp.einsum('xy,zx->zy', dists["p_y_x"], dists["p_x_z"])
    dists["p_y_z"] /= jnp.sum(dists["p_y_z"], axis=1)[:, None]
    dists["p_yz"] = dists["p_y_z"] * dists["p_z"][:,None]
    dists["p_yz"] /= jnp.sum(dists["p_yz"])
    i_yz = mi(dists["p_y"], dists["p_z"], dists["p_yz"])
    i_yz /= entropy(dists["p_y"]) # normalise by the ceiling H(y)

    ##### STEP 3: COMPUTE I(Z;W) #####
    # first, compute p(x|w) = p(g|w) * delta(x[:3],w)
    delta_w_x = jnp.array([
        jnp.all(supports["x_w"] == w, axis=1) for w in supports["w"]
    ])
    dists["p_x_w"] = delta_w_x * dists["p_g_w"][:, supports["x_g"]]
    dists["p_x_w"] /= jnp.sum(dists["p_x_w"], axis=1)[:, None]
    # then use it to compute p(z|w) by marginalising over x
    dists["p_z_w"] = jnp.dot(dists["p_x_w"], dists["p_z_x"])
    dists["p_z_w"] /= jnp.sum(dists["p_z_w"], axis=1)[:, None]
    # finally, compute p(w,z) and I(W;Z)
    dists["p_wz"] = dists["p_w"][:, None] * dists["p_z_w"]
    dists["p_wz"] /= jnp.sum(dists["p_wz"])
    i_wz = mi(dists["p_w"], dists["p_z"], dists["p_wz"])
    i_wz /= entropy(dists["p_w"]) # normalise by the ceiling H(w)

    ##### STEP 4: COMPUTE I(Z;G) #####
    # first, compute p(z|g) = sum_x p(z|x)p(x|g)
    dists["p_z_g"] = jnp.dot(dists["p_x_g"], dists["p_z_x"])
    dists["p_z_g"] /= jnp.sum(dists["p_z_g"], axis=1)[:, None]
    # then compute p(g,z) and I(G;Z)
    dists["p_gz"] = dists["p_g"][:, None] * dists["p_z_g"]
    dists["p_gz"] /= jnp.sum(dists["p_gz"])
    i_gz = mi(dists["p_g"], dists["p_z"], dists["p_gz"])
    i_gz /= entropy(dists["p_g"]) # normalise by the ceiling H(g)

    return i_xz, i_yz, i_wz, i_gz

def run_for_params(
    n_seeds: int,
    n_groups: int,
    group_means: jax.Array,
    group_sigma: float,
    w_min: float,
    w_max: float,
    w_steps: int,
    choice_beta: float,
    betas: jax.Array,
    max_steps=50
):
    group_cov = group_sigma * jnp.eye(n_foods)
    data, supports = [], init_supports(n_groups, n_foods, w_min, w_max, w_steps)
    initial_dists = init_constant_dists(supports, group_means, group_cov, choice_beta)

    for seed in tqdm(range(n_seeds)):
        key, dists = jr.PRNGKey(seed), {k: jnp.copy(v) for k, v in initial_dists.items()}
        dists = init_p_y_z(key, supports, dists)
        for beta in sorted(betas)[::-1]:
            dists, e_ds, steps_run = jax.jit(blahut_arimoto, static_argnums=(3,))(supports, dists, beta, max_steps=max_steps)
            i_xz, i_yz, i_wz, i_gz = compute_mis(supports, dists)
            data.append({
                "group_sigma": group_sigma,
                "seed": seed,
                "beta": float(beta),
                "choice_beta": choice_beta,
                "distortion": float(e_ds[steps_run-1]),
                "i_xz": float(i_xz),
                "i_yz": float(i_yz),
                "i_wz": float(i_wz),
                "i_gz": float(i_gz)
            })
    return pd.DataFrame(data)

n_seeds, n_groups, n_foods = 1, 3, 3
group_means = jnp.array([
    [0.9, 0.1, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.1, 0.9]
])
choice_beta = 0.2
w_min, w_max, w_steps = 0.1, 0.9, 5
betas = jnp.arange(101)
group_sigmas = [0.025, 0.1, 0.25, 1.0]

# # visualise p(y|g) for different group variances
# p_y_gs = {}
# for sigma in tqdm([0.01, 0.1, 0.5, 10.0]):
#     group_cov = sigma * jnp.eye(n_foods)
#     dists = init_dists(key, supports, group_means, group_cov, choice_beta)
#     p_y_gs[sigma] = dists["p_y_g"]

# visualise_p_y_gs(p_y_gs)

data = pd.DataFrame({
    "group_sigma": [],
    "seed": [],
    "beta": [],
    "choice_beta": [],
    "distortion": [],
    "i_xz": [],
    "i_yz": [],
    "i_wz": [],
    "i_gz": []
})
for group_sigma in group_sigmas:
    _data = run_for_params(
        n_seeds,
        n_groups,
        group_means,
        group_sigma,
        w_min,
        w_max,
        w_steps,
        choice_beta,
        betas,
        max_steps=50
    )
    data = pd.concat([data, _data], ignore_index=True)

# plot I(X;Z) as a function of beta
fig, axs = plt.subplots(1, len(group_sigmas), sharey=True)
for i, gs in enumerate(group_sigmas):
    tmp = data.loc[data["group_sigma"] == gs]
    sns.lineplot(data=tmp, x="beta", y="i_xz", ax=axs[i], **plot_kwargs)
    axs[i].set(ylim=(-0.05,1.05), xlabel=r"$\beta$", ylabel=r"$I(X;Z)$")
save_fig(fig, "ixz_vs_beta")

# plot capacity curves
fig, axs = plt.subplots(1, len(group_sigmas), sharey=True)
for i, gs in enumerate(group_sigmas):
    tmp = data.loc[data["group_sigma"] == gs]
    sns.lineplot(data=tmp, x="i_xz", y="i_yz", ax=axs[i], **plot_kwargs)
    axs[i].set(ylim=(-0.05,1.05), xlabel=r"$I(X;Z)$", ylabel=r"$I(Y;Z)$")
save_fig(fig, "capacity_curves")

# plot I(W;Z) and I(G;Z) as a function of beta
fig, axs = plt.subplots(1, len(group_sigmas), sharey=True)
for i, gs in enumerate(group_sigmas):
    tmp = data.loc[data["group_sigma"] == gs]
    legend = i == len(group_sigmas) - 1
    sns.lineplot(data=tmp, x="beta", y="i_wz", ax=axs[i], color=colors[0], label="I(W;Z)", legend=legend, **plot_kwargs)
    sns.lineplot(data=tmp, x="beta", y="i_gz", ax=axs[i], color=colors[1], label="I(G;Z)", legend=legend, **plot_kwargs)
    axs[i].set(ylim=(-0.05,1.05), xlabel=r"$\beta$", ylabel="MI")
save_fig(fig, "mis_vs_beta")

# plot I(W;Z) and I(G;Z) as a function of I(X;Z)
fig, axs = plt.subplots(1, len(group_sigmas), sharey=True)
for i, gs in enumerate(group_sigmas):
    tmp = data.loc[data["group_sigma"] == gs]
    legend = i == len(group_sigmas) - 1
    sns.lineplot(data=tmp, x="i_xz", y="i_wz", ax=axs[i], color=colors[0], label="I(W;Z)", legend=legend, **plot_kwargs)
    sns.lineplot(data=tmp, x="i_xz", y="i_gz", ax=axs[i], color=colors[1], label="I(G;Z)", legend=legend, **plot_kwargs)
    axs[i].set(ylim=(-0.05,1.05), xlabel=r"$I(X;Z)$", ylabel="MI")
save_fig(fig, "mis_vs_ixz")
