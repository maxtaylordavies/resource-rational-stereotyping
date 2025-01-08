from itertools import product
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
plt.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.titlesize": 18,
    }
)
plot_dir = "./plots"
plot_kwargs = {"linewidth": 2.5}
colors = ["#D81C8D", "#5C1CCD"]

def save_fig(fig, name, formats=["png"]):
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(os.path.join(plot_dir, f"{name}.{fmt}"))

@jax.jit
def p_w_g(w, g, group_means, group_cov):
    return jax.scipy.stats.multivariate_normal.pdf(w, mean=group_means[g], cov=group_cov)

@jax.jit
def p_c_w(c, w, choice_beta):
    n = jnp.sum(c)
    p = p_y_w(w, choice_beta)
    return jax.scipy.stats.multinomial.pmf(c, n, p)

@jax.jit
def p_c_g(c, g, w_support, group_means, group_cov, choice_beta):
    p = 0
    for w in w_support:
        p += p_w_g(w, g, group_means, group_cov) * p_c_w(c, w, choice_beta)
    return p

@jax.jit
def p_y_w(w, beta):
    probs = jnp.exp(w / beta)
    return probs / jnp.sum(probs)

def init_supports(n_groups, n_foods, n_hist, w_min, w_max, w_steps):
    supports = {
        "g": jnp.arange(n_groups),
        "w_i": jnp.linspace(w_min, w_max, w_steps),
        "y": jnp.arange(n_foods)
    }
    supports["w"] = jnp.array([[w1, w2, w3] for w1 in supports["w_i"] for w2 in supports["w_i"] for w3 in supports["w_i"]])

    c_i = list(range(n_hist + 1))
    tmp = [x for x in product(c_i, c_i, c_i) if sum(x) == n_hist]
    supports["c"] = jnp.array(tmp)

    supports["x"] = jnp.array([
        [c[0], c[1], c[2], g] for c in supports["c"] for g in supports["g"]
    ])
    supports["x_c"] = jnp.array([x[:3] for x in supports["x"]])
    supports["x_g"] = jnp.array([x[3] for x in supports["x"]]).astype(int)

    supports["z"] = jnp.arange(len(supports["x"]))

    return supports

@jax.jit
def init_dists(key, supports, mu, sigma, rho, choice_beta):
    n_groups, n_x, n_y, n_z = len(supports["g"]), len(supports["x"]), len(supports["y"]), len(supports["z"])

    # sample group means
    cov = sigma * jnp.eye(len(mu))
    group_means = jr.multivariate_normal(key, mu, cov, (n_groups,))
    group_cov = rho * cov

    dists = {"p_g": jnp.ones((n_groups)) / n_groups}

    dists["p_w_g"] = jax.vmap(
        lambda g: jax.vmap(
            lambda w: p_w_g(w, g, group_means, group_cov),
            in_axes=(0)
        )(supports["w"]), in_axes=(0)
    )(supports["g"])
    dists["p_w_g"] /= jnp.sum(dists["p_w_g"], axis=1)[:,None]

    dists["p_w"] = dists["p_w_g"].T @ dists["p_g"]
    dists["p_w"] /= jnp.sum(dists["p_w"])

    dists["p_c_w"] = jax.vmap(
        lambda w: jax.vmap(
            lambda c: p_c_w(c, w, choice_beta),
            in_axes=(0)
        )(supports["c"]), in_axes=(0)
    )(supports["w"])
    dists["p_c_w"] /= jnp.sum(dists["p_c_w"], axis=1)[:,None]

    dists["p_c_g"] = jnp.einsum('wc,gw->gc', dists["p_c_w"], dists["p_w_g"])
    dists["p_c_g"] /= jnp.sum(dists["p_c_g"], axis=1)[:,None]

    dists["p_c"] = dists["p_c_g"].T @ dists["p_g"]
    dists["p_c"] /= jnp.sum(dists["p_c"])

    dists["p_g_c"] = (dists["p_c_g"].T * dists["p_g"]) / dists["p_c"][:, None]
    dists["p_g_c"] /= jnp.sum(dists["p_g_c"], axis=1)[:, None]

    dists["p_w_c"] = (dists["p_c_w"].T * dists["p_w"]) / dists["p_c"][:, None]
    dists["p_w_c"] /= jnp.sum(dists["p_w_c"], axis=1)[:, None]

    dists["p_x_g"] = jnp.repeat(dists["p_c_g"], n_groups, axis=1)
    dists["p_x_g"] /= jnp.sum(dists["p_x_g"], axis=1)[:,None]

    dists["p_x"] = dists["p_x_g"].T @ dists["p_g"]
    dists["p_x"] /= jnp.sum(dists["p_x"])

    dists["p_z"] = jnp.ones((n_z)) / n_z

    dists["p_y_w"] = jax.vmap(
        lambda w: p_y_w(w, choice_beta),
        in_axes=(0)
    )(supports["w"])

    dists["p_y_c"] = jnp.einsum('wy,cw->cy', dists["p_y_w"], dists["p_w_c"])
    dists["p_y_c"] /= jnp.sum(dists["p_y_c"], axis=1)[:, None]

    dists["p_y_x"] = jnp.repeat(dists["p_y_c"], n_groups, axis=0)

    dists["p_y_g"] = jnp.einsum('xy,gx->gy', dists["p_y_x"], dists["p_x_g"])
    dists["p_y_g"] /= jnp.sum(dists["p_y_g"], axis=1)[:,None]

    # initialise p(z|s) and p(z|x) as uniform
    dists["p_z_x"] = jnp.ones((n_x, n_z)) / n_z

    # initialise p(y|z) with random values
    dists["p_y_z"] = jr.uniform(key, (n_z, n_y))
    dists["p_y_z"] /= jnp.sum(dists["p_y_z"], axis=1)[:, None]

    dists["p_x_z"] = jnp.zeros((n_z, n_x))
    dists["p_xz"] = jnp.zeros((n_x, n_z))

    return dists

@jax.jit
def compute_mis(supports: Dict[str, jax.Array], dists: Dict[str, jax.Array]):
    ##### STEP 1: COMPUTE I(Z;X) #####
    dists["p_xz"] = dists["p_x"][:,None] * dists["p_z_x"]
    dists["p_xz"] /= jnp.sum(dists["p_xz"])
    i_xz = mi(dists, "x", "z")

    ##### STEP 2: COMPUTE I(Z;Y) #####
    dists["p_y"] = dists["p_y_x"].T @ dists["p_x"]
    dists["p_y"] /= jnp.sum(dists["p_y"])
    dists["p_y_z"] = jnp.einsum('xy,zx->zy', dists["p_y_x"], dists["p_x_z"])
    dists["p_y_z"] /= jnp.sum(dists["p_y_z"], axis=1)[:, None]
    dists["p_yz"] = dists["p_y_z"] * dists["p_z"][:,None]
    dists["p_yz"] /= jnp.sum(dists["p_yz"])
    i_yz = mi(dists, "y", "z")

    ##### STEP 3: COMPUTE I(Z;C) #####
    # first, compute p(x|c) = p(g|c) * delta(x[:3],c)
    delta_c_x = jnp.array([
        jnp.all(supports["x_c"] == c, axis=1) for c in supports["c"]
    ])
    dists["p_x_c"] = delta_c_x * dists["p_g_c"][:, supports["x_g"]]
    dists["p_x_c"] /= jnp.sum(dists["p_x_c"], axis=1)[:, None]
    # then use it to compute p(z|w) by marginalising over x
    dists["p_z_c"] = jnp.einsum('xz,cx->cz', dists["p_z_x"], dists["p_x_c"])
    dists["p_z_c"] /= jnp.sum(dists["p_z_c"], axis=1)[:, None]
    # finally, compute p(w,z) and I(W;Z)
    dists["p_cz"] = dists["p_c"][:, None] * dists["p_z_c"]
    dists["p_cz"] /= jnp.sum(dists["p_cz"])
    i_cz = mi(dists, "c", "z")

    ##### STEP 4: COMPUTE I(Z;G) #####
    # first, compute p(z|g) = sum_x p(z|x)p(x|g)
    dists["p_z_g"] = jnp.einsum('xz,gx->gz', dists["p_z_x"], dists["p_x_g"])
    dists["p_z_g"] /= jnp.sum(dists["p_z_g"], axis=1)[:, None]
    # then compute p(g,z) and I(G;Z)
    dists["p_gz"] = dists["p_g"][:, None] * dists["p_z_g"]
    dists["p_gz"] /= jnp.sum(dists["p_gz"])
    i_gz = mi(dists, "g", "z")

    ##### STEP 5: NORMALISE ALL MI VALUES #####
    # normalise I(X;Z) by the ceiling H(X)
    i_xz /= entropy(dists["p_x"])
    # normalise I(Y;Z) by the ceiling I(Y;X)
    dists["p_yx"] = dists["p_y_x"] * dists["p_x"][:,None]
    dists["p_yx"] /= jnp.sum(dists["p_yx"])
    i_yz /= mi(dists, "y", "x")
    # normalise I(C;Z) by the ceiling H(C)
    i_cz /= entropy(dists["p_c"])
    # normalise I(G;Z) by the ceiling H(G)
    i_gz /= entropy(dists["p_g"])

    return i_xz, i_yz, i_cz, i_gz

def run_for_params(
    n_seeds: int,
    n_groups: int,
    n_hist: int,
    mu: jax.Array,
    sigma: float,
    rho: float,
    w_min: float,
    w_max: float,
    w_steps: int,
    choice_beta: float,
    betas: jax.Array,
    max_steps=50
):
    data, supports = [], init_supports(n_groups, n_foods, n_hist, w_min, w_max, w_steps)

    for seed in range(n_seeds):
        key = jr.PRNGKey(seed)
        dists = init_dists(key, supports, mu, sigma, rho, choice_beta)
        for beta in sorted(betas)[::-1]:
            dists, e_ds, steps_run = jax.jit(blahut_arimoto, static_argnums=(3,))(supports, dists, beta, max_steps=max_steps)
            i_xz, i_yz, i_cz, i_gz = compute_mis(supports, dists)
            data.append({
                "rho": rho,
                "seed": seed,
                "beta": float(beta),
                "choice_beta": choice_beta,
                "distortion": float(e_ds[steps_run-1]),
                "i_xz": float(i_xz),
                "i_yz": float(i_yz),
                "i_cz": float(i_cz),
                "i_gz": float(i_gz)
            })
    return pd.DataFrame(data)

n_seeds, n_groups, n_foods, n_hist = 50, 3, 3, 15
choice_beta = 0.25
w_min, w_max, w_steps = -4.0, 4.0, 41
betas = jnp.array([2**x for x in jnp.arange(-1, 9.5, 0.5)])
mu, sigma = jnp.zeros(n_foods), 1.0
rhos = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
rhos_subset = [0.01, 0.1, 1.0, 10.0]

print("Running with:")
print(f"\tn_seeds = {n_seeds}")
print(f"\tn_groups = {n_groups}")
print(f"\tn_foods = {n_foods}")
print(f"\tn_hist = {n_hist}")
print(f"\tchoice_beta = {choice_beta}")
print(f"\tw_min = {w_min}")
print(f"\tw_max = {w_max}")
print(f"\tw_steps = {w_steps}")
print(f"\tbetas = {betas}")
print(f"\tmu = {mu}")
print(f"\tsigma = {sigma}")
print(f"\trhos = {rhos}")


# # visualise p(y|g) for different group variances
# key = jr.PRNGKey(78)
# supports = init_supports(n_groups, n_foods, n_hist, w_min, w_max, w_steps)
# sigma, rhos, data = 1.0, [0.1, 1.0, 10.0, 100.0], []

# for rho in tqdm(rhos):
#     dists, group_means = init_dists(key, supports, jnp.zeros(n_foods), sigma, rho, choice_beta)
#     for g in range(n_groups):
#         for y in range(n_foods):
#             p = float(dists["p_y_g"][g,y])
#             data.append({ "rho": rho, "g": g, "y": y, "p": p })
# data = pd.DataFrame(data)

# fruit_colors = ["#FF5050", "#FF9E18", "#0BAD67"]
# fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8))
# for i, rho in enumerate(rhos):
#     ax = axs[i // 2, i % 2]
#     tmp = data.loc[data["rho"] == rho]
#     sns.barplot(x="g", y="p", hue="y", data=tmp, ax=ax, palette=fruit_colors, legend=False)
#     title = r"$\rho=" + str(rho) + r"$"
#     xticks = [g for g in range(n_groups)]
#     xticklabels = [r"$g=" + str(g) + r"$" for g in range(n_groups)]
#     ax.set(title=title, xlabel="", ylabel=r"$p(y|g)$", xticks=xticks, xticklabels=xticklabels, ylim=(0, 1))
# fig.suptitle(r"Group-conditioned fruit choice distributions for varying $\rho$", fontsize=23)
# save_fig(fig, "p_y_g", formats=["svg", "png"])

data = pd.DataFrame({
    "rho": [],
    "seed": [],
    "beta": [],
    "choice_beta": [],
    "distortion": [],
    "i_xz": [],
    "i_yz": [],
    "i_cz": [],
    "i_gz": []
})
for rho in tqdm(rhos):
    _data = run_for_params(
        n_seeds,
        n_groups,
        n_hist,
        mu,
        sigma,
        rho,
        w_min,
        w_max,
        w_steps,
        choice_beta,
        betas,
        max_steps=50
    )
    data = pd.concat([data, _data], ignore_index=True)
data["stereotyping"] = data["i_gz"] - data["i_cz"]

# plot expected distortion as a function of beta for example rho = 1.0
single_rho_data = data.loc[data["rho"] == 1.0]
prelim_plot_kwargs = {"linewidth": 4.0, "color": "#EE1195", "marker": "o", "markersize": 15}
fig, ax = plt.subplots()
sns.lineplot(data=single_rho_data, x="beta", y="distortion", ax=ax, **prelim_plot_kwargs)
ax.set(xlabel=r"$\beta$", ylabel=r"$\langle d(x,z) \rangle$", title=r"Example distortion curve for $\rho=1.0$")
ax.set_xscale("log", base=2)
save_fig(fig, "distortion_vs_beta", formats=["png", "svg"])

# plot information curve for example rho = 1.0
fig, ax = plt.subplots()
sns.lineplot(data=single_rho_data, x="i_xz", y="i_yz", ax=ax, **prelim_plot_kwargs)
ax.set(ylim=(-0.05, 1.05), xlabel=r"$I(X;Z)/H(X)$", ylabel=r"$I(Y;Z)/I(Y;X)$", title=r"Example normalised information curve for $\rho=1.0$")
save_fig(fig, "information_curve", formats=["png", "svg"])

# plot I(C;Z) and I(G;Z) as a function of beta
fig, axs = plt.subplots(1, len(rhos_subset), figsize=(20, 4.5), sharey=True)
for i, rho in enumerate(rhos_subset):
    tmp = data.loc[data["rho"] == rho]
    legend = i == len(rhos) - 1
    sns.lineplot(data=tmp, x="beta", y="i_cz", ax=axs[i], color=colors[0], label="I(C;Z)/H(C)", legend=legend, **plot_kwargs)
    sns.lineplot(data=tmp, x="beta", y="i_gz", ax=axs[i], color=colors[1], label="I(G;Z)/H(G)", legend=legend, **plot_kwargs)
    axs[i].set(ylim=(-0.05,1.05), xlabel=r"$\beta$", ylabel="Information extracted", title=r"$\rho=" + str(rho) + r"$")
    axs[i].set_xscale("log", base=2)
fig.suptitle(r"Information extracted about $G$ and $C$ as a function of $\beta$", fontsize=20)
save_fig(fig, "mis_vs_beta", formats=["png", "svg"])

# plot I(C;Z) and I(G;Z) as a function of I(X;Z)
fig, axs = plt.subplots(1, len(rhos_subset), figsize=(20, 4.5), sharey=True)
for i, rho in enumerate(rhos_subset):
    tmp = data.loc[data["rho"] == rho]
    legend = i == len(rhos) - 1
    sns.scatterplot(data=tmp, x="i_xz", y="i_cz", ax=axs[i], color=colors[0], alpha=0.5, label="I(C;Z)/H(C)", legend=legend)
    sns.scatterplot(data=tmp, x="i_xz", y="i_gz", ax=axs[i], color=colors[1], alpha=0.5, label="I(G;Z)/H(G)", legend=legend)
    axs[i].set(ylim=(-0.05,1.05), xlabel=r"$I(X;Z)$", ylabel="Information extracted", title=r"$\rho=" + str(rho) + r"$")
fig.suptitle(r"Information extracted about $G$ and $C$ as a function of $I(X;Z)$", fontsize=20)
save_fig(fig, "mis_vs_ixz", formats=["png", "svg"])

# plot a heatmap of I(G;Z) - I(C;Z) over group sigma and beta
heatmap_data: pd.DataFrame = data[["rho", "beta", "stereotyping"]]
heatmap_data = heatmap_data.loc[heatmap_data["beta"] >= 1.0]
heatmap_data = heatmap_data.groupby(["rho", "beta"]).mean().reset_index()
heatmap_data = heatmap_data.pivot(index="rho", columns="beta", values="stereotyping")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(heatmap_data, ax=ax, cmap="viridis")
ax.set(xlabel=r"$\beta$", ylabel=r"$\rho$", title=r"Amount of stereotyping in optimal encoding $p^*$")
ax.set_xticklabels([r"$2^{" + str(int(np.log2(x))) + r"}$" for x in heatmap_data.columns], rotation=45)
[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
save_fig(fig, "heatmap", formats=["png", "svg"])
