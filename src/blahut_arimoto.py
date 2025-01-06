from typing import Dict

import jax
import jax.numpy as jnp

from src.utils import kl

@jax.jit
def distortion(dists: Dict[str, jax.Array], x, z):
    return kl(dists["p_y_x"][x], dists["p_y_z"][z])

@jax.jit
def do_blahut_arimoto_iteration(supports: Dict[str, jax.Array], dists: Dict[str, jax.Array], beta: float):
    """
    Perform one iteration of the Blahut-Arimoto algorithm.

    Args:
        supports: The support sets for the variables in the model.
        dists: A dictionary of probability distributions.
        beta: The lagrange multiplier for the distortion term.

    Returns:
        dists: A dictionary of updated probability distributions.
        e_d: The expected distortion under the current distribution p(x,z).
    """
    n_x, n_z = len(supports["x"]), len(supports["z"])

    ##### STEP 1: Update p(z|x) #####
    # first use vmap to compute the distortions for all s,x,z
    d_func = lambda x, z: distortion(dists, x, z)
    distortions = jax.vmap(
        jax.vmap(d_func, in_axes=(None, 0)),
        in_axes=(0, None)
    )(jnp.arange(n_x), jnp.arange(n_z))

    # compute the expected distortion under p(x,z)
    dists["p_xz"] = dists["p_x"] * dists["p_z_x"]
    dists["p_xz"] /= dists["p_xz"].sum()
    e_d = jnp.mean(dists["p_xz"] * distortions)

    # update p(z|x) = p(z) * exp(-beta * distortion)
    dists["p_z_x"] = dists["p_z"] * jnp.exp(-beta * distortions)
    dists["p_z_x"] /= jnp.sum(dists["p_z_x"], axis=1)[:,None]

    ##### STEP 2: Update p(z) #####
    dists["p_z"] = dists["p_z_x"].T @ dists["p_x"]
    dists["p_z"] /= jnp.sum(dists["p_z"])

    ##### STEP 3: Update p(y|z) by marginalising over x #####
    # first compute p(x|z) via Bayes' rule
    dists["p_x_z"] = ((dists["p_z_x"] * dists["p_x"][:, None]) / dists["p_z"][None, :]).T
    dists["p_x_z"] = jnp.nan_to_num(dists["p_x_z"], nan=1e-9)
    dists["p_x_z"] /= jnp.sum(dists["p_x_z"], axis=1)[:, None]

    # then compute p(y|z) = sum_x p(y|x)p(x|z)
    dists["p_y_z"] = jnp.einsum('xy,zx->zy', dists["p_y_x"], dists["p_x_z"])
    dists["p_y_z"] /= jnp.sum(dists["p_y_z"], axis=1)[:, None]

    # return all the modified distributions
    return dists, e_d

def blahut_arimoto(supports: Dict[str, jax.Array], dists: Dict[str, jax.Array], beta: float, max_steps: int = 100, tol: float = 1e-9):
    """
    Implements the Blahut-Arimoto algorithm.

    Args:
        supports: The support sets for the variables in the model.
        dists: A dictionary of probability distributions.
        beta: The lagrange multiplier for the distortion term.
        n_steps: The number of iterations to run the algorithm for.

    Returns:
        dists: A dictionary of updated probability distributions.
        e_ds: The expected distortion over time.
    """
    e_ds = jnp.zeros(max_steps)

    # use jax.lax.while_loop to run the algorithm up to a maximum number of steps
    # and until the expected distortion converges
    def cond_fun(args):
        dists, e_d, prev_e_d, e_ds, step = args
        return jnp.logical_and(step < max_steps, jnp.abs(e_d - prev_e_d) > tol)

    def body_fun(args):
        dists, e_d, prev_e_d, e_ds, step = args
        dists, new_e_d = do_blahut_arimoto_iteration(supports, dists, beta)
        e_ds = e_ds.at[step].set(new_e_d)
        return dists, new_e_d, e_d, e_ds, step + 1

    dists, _, _, e_ds, step = jax.lax.while_loop(cond_fun, body_fun, (dists, 0.0, 100.0, e_ds, 0))
    return dists, e_ds, step
