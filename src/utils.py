import jax
import jax.numpy as jnp
from jax.scipy.special import rel_entr

@jax.jit
def entropy(p: jax.Array) -> jax.Array:
    """
    Compute the entropy of a probability distribution.

    Args:
        p: The probability distribution.

    Returns:
        The entropy of the distribution.
    """
    return -jnp.sum(p * jnp.log2(p) * (p > 0))

@jax.jit
def kl(p: jax.Array, q: jax.Array) -> jax.Array:
    """
    Compute the Kullback-Leibler divergence between two probability distributions.

    Args:
        p: The first probability distribution.
        q: The second probability distribution.

    Returns:
        The Kullback-Leibler divergence between the two distributions.
    """
    return jnp.sum(rel_entr(p, q))

# @jax.jit
def mi(p: jax.Array, q: jax.Array, pq: jax.Array) -> jax.Array:
    """
    Compute the mutual information between two probability distributions.

    Args:
        p: The first probability distribution.
        q: The second probability distribution.

    Returns:
        The mutual information between the two distributions.
    """
    return entropy(p) + entropy(q) - entropy(pq)
