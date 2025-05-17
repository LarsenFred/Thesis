import jax.numpy as jnp

def prob_survival(period, options, survival):
    alpha1 = options["alpha1"]
    alpha2 = options["alpha2"]

    # death probability must be jnp.exp, and clipped to [0,1]
    death_prob = alpha1 * (jnp.exp(alpha2 * period) - 1.0)
    death_prob = jnp.clip(death_prob, 0.0, 1.0)     # just in case it ever goes outside

    # alive: P(dead next) = death_prob,  P(alive next) = 1 - death_prob
    probs_alive = jnp.stack([death_prob, 1.0 - death_prob], axis=-1)   # shape (n_agents, 2)

    # dead:   P(dead next) = 1,           P(alive next) = 0
    probs_dead = jnp.broadcast_to(jnp.array([1.0, 0.0]), probs_alive.shape)

    # select rowwise based on current survival
    # survival is shape (n_agents,), so survival[...,None] is (n_agents,1)
    return jnp.where(survival[..., None] == 1, probs_alive, probs_dead)


