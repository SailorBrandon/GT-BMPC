import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp

@jax.jit
def is_separating_axis(n, P1, P2):
    # Project points of P1 onto the axis n
    projections1 = jnp.dot(P1, n)
    min1 = jnp.min(projections1)
    max1 = jnp.max(projections1)

    # Project points of P2 onto the axis n
    projections2 = jnp.dot(P2, n)
    min2 = jnp.min(projections2)
    max2 = jnp.max(projections2)

    # Check if there is an overlap on this axis
    return jnp.logical_not((max1 >= min2) & (max2 >= min1))

@jax.jit
def find_edges_norms(P):
    # Calculate edges and their perpendicular norms
    edges = jnp.roll(P, -1, axis=0) - P
    norms = jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)
    return edges, norms

@jax.jit
def collide_jax(P1, P2):
    """
    Check if two polygons overlap using the Separating Axis Theorem.
    Args:
        P1, P2: List of vertices of the polygons.
    """
    P1 = jnp.array(P1, dtype=jnp.float32)
    P2 = jnp.array(P2, dtype=jnp.float32)
    
    # Find the edges and normals for both polygons
    _, norms1 = find_edges_norms(P1)
    _, norms2 = find_edges_norms(P2)
    
    # Combine the norms
    norms = jnp.concatenate([norms1, norms2], axis=0)

    # Check if any separating axis exists
    separating_axis = jax.vmap(lambda n: is_separating_axis(n, P1, P2))(norms)
    
    return jnp.logical_not(jnp.any(separating_axis))
