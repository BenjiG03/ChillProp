import jax
import jax.numpy as jnp
import CoolProp.CoolProp as CP
import chillprop.highlevel as CH
import time
import numpy as np

jax.config.update("jax_enable_x64", True)

def benchmark():
    fluid = "Nitrogen"
    T_vals = np.linspace(100, 500, 100)
    P_vals = np.linspace(1e5, 1e7, 100)
    
    # Pre-load/JIT
    print("Pre-warming JAX JIT...")
    CH.PropsSI('D', 'T', 300, 'P', 1e6, fluid)
    
    # 1. CoolProp Serial
    start = time.time()
    for T in T_vals:
        for P in P_vals:
            CP.PropsSI('D', 'T', T, 'P', P, fluid)
    cp_time = time.time() - start
    print(f"CoolProp Serial: {cp_time:.4f}s")
    
    # 2. ChillProp Serial (Python loop)
    # This will still be fast because internal functions are JITted
    start = time.time()
    for T in T_vals:
        for P in P_vals:
            CH.PropsSI('D', 'T', T, 'P', P, fluid)
    ch_serial_time = time.time() - start
    print(f"ChillProp Serial: {ch_serial_time:.4f}s")
    
    # 3. ChillProp Vectorized (JIT)
    params = CH.get_params(fluid)
    from chillprop.solver import solve_rho_PT
    
    @jax.jit
    def batch_props(T, P):
        return jax.vmap(solve_rho_PT, in_axes=(None, 0, 0))(params, P, T)
    
    T_grid, P_grid = np.meshgrid(T_vals, P_vals)
    T_flat = jnp.array(T_grid.flatten())
    P_flat = jnp.array(P_grid.flatten())
    
    # Warmup
    batch_props(T_flat[:10], P_flat[:10]).block_until_ready()
    
    start = time.time()
    batch_props(T_flat, P_flat).block_until_ready()
    ch_vec_time = time.time() - start
    print(f"ChillProp Vectorized: {ch_vec_time:.4f}s")
    
    print(f"\nSpeedup (Vectorized vs CP): {cp_time/ch_vec_time:.1f}x")

if __name__ == "__main__":
    benchmark()
