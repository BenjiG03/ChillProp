import jax
import jax.numpy as jnp
import CoolProp.CoolProp as CP
import chillprop.highlevel as CH
import numpy as np

jax.config.update("jax_enable_x64", True)

def test_anchor():
    fluid = "Nitrogen"
    params = CH.get_params(fluid)
    
    # hs_anchor from JSON: T=138.8112, rho=10065.5113
    T = 138.8112
    rho = 10065.5113
    
    # CoolProp
    as_cp = CP.AbstractState("HEOS", fluid)
    as_cp.update(CP.DmolarT_INPUTS, rho, T)
    h_cp = as_cp.hmolar()
    s_cp = as_cp.smolar()
    
    # ChillProp
    from chillprop.core import props
    res = props(params, jnp.array(rho), jnp.array(T))
    h_ch = float(res['h'])
    s_ch = float(res['s'])
    
    print(f"Anchor Point T={T}, rho={rho}")
    print(f"H: CP={h_cp:.6f}, CH={h_ch:.6f}, diff={h_ch-h_cp:.6f}")
    print(f"S: CP={s_cp:.6f}, CH={s_ch:.6f}, diff={s_ch-s_cp:.6f}")

if __name__ == "__main__":
    test_anchor()
