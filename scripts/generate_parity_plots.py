import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import jax
# Enable x64 precision for accurate comparison
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import CoolProp.CoolProp as CP

# Ensure chillprop is in path
sys.path.append(os.getcwd())
import chillprop.highlevel as CH

def generate_plots():
    fluid = "Air"
    print(f"--- Generating Parity Plots for {fluid} ---")
    
    # Create output directory
    os.makedirs("validation_plots", exist_ok=True)
    
    # Define range
    # Original: 70 to 1000 with 50 steps (~18.6 K/step)
    # New: 70 to 4000 with similar step size => ~211 steps
    T_range = np.linspace(70, 4000, 212)
    P_values = [1e5, 1e6, 5e6, 2e7] # Isobars
    
    properties = [
        ('D', 'Density', 'kg/m^3'),
        ('H', 'Enthalpy', 'J/kg'),
        ('S', 'Entropy', 'J/kg/K'),
        ('V', 'Viscosity', 'Pa-s'),
        ('L', 'Conductivity', 'W/m/K')
    ]
    
    for prop_key, prop_name, unit in properties:
        print(f"Processing {prop_name}...")
        plt.figure(figsize=(10, 6))
        
        for P in P_values:
            chill_vals = []
            cp_vals = []
            valid_T = []
            
            for T in T_range:
                try:
                    # CoolProp
                    try:
                        val_cp = CP.PropsSI(prop_key, 'T', T, 'P', P, fluid)
                    except:
                        continue # Skip invalid points for CP
                        
                    # ChillProp
                    try:
                        val_chill = float(CH.PropsSI(prop_key, 'T', T, 'P', P, fluid))
                    except:
                        continue
                        
                    cp_vals.append(val_cp)
                    chill_vals.append(val_chill)
                    valid_T.append(T)
                except Exception as e:
                    continue
            
            if not valid_T:
                continue
                
            # Plot relative error
            cp_arr = np.array(cp_vals)
            chill_arr = np.array(chill_vals)
            valid_T_arr = np.array(valid_T)
            
            # Avoid division by zero
            mask = cp_arr != 0
            rel_error = np.zeros_like(cp_arr)
            rel_error[mask] = np.abs((chill_arr[mask] - cp_arr[mask]) / cp_arr[mask])
            
            plt.plot(valid_T, rel_error, 'o-', label=f'P={P/1e6:.1f} MPa')
            
            # Statistics
            # Split into Low T (< 150K) and High T (>= 150K)
            low_t_mask = valid_T_arr < 150.0
            high_t_mask = valid_T_arr >= 150.0
            
            def get_stats(err_arr):
                if len(err_arr) == 0: return 0.0, 0.0
                return np.max(err_arr), np.sqrt(np.mean(err_arr**2))

            max_low, rmse_low = get_stats(rel_error[low_t_mask])
            max_high, rmse_high = get_stats(rel_error[high_t_mask])
            
            print(f"  P={P/1e6:5.1f} MPa | Low T (<150K): Max={max_low:.2e}, RMSE={rmse_low:.2e} | High T: Max={max_high:.2e}, RMSE={rmse_high:.2e}")

        plt.yscale('log')
        plt.xlabel('Temperature (K)')
        plt.ylabel(f'Relative Error in {prop_name}')
        plt.title(f'{prop_name} Parity: ChillProp vs CoolProp ({fluid})')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        
        filename = f"validation_plots/{prop_name}_parity.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

if __name__ == "__main__":
    generate_plots()
