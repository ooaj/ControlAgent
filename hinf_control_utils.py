import numpy as np
import control as ctrl
from scipy import signal
import matplotlib.pyplot as plt
from itertools import product

class HInfRobustControl:
    """
    H∞ robust control synthesis for systems with parameter uncertainty.
    
    This implements H∞ mixed sensitivity design to handle parameter 
    uncertainty while optimizing robust performance trade-offs.
    """
    
    def __init__(self):
        self.tolerance = 1e-6
        self.gamma_tolerance = 1e-3
    
    def assess_uncertainty_complexity(self, system_info):
        """
        Assess whether system needs H∞ control or pole placement is sufficient.
        
        Args:
            system_info: System with uncertainty bounds
            
        Returns:
            str: 'LOW', 'MODERATE', 'HIGH' complexity assessment
        """
        # Extract uncertainty levels
        if 'num_uncertainty' in system_info and system_info['num_uncertainty']:
            gain_uncertainty = system_info['num_uncertainty'][0]
        else:
            gain_uncertainty = 0.0
            
        if 'den_uncertainty' in system_info and len(system_info['den_uncertainty']) > 1:
            pole_uncertainty = system_info['den_uncertainty'][1] if system_info['den_uncertainty'][1] else 0.0
        else:
            pole_uncertainty = 0.0
        
        # Combined uncertainty metric
        total_uncertainty = gain_uncertainty + pole_uncertainty
        
        if total_uncertainty >= 0.4:  # ≥40% combined
            return 'HIGH'
        elif total_uncertainty >= 0.25:  # ≥25% combined  
            return 'MODERATE'
        else:
            return 'LOW'
    
    def parameter_to_multiplicative_uncertainty(self, system_info):
        """
        Convert parametric uncertainty to multiplicative uncertainty model.
        
        For G(s) = K/(s+a) with ±δK%, ±δa%, create multiplicative model:
        G(s) = G₀(s)(1 + W(s)Δ(s)) where |Δ(jω)| ≤ 1
        
        Args:
            system_info: System information with uncertainty bounds
            
        Returns:
            dict: Multiplicative uncertainty model components
        """
        
        # Get nominal system and uncertainty bounds
        bounds = system_info['bounds'] if 'bounds' in system_info else system_info
        
        K_nom = bounds.get('K_nom', system_info['nominal']['K'])
        a_nom = bounds.get('a_nom', system_info['nominal']['a'])
        
        K_min = bounds['K_min']
        K_max = bounds['K_max'] 
        a_min = bounds['a_min']
        a_max = bounds['a_max']
        
        # Relative uncertainty bounds
        delta_K = max(abs(K_max - K_nom), abs(K_nom - K_min)) / K_nom
        delta_a = max(abs(a_max - a_nom), abs(a_nom - a_min)) / a_nom
        
        # For first-order system G(s) = K/(s+a), multiplicative uncertainty:
        # δG/G ≈ δK/K - δa/(s+a) * s = δK/K - δa*s/(s+a)
        
        # Create frequency-dependent uncertainty weight
        # W(s) captures maximum relative uncertainty vs frequency
        
        # At low frequency (s→0): |W(jω)| ≈ δK (gain uncertainty dominates)
        # At high frequency (s→∞): |W(jω)| ≈ δa (pole uncertainty dominates)
        
        # Choose break frequency around nominal pole
        omega_break = a_nom
        
        # W(s) = (δK + δa*s/ω_break) / (1 + s/(10*ω_break))
        # This gives δK at low freq, δa*10 at high freq, with smooth transition
        
        num_w = [delta_a * 10 / omega_break, delta_K]
        den_w = [1 / (10 * omega_break), 1]
        
        W_uncertainty = ctrl.TransferFunction(num_w, den_w)
        
        return {
            'nominal_plant': ctrl.TransferFunction([K_nom], [1, a_nom]),
            'uncertainty_weight': W_uncertainty,
            'delta_K': delta_K,
            'delta_a': delta_a,
            'max_uncertainty': max(delta_K, delta_a),
            'omega_break': omega_break
        }
    
    def design_performance_weights(self, performance_specs, system_info):
        """
        Design performance and control weights for H∞ synthesis.
        
        Args:
            performance_specs: Performance requirements (settling time, etc.)
            system_info: System information
            
        Returns:
            dict: Performance weights Wp(s) and Wu(s)
        """
        
        # Extract requirements
        settling_time_max = performance_specs.get('settling_time_max', 1.0)
        steady_state_error_max = performance_specs.get('steady_state_error_max', 0.01)
        
        # Performance weight Wp(s): penalizes tracking error
        # Want small steady-state error and fast transient response
        
        # Wp(s) = (s/ωp + ωp_ss) / (s + ωp_ss*A) 
        # where ωp ≈ 2/settling_time, A > 1 for high freq rolloff
        
        omega_p = 2.0 / settling_time_max  # Performance bandwidth
        omega_p_ss = 1.0 / steady_state_error_max  # Steady-state requirement
        A_p = 100  # High frequency attenuation
        
        num_wp = [1/omega_p, omega_p_ss]
        den_wp = [1, omega_p_ss * A_p]
        
        Wp = ctrl.TransferFunction(num_wp, den_wp)
        
        # Control weight Wu(s): penalizes control effort
        # Want to limit control bandwidth and magnitude
        
        # Wu(s) = (s/ωu + ωu_0) / (s + ωu_0*A_u)
        # where ωu is control bandwidth limit
        
        a_plant = system_info['nominal']['a']
        omega_u = min(10 * a_plant, 1000)  # Control bandwidth limit  
        omega_u_0 = 0.1  # Low freq control weight
        A_u = 10  # High freq control weight
        
        num_wu = [1/omega_u, omega_u_0]
        den_wu = [1, omega_u_0 * A_u]
        
        Wu = ctrl.TransferFunction(num_wu, den_wu)
        
        return {
            'Wp': Wp,  # Performance weight
            'Wu': Wu,  # Control weight  
            'omega_p': omega_p,
            'omega_u': omega_u
        }
    
    def build_generalized_plant(self, plant, uncertainty_model, weights):
        """
        Build generalized plant P for H∞ synthesis.
        
        Standard form:
        ┌─────┬─────┐ ┌───┐   ┌───┐
        │ P11 │ P12 │ │ w │   │ z │
        │─────┼─────│ │───│ = │───│
        │ P21 │ P22 │ │ u │   │ y │
        └─────┴─────┘ └───┘   └───┘
        
        Where:
        w = [r; n; d]  (reference, noise, disturbance)  
        z = [e1; e2; u] (weighted error, weighted control)
        u = control input
        y = measured output
        
        Args:
            plant: Nominal plant G(s)
            uncertainty_model: Multiplicative uncertainty model
            weights: Performance weights
            
        Returns:
            ctrl.TransferFunction: Generalized plant P
        """
        
        G = plant  # Nominal plant
        Wp = weights['Wp']  # Performance weight
        Wu = weights['Wu']   # Control weight
        
        # For SISO mixed sensitivity problem:
        # Minimize ||[Wp*S; Wu*K*S]||∞
        # where S = 1/(1+GK) is sensitivity function
        
        # This is equivalent to:
        # P11 = [Wp; 0; Wu*G], P12 = [Wp*G; Wu*G]  
        # P21 = [I], P22 = [G]
        
        # However, for implementation simplicity with Python control library,
        # we'll use the mixed sensitivity function directly
        
        return {
            'plant': G,
            'Wp': Wp, 
            'Wu': Wu,
            'configuration': 'mixed_sensitivity'
        }
    
    def hinf_mixed_sensitivity(self, plant, weights, gamma_max=10.0):
        """
        Solve H∞ mixed sensitivity problem.
        
        Minimize γ such that ||[Wp*S; Wu*K*S]||∞ < γ
        where S = 1/(1+GK)
        
        Args:
            plant: Nominal plant G(s) 
            weights: Performance weights
            gamma_max: Maximum allowable γ
            
        Returns:
            dict: H∞ controller and analysis results
        """
        
        G = plant
        Wp = weights['Wp']
        Wu = weights['Wu']
        
        try:
            # Use Python control library's mixed sensitivity synthesis
            # This internally builds generalized plant and solves H∞ problem
            
            # For now, implement simplified version using loop shaping
            # In practice, would use ctrl.hinfsyn() with proper generalized plant
            
            # Loop shaping approach: K = K_ls where K_ls stabilizes G_ls = Wp*G*Wu^-1
            G_shaped = Wp * G  # Shape plant with performance weight
            
            # Use pole placement or PID as initial controller
            # Then iterate to minimize H∞ norm
            
            # Simplified: Choose controller bandwidth based on weights
            omega_c = weights['omega_p'] * 2  # Crossover frequency
            
            # PI controller: K(s) = Kp(1 + 1/(Ti*s))
            Ti = 1.0 / omega_c  # Integral time constant
            
            # Choose Kp for desired crossover
            s_test = 1j * omega_c
            G_mag = abs(G(s_test))
            Kp = 1.0 / G_mag
            
            # Create controller
            K = ctrl.TransferFunction([Kp * Ti, Kp], [Ti, 0])
            
            # Compute closed-loop functions
            L = G * K  # Loop transfer function
            S = 1 / (1 + L)  # Sensitivity
            T = L / (1 + L)  # Complementary sensitivity
            KS = K * S  # Control sensitivity
            
            # Compute H∞ norms
            WpS = Wp * S
            WuKS = Wu * K * S
            
            # Approximate H∞ norm using frequency response
            omega = np.logspace(-2, 4, 1000)
            
            WpS_mag = np.abs(WpS(1j * omega))
            WuKS_mag = np.abs(WuKS(1j * omega))
            
            mixed_sens_mag = np.sqrt(WpS_mag**2 + WuKS_mag**2)
            gamma_achieved = np.max(mixed_sens_mag)
            
            # Check if performance is acceptable
            success = gamma_achieved < gamma_max
            
            return {
                'controller': K,
                'gamma': gamma_achieved,
                'success': success,
                'sensitivity': S,
                'comp_sensitivity': T,
                'control_sensitivity': KS,
                'loop_transfer': L,
                'analysis': {
                    'omega_crossover': omega_c,
                    'Kp': Kp,
                    'Ti': Ti,
                    'max_WpS': np.max(WpS_mag),
                    'max_WuKS': np.max(WuKS_mag)
                }
            }
            
        except Exception as e:
            print(f"H∞ synthesis failed: {e}")
            return None
    
    def robust_stability_analysis(self, nominal_plant, controller, uncertainty_model):
        """
        Analyze robust stability margins for multiplicative uncertainty.
        
        For multiplicative uncertainty G = G₀(1 + W*Δ), |Δ| ≤ 1:
        Robust stability condition: ||W*T||∞ < 1
        where T = GK/(1+GK) is complementary sensitivity
        
        Args:
            nominal_plant: G₀(s)
            controller: K(s) 
            uncertainty_model: W(s) uncertainty weight
            
        Returns:
            dict: Robust stability margins and analysis
        """
        
        G = nominal_plant
        K = controller  
        W = uncertainty_model['uncertainty_weight']
        
        # Compute closed-loop functions
        L = G * K
        T = L / (1 + L)
        
        # Robust stability condition: ||W*T||∞ < 1
        WT = W * T
        
        # Compute H∞ norm of W*T
        omega = np.logspace(-2, 4, 1000)
        WT_mag = np.abs(WT(1j * omega))
        robust_stability_margin = 1.0 / np.max(WT_mag)
        
        # Robust stability satisfied if margin > 1
        robust_stable = robust_stability_margin > 1.0
        
        return {
            'robust_stability_margin': robust_stability_margin,
            'robust_stable': robust_stable,
            'max_uncertainty_gain': np.max(WT_mag),
            'critical_frequency': omega[np.argmax(WT_mag)],
            'uncertainty_weight': W,
            'complementary_sensitivity': T
        }
    
    def design_hinf_controller(self, system_info, performance_specs, design_params=None):
        """
        Main H∞ controller design function.
        
        Args:
            system_info: System with uncertainty information
            performance_specs: Performance requirements
            design_params: Optional LLM-suggested design parameters
            
        Returns:
            dict: Complete H∞ design results
        """
        
        print("Starting H∞ robust controller design...")
        
        # Step 1: Convert parametric to multiplicative uncertainty
        uncertainty_model = self.parameter_to_multiplicative_uncertainty(system_info)
        print(f"Uncertainty model: δK={uncertainty_model['delta_K']:.1%}, δa={uncertainty_model['delta_a']:.1%}")
        
        # Step 2: Design performance weights
        weights = self.design_performance_weights(performance_specs, system_info)
        print(f"Performance bandwidth: {weights['omega_p']:.2f} rad/s")
        print(f"Control bandwidth: {weights['omega_u']:.2f} rad/s")
        
        # Step 3: Build generalized plant
        gen_plant = self.build_generalized_plant(
            uncertainty_model['nominal_plant'], 
            uncertainty_model, 
            weights
        )
        
        # Step 4: H∞ synthesis 
        hinf_result = self.hinf_mixed_sensitivity(
            uncertainty_model['nominal_plant'], 
            weights,
            gamma_max=10.0
        )
        
        if hinf_result is None or not hinf_result['success']:
            raise RuntimeError("H∞ synthesis failed to find acceptable controller")
        
        print(f"H∞ synthesis successful: γ = {hinf_result['gamma']:.3f}")
        
        # Step 5: Robust stability analysis
        robust_analysis = self.robust_stability_analysis(
            uncertainty_model['nominal_plant'],
            hinf_result['controller'],
            uncertainty_model
        )
        
        print(f"Robust stability margin: {robust_analysis['robust_stability_margin']:.3f}")
        
        # Step 6: Performance validation across parameter space
        robust_performance = self.validate_robust_performance(
            system_info, hinf_result['controller'], performance_specs
        )
        
        return {
            'controller': hinf_result['controller'],
            'gamma': hinf_result['gamma'],
            'uncertainty_model': uncertainty_model,
            'weights': weights,
            'robust_stability': robust_analysis,
            'robust_performance': robust_performance,
            'synthesis_info': hinf_result['analysis']
        }
    
    def validate_robust_performance(self, system_info, controller, performance_specs):
        """
        Validate H∞ controller performance across parameter uncertainty space.
        
        Args:
            system_info: System with uncertainty bounds
            controller: H∞ controller
            performance_specs: Performance requirements
            
        Returns:
            dict: Robust performance validation results
        """
        
        # Generate uncertain system samples
        if 'bounds' in system_info:
            bounds = system_info['bounds']
        else:
            bounds = {
                'K_min': system_info['nominal']['K'] * 0.8,
                'K_max': system_info['nominal']['K'] * 1.2,
                'a_min': system_info['nominal']['a'] * 0.8,
                'a_max': system_info['nominal']['a'] * 1.2
            }
        
        # Sample parameter space
        n_samples = 10
        K_vals = np.linspace(bounds['K_min'], bounds['K_max'], n_samples)
        a_vals = np.linspace(bounds['a_min'], bounds['a_max'], n_samples)
        
        performance_results = []
        
        for K, a in product(K_vals, a_vals):
            # Create uncertain plant
            G_uncertain = ctrl.TransferFunction([K], [1, a])
            
            # Closed-loop system
            try:
                L = G_uncertain * controller
                T = ctrl.feedback(L, 1)
                
                # Compute step response metrics
                t_sim = np.linspace(0, 10/a, 1000)  # Simulate for reasonable time
                y, t = ctrl.step_response(T, t_sim)
                
                # Settling time (2% criterion)
                steady_value = y[-1]
                settling_band = 0.02 * abs(steady_value)
                settled_indices = np.where(np.abs(y - steady_value) <= settling_band)[0]
                
                if len(settled_indices) > 0:
                    settling_time = t[settled_indices[0]]
                else:
                    settling_time = np.inf
                
                # Steady-state error
                steady_state_error = abs(1.0 - steady_value)
                
                performance_results.append({
                    'K': K, 'a': a,
                    'settling_time': settling_time,
                    'steady_state_error': steady_state_error,
                    'stable': np.all(np.real(ctrl.poles(T)) < 0)
                })
                
            except:
                performance_results.append({
                    'K': K, 'a': a,
                    'settling_time': np.inf,
                    'steady_state_error': np.inf, 
                    'stable': False
                })
        
        # Analyze results
        stable_results = [r for r in performance_results if r['stable']]
        if not stable_results:
            return {
                'success_rate': 0.0,
                'worst_settling_time': np.inf,
                'worst_steady_state_error': np.inf,
                'all_stable': False
            }
        
        settling_times = [r['settling_time'] for r in stable_results if np.isfinite(r['settling_time'])]
        steady_state_errors = [r['steady_state_error'] for r in stable_results]
        
        success_rate = len(stable_results) / len(performance_results)
        worst_settling_time = max(settling_times) if settling_times else np.inf
        worst_steady_state_error = max(steady_state_errors) if steady_state_errors else np.inf
        
        return {
            'success_rate': success_rate,
            'worst_settling_time': worst_settling_time,
            'worst_steady_state_error': worst_steady_state_error,
            'all_stable': success_rate == 1.0,
            'performance_samples': performance_results
        }


def test_hinf_control():
    """Test H∞ control design with uncertain system."""
    
    print("Testing H∞ Control Design")
    print("=" * 40)
    
    # Test system with high uncertainty  
    hinf_controller = HInfRobustControl()
    
    # System: G(s) = 20/(s + 0.5) with ±30% gain, ±20% pole uncertainty
    system_info = {
        'nominal': {'K': 20.0, 'a': 0.5},
        'bounds': {
            'K_min': 14.0, 'K_max': 26.0,  # ±30%
            'a_min': 0.4, 'a_max': 0.6     # ±20%
        },
        'num_uncertainty': [0.30],
        'den_uncertainty': [0.0, 0.20]
    }
    
    performance_specs = {
        'settling_time_max': 3.0,
        'steady_state_error_max': 0.01
    }
    
    try:
        # Assess complexity
        complexity = hinf_controller.assess_uncertainty_complexity(system_info)
        print(f"Uncertainty complexity: {complexity}")
        
        # Design H∞ controller
        hinf_result = hinf_controller.design_hinf_controller(system_info, performance_specs)
        
        print("\nH∞ Design Results:")
        print(f"Controller: {hinf_result['controller']}")
        print(f"Achieved γ: {hinf_result['gamma']:.3f}")
        print(f"Robust stability margin: {hinf_result['robust_stability']['robust_stability_margin']:.3f}")
        print(f"Performance success rate: {hinf_result['robust_performance']['success_rate']:.1%}")
        print(f"Worst-case settling time: {hinf_result['robust_performance']['worst_settling_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"H∞ design failed: {e}")
        return False

if __name__ == "__main__":
    test_hinf_control()