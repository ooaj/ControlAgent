import numpy as np
import control as ctrl
from itertools import product
from scipy import signal
import matplotlib.pyplot as plt

class RobustPolePlace:
    """
    Utility class for robust pole placement under parameter uncertainty.
    
    This implements dyadic state feedback for first-order systems with
    parameter bounds, ensuring closed-loop poles remain in desired regions
    despite parameter variations.
    """
    
    def __init__(self):
        self.tolerance = 1e-8
    
    def tf_to_ss_uncertain(self, num_nominal, den_nominal, num_uncertainty, den_uncertainty):
        """
        Convert transfer function with uncertainty bounds to state-space representations.
        
        Args:
            num_nominal: Nominal numerator coefficients [K]
            den_nominal: Nominal denominator coefficients [1, a] 
            num_uncertainty: Uncertainty bounds on numerator [±δK/K]
            den_uncertainty: Uncertainty bounds on denominator [0, ±δa/a]
            
        Returns:
            dict: Contains nominal system and uncertainty bounds
        """
        
        # Extract nominal parameters
        K_nom = num_nominal[0]
        a_nom = den_nominal[1]
        
        # Calculate uncertainty bounds
        K_min = K_nom * (1 - num_uncertainty[0])
        K_max = K_nom * (1 + num_uncertainty[0])
        a_min = a_nom * (1 - den_uncertainty[1]) 
        a_max = a_nom * (1 + den_uncertainty[1])
        
        # Nominal state-space: ẋ = -a*x + b*u, y = x
        # For G(s) = K/(s+a), we get: A = [-a], B = [K], C = [1], D = [0]
        A_nom = np.array([[-a_nom]])
        B_nom = np.array([[K_nom]])
        C_nom = np.array([[1.0]])
        D_nom = np.array([[0.0]])
        
        return {
            'nominal': {
                'A': A_nom, 'B': B_nom, 'C': C_nom, 'D': D_nom,
                'K': K_nom, 'a': a_nom
            },
            'bounds': {
                'K_min': K_min, 'K_max': K_max,
                'a_min': a_min, 'a_max': a_max
            }
        }
    
    def generate_uncertain_systems(self, system_info, n_samples=20):
        """
        Generate sample systems within uncertainty bounds for Monte Carlo analysis.
        
        Args:
            system_info: Output from tf_to_ss_uncertain()
            n_samples: Number of samples per parameter (total = n_samples²)
            
        Returns:
            list: List of (A, B) matrices representing uncertain systems
        """
        bounds = system_info['bounds']
        
        # Create parameter grids
        K_vals = np.linspace(bounds['K_min'], bounds['K_max'], n_samples)
        a_vals = np.linspace(bounds['a_min'], bounds['a_max'], n_samples)
        
        uncertain_systems = []
        
        for K, a in product(K_vals, a_vals):
            A = np.array([[-a]])
            B = np.array([[K]])
            uncertain_systems.append((A, B))
            
        return uncertain_systems
    
    def pole_placement_gain(self, A, B, desired_poles):
        """
        Compute state feedback gain for desired pole placement.
        
        Args:
            A, B: State-space matrices
            desired_poles: List of desired closed-loop pole locations
            
        Returns:
            K: Feedback gain matrix such that eig(A - B*K) = desired_poles
        """
        try:
            # Use control library's place function
            K = ctrl.place(A, B, desired_poles)
            return K
        except:
            # Fallback: manual calculation for 1st order systems
            if A.shape[0] == 1 and len(desired_poles) == 1:
                # For ẋ = ax + bu, u = -kx
                # Closed-loop: ẋ = (a - bk)x
                # Want: a - bk = desired_pole
                # So: k = (a - desired_pole) / b
                a = A[0, 0]
                b = B[0, 0]
                desired_pole = desired_poles[0]
                
                if abs(b) < self.tolerance:
                    raise ValueError("System not controllable (B ≈ 0)")
                    
                k = (a - desired_pole) / b
                return np.array([[k]])
            else:
                raise ValueError("Cannot compute gain for this system")
    
    def evaluate_robustness(self, system_info, desired_poles, n_samples=20):
        """
        Evaluate robustness of pole placement across parameter uncertainty.
        
        Args:
            system_info: System with uncertainty bounds
            desired_poles: Target pole locations
            n_samples: Samples for Monte Carlo evaluation
            
        Returns:
            dict: Robustness metrics including worst-case poles, success rate
        """
        uncertain_systems = self.generate_uncertain_systems(system_info, n_samples)
        
        results = {
            'success_count': 0,
            'total_count': len(uncertain_systems),
            'achieved_poles': [],
            'gains': [],
            'failed_systems': []
        }
        
        for i, (A, B) in enumerate(uncertain_systems):
            try:
                K = self.pole_placement_gain(A, B, desired_poles)
                A_cl = A - B @ K
                achieved_poles = np.linalg.eigvals(A_cl)
                
                results['achieved_poles'].append(achieved_poles)
                results['gains'].append(K)
                results['success_count'] += 1
                
            except Exception as e:
                results['failed_systems'].append({
                    'index': i, 'A': A, 'B': B, 'error': str(e)
                })
        
        # Calculate metrics
        results['success_rate'] = results['success_count'] / results['total_count']
        
        if results['achieved_poles']:
            all_poles = np.array(results['achieved_poles']).flatten()
            results['pole_statistics'] = {
                'mean': np.mean(all_poles),
                'std': np.std(all_poles),
                'min': np.min(all_poles),
                'max': np.max(all_poles)
            }
        
        return results
    
    def robust_pole_placement(self, system_info, performance_specs, max_iterations=50, llm_suggested_pole=None):
        """
        Design robust pole placement considering parameter uncertainty.
        
        This is the main function that finds pole locations that satisfy
        performance requirements across all parameter variations.
        
        Args:
            system_info: System with uncertainty information
            performance_specs: Dict with 'settling_time_max', 'damping_min', etc.
            max_iterations: Maximum design iterations
            llm_suggested_pole: LLM's suggested pole location (used as starting point)
            
        Returns:
            dict: Robust design results including gain and performance metrics
        """
        
        # Extract performance requirements
        settling_time_max = performance_specs.get('settling_time_max', 1.0)
        damping_min = performance_specs.get('damping_min', 0.7)
        
        # Use LLM suggestion as starting point if provided
        if llm_suggested_pole is not None and llm_suggested_pole < 0:
            desired_pole = llm_suggested_pole
            print(f"Starting robust design from LLM suggested pole: {desired_pole:.3f}")
        else:
            # For first-order systems, settling time ≈ 4/|Re(pole)|
            # So |Re(pole)| ≥ 4/settling_time_max
            min_pole_magnitude = 4.0 / settling_time_max
            desired_pole = -min_pole_magnitude * 1.5  # Add safety margin
            print(f"Starting robust design from performance-based pole: {desired_pole:.3f}")
        
        best_design = None
        best_performance = -np.inf
        
        for iteration in range(max_iterations):
            # Evaluate current design
            robustness_results = self.evaluate_robustness(
                system_info, [desired_pole], n_samples=15
            )
            
            if robustness_results['success_rate'] < 0.8:
                # Too aggressive, make pole more negative (faster/more stable)
                desired_pole *= 1.2
                continue
            
            # Check if settling time requirements are met
            if 'pole_statistics' in robustness_results:
                worst_pole = robustness_results['pole_statistics']['max']  # Least negative = slowest
                worst_settling_time = 4.0 / abs(worst_pole) if worst_pole < 0 else np.inf
                
                if worst_settling_time <= settling_time_max:
                    # Good design found
                    performance_score = robustness_results['success_rate'] - abs(desired_pole) * 0.01
                    
                    if performance_score > best_performance:
                        best_performance = performance_score
                        best_design = {
                            'desired_pole': desired_pole,
                            'robustness_results': robustness_results,
                            'worst_settling_time': worst_settling_time,
                            'nominal_gain': self.pole_placement_gain(
                                system_info['nominal']['A'],
                                system_info['nominal']['B'],
                                [desired_pole]
                            )
                        }
                
                # Try slightly different pole for next iteration
                desired_pole *= (0.95 if iteration % 2 == 0 else 1.05)
            else:
                # No successful placements, make much more conservative
                desired_pole *= 2.0
        
        if best_design is None:
            raise RuntimeError("Could not find robust pole placement solution")
        
        return best_design
    
    def calculate_phase_margin(self, system_info, feedback_gain):
        """
        Calculate the actual phase margin for state feedback system.
        
        For state feedback u = -K*x, the loop transfer function is L(s) = K*G(s)
        where G(s) is the plant transfer function.
        
        Args:
            system_info: System information with nominal parameters
            feedback_gain: State feedback gain K
            
        Returns:
            float: Phase margin in degrees
        """
        try:
            # Get nominal system parameters
            K_plant = system_info['nominal']['K']  # Plant gain
            a_nom = system_info['nominal']['a']    # Plant pole
            
            # For first-order system G(s) = K/(s+a), with state feedback gain K_fb
            # Loop transfer function: L(s) = K_fb * K / (s + a)
            # This is still first-order, so phase margin calculation:
            
            # At crossover frequency wc: |L(jw)| = 1
            # |K_fb * K / (jw + a)| = 1
            # K_fb * K / sqrt(wc^2 + a^2) = 1
            # wc = sqrt((K_fb * K)^2 - a^2)
            
            loop_gain = feedback_gain * K_plant
            
            if loop_gain <= abs(a_nom):
                # No crossover - system has infinite phase margin
                return 90.0
            
            wc = np.sqrt(loop_gain**2 - a_nom**2)
            
            # Phase of L(jwc) = phase of (K_fb * K) - phase of (jwc + a)
            # = 0 - arctan(wc/a)
            phase_at_crossover = -np.arctan(wc / a_nom) * 180 / np.pi
            
            # Phase margin = 180 + phase_at_crossover
            phase_margin = 180 + phase_at_crossover
            
            return max(0.0, phase_margin)  # Ensure non-negative
            
        except Exception as e:
            print(f"Warning: Phase margin calculation failed: {e}")
            # Fallback: estimate based on closed-loop pole location
            closed_loop_pole = system_info['nominal']['A'][0,0] - system_info['nominal']['B'][0,0] * feedback_gain
            if closed_loop_pole < 0:
                # Stable system - estimate phase margin
                return min(90.0, abs(closed_loop_pole) * 10)  # Rough estimate
            else:
                return 0.0  # Unstable

def test_robust_control():
    """Test function to verify the robust control utilities."""
    
    # Test system: G(s) = 20/(s + 0.5) with ±20% gain, ±15% pole uncertainty
    rpp = RobustPolePlace()
    
    num_nominal = [20.0]
    den_nominal = [1.0, 0.5]
    num_uncertainty = [0.20]  # ±20%
    den_uncertainty = [0.0, 0.15]  # ±15% on pole
    
    # Convert to uncertain state-space
    system_info = rpp.tf_to_ss_uncertain(
        num_nominal, den_nominal, num_uncertainty, den_uncertainty
    )
    
    print("Nominal system:")
    print(f"A = {system_info['nominal']['A']}")
    print(f"B = {system_info['nominal']['B']}")
    print(f"Parameter bounds: K ∈ [{system_info['bounds']['K_min']:.2f}, {system_info['bounds']['K_max']:.2f}]")
    print(f"Parameter bounds: a ∈ [{system_info['bounds']['a_min']:.2f}, {system_info['bounds']['a_max']:.2f}]")
    
    # Performance specifications
    performance_specs = {
        'settling_time_max': 2.0,  # Max 2 seconds settling time
        'damping_min': 0.7
    }
    
    # Design robust controller
    try:
        robust_design = rpp.robust_pole_placement(system_info, performance_specs)
        
        print(f"\nRobust Design Results:")
        print(f"Desired pole: {robust_design['desired_pole']:.3f}")
        print(f"Success rate: {robust_design['robustness_results']['success_rate']:.1%}")
        print(f"Worst-case settling time: {robust_design['worst_settling_time']:.3f} sec")
        print(f"Nominal gain: {robust_design['nominal_gain']}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_robust_control()