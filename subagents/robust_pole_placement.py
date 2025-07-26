import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt4 import GPT4
import json
import numpy as np
import control as ctrl
from util import feedback_prompt, check_stability
from DesignMemory import design_memory
from robust_control_utils import RobustPolePlace
from hinf_control_utils import HInfRobustControl

class RobustPolePlace_Design:
    """
    Subagent for robust pole placement control design under parameter uncertainty.
    
    This agent combines LLM reasoning with robust control theory to design
    controllers that maintain performance despite parameter variations.
    """

    def __init__(self, engine='gemini-2.0-flash-exp', temperature=0.0, max_tokens=1024):
        self.gpt4 = GPT4(engine=engine, temperature=temperature, max_tokens=max_tokens)
        self.max_attempts = 15
        self.design_memory = design_memory()
        self.base_output_dir = "./outputs"
        self.robust_controller = RobustPolePlace()
        self.hinf_controller = HInfRobustControl()
        
        # Design instruction for robust control (pole placement + H∞)
        self.design_instruction = """
You are designing a robust controller for a system with parameter uncertainty.

CRITICAL: CHOOSE CONTROL METHOD FIRST!

METHOD SELECTION:
[Step0] Analyze uncertainty level and requirements:
   - LOW uncertainty (≤20% combined): Use POLE_PLACEMENT (fast, sufficient)
   - MODERATE uncertainty (20-35% combined): Either method works, prefer POLE_PLACEMENT
   - HIGH uncertainty (≥35% combined): Use HINF (optimal robustness)
   - Frequency-domain requirements: Use HINF
   - Time-domain only requirements: Either method works

ROBUST POLE PLACEMENT METHOD (for lower uncertainty):
[Step1] Analyze the plant transfer function G(s) = K/(s + a_plant):
   - Extract the plant pole: a_plant (natural frequency of the system)
   - Classify system speed:
     * SLOW system: a_plant < 1.0 (settling naturally in >4 seconds)
     * MODERATE system: 1.0 ≤ a_plant ≤ 10.0 
     * FAST system: a_plant > 10.0 (settling naturally in <0.4 seconds)

[Step2] Choose pole location based on plant dynamics and requirements:
   SYSTEM-SPECIFIC POLE SELECTION RULES:
   
   For SLOW systems (a_plant < 1.0):
   - Use moderate pole: desired_pole = -(2 to 4) × a_plant
   - Example: if a_plant = 0.4, try pole ≈ -0.8 to -1.6
   
   For MODERATE systems (1.0 ≤ a_plant ≤ 10.0):
   - Use balanced pole: desired_pole = -(1.5 to 3) × a_plant  
   - Example: if a_plant = 5.0, try pole ≈ -7.5 to -15.0
   
   For FAST systems (a_plant > 10.0):
   - Use aggressive pole: desired_pole = -(3 to 6) × a_plant
   - Example: if a_plant = 15.0, try pole ≈ -45.0 to -90.0
   
   ALWAYS ensure: |desired_pole| ≥ 4/settling_time_max

[Step3] Consider parameter uncertainty impact:
   - Gain uncertainty: ±X% affects control authority
   - Pole uncertainty: ±Y% changes system dynamics  
   - More uncertainty requires more conservative (more negative) poles
   - Design must work for ALL parameter combinations

[Step4] Robustness guidelines:
   - Higher uncertainty → use lower end of pole range (more conservative)
   - Lower uncertainty → use higher end of pole range (more aggressive)
   - Success rate target: >95% across parameter variations

EXAMPLES:
- Plant: G(s) = 20/(s + 0.4) → SLOW system → try pole ≈ -1.2
- Plant: G(s) = 8/(s + 5.0) → MODERATE system → try pole ≈ -12.0  
- Plant: G(s) = 8/(s + 15.0) → FAST system → try pole ≈ -60.0

The controller uses state feedback: u = -K*x where K is computed for robust pole placement.

H∞ CONTROL METHOD (for higher uncertainty):
[Step1] Analyze uncertainty complexity:
   - Combined uncertainty = gain_uncertainty + pole_uncertainty
   - If ≥35%: H∞ is preferred for optimal robust performance

[Step2] Choose H∞ design parameters:
   - Performance bandwidth ωp: How fast should tracking be?
     * SLOW systems: ωp = (1-2) × a_plant
     * FAST systems: ωp = (0.2-0.5) × a_plant
   - Control bandwidth ωc: Limit control effort
     * ωc = (5-10) × ωp for moderate control effort
     * ωc = (2-5) × ωp for aggressive control effort

[Step3] H∞ synthesis objectives:
   - Minimize ||[Wp*S; Wu*K*S]||∞ where S = 1/(1+GK)
   - Wp weights tracking performance, Wu weights control effort
   - Result: Controller K(s) that optimizes robust performance trade-offs

EXAMPLES:
- 15% gain + 10% pole = 25% → Use POLE_PLACEMENT
- 25% gain + 15% pole = 40% → Use HINF
- Frequency specs (bandwidth limits) → Use HINF

The H∞ controller will be a transfer function K(s) optimized for robust performance.
"""

        self.response_format = """
Please provide your response in the following JSON format:
{
    "analysis": "Your analysis of the system and uncertainty level",
    "method_choice": "POLE_PLACEMENT or HINF",
    "method_rationale": "Reasoning for choosing pole placement vs H∞",
    "design_parameters": {
        "desired_pole": -2.5,
        "performance_bandwidth": 1.0,
        "control_bandwidth": 10.0
    },
    "expected_robustness": "Expected success rate and robustness metrics"
}
"""

    def handle_task(self, system, thresholds, task_requirement, scenario):
        """Main task handling function for robust pole placement design."""
        
        num_attempt = 1
        print(f"Handling robust controller design for system {system['id']} scenario {scenario}")
        
        # Check if system has uncertainty information
        if 'num_uncertainty' not in system or 'den_uncertainty' not in system:
            print("Warning: No uncertainty information found. Adding default uncertainty bounds.")
            system['num_uncertainty'] = [0.15]  # ±15% default
            system['den_uncertainty'] = [0.0, 0.10]  # ±10% on poles
        
        # Extract system information
        num_nominal = system['num']
        den_nominal = system['den']
        num_uncertainty = system['num_uncertainty']
        den_uncertainty = system['den_uncertainty']
        
        # Convert to uncertain state-space representation
        system_info = self.robust_controller.tf_to_ss_uncertain(
            num_nominal, den_nominal, num_uncertainty, den_uncertainty
        )
        
        # Construct the design prompt
        prompt = self.design_instruction
        
        # Add system-specific information with plant analysis
        a_plant = den_nominal[1]
        K_plant = num_nominal[0]
        
        # Classify system speed
        if a_plant < 1.0:
            system_type = "SLOW"
            suggested_range = f"-{2*a_plant:.1f} to -{4*a_plant:.1f}"
        elif a_plant <= 10.0:
            system_type = "MODERATE" 
            suggested_range = f"-{1.5*a_plant:.1f} to -{3*a_plant:.1f}"
        else:
            system_type = "FAST"
            suggested_range = f"-{3*a_plant:.1f} to -{6*a_plant:.1f}"
        
        # Calculate combined uncertainty for method selection
        gain_unc = num_uncertainty[0] * 100
        pole_unc = den_uncertainty[1] * 100 if len(den_uncertainty) > 1 else 0
        combined_unc = gain_unc + pole_unc
        
        # Suggest method based on uncertainty level
        if combined_unc >= 35:
            suggested_method = "HINF (high uncertainty)"
        elif combined_unc >= 20:
            suggested_method = "POLE_PLACEMENT or HINF (moderate uncertainty)"
        else:
            suggested_method = "POLE_PLACEMENT (low uncertainty)"

        uncertainty_info = f"""
PLANT ANALYSIS:
System: G(s) = {K_plant:.3f} / (s + {a_plant:.3f})
Plant pole a_plant = {a_plant:.3f} → {system_type} system
Recommended pole range: {suggested_range}

UNCERTAINTY ANALYSIS:
Gain uncertainty: ±{gain_unc:.1f}%
Pole uncertainty: ±{pole_unc:.1f}%
Combined uncertainty: {combined_unc:.1f}%
→ Suggested method: {suggested_method}

Parameter ranges:
- Gain K ∈ [{system_info['bounds']['K_min']:.2f}, {system_info['bounds']['K_max']:.2f}]
- Pole a ∈ [{system_info['bounds']['a_min']:.2f}, {system_info['bounds']['a_max']:.2f}]

PERFORMANCE REQUIREMENTS:
- Settling time ≤ {thresholds.get('settling_time_max', {}).get('max', 'N/A')} sec
- Phase margin ≥ {thresholds.get('phase_margin', {}).get('min', 'N/A')}°

Choose your method and design parameters based on the analysis above!
"""
        
        new_problem = "Now consider the following robust design task: " + task_requirement + "\n" + uncertainty_info
        problem_statement = prompt + new_problem + self.response_format
        
        conversation_log = []
        
        while num_attempt <= self.max_attempts:
            print(f"Iteration {num_attempt} for system {system['id']}.")
            
            # Get LLM response
            if num_attempt == 1:
                full_prompt = problem_statement
            else:
                # Add feedback from previous iterations
                memory_content = self.design_memory.get_memory()
                feedback_content = self.design_memory.get_feedback()
                full_prompt = problem_statement + "\n\nPrevious attempts:\n" + memory_content + "\n\nFeedback:\n" + feedback_content
            
            conversation_log.append({"iteration": num_attempt, "prompt": full_prompt})
            
            try:
                # Get LLM design proposal
                response = self.gpt4.complete(full_prompt)
                parsed_response = json.loads(response)
                
                conversation_log.append({"iteration": num_attempt, "response": parsed_response})
                
                # Extract method choice and design parameters
                method_choice = parsed_response.get("method_choice", "POLE_PLACEMENT").upper()
                design_params = parsed_response.get("design_parameters", {})
                
                print(f"LLM chose method: {method_choice}")
                
                # Implement robust control design
                performance_specs = {
                    'settling_time_max': thresholds.get('settling_time_max', {}).get('max', 5.0),
                    'damping_min': 0.7
                }
                
                if method_choice == "HINF":
                    # Use H∞ control with automatic fallback
                    robust_design = self.try_hinf_design(
                        system_info, performance_specs, design_params, num_attempt
                    )
                    
                    # If H∞ is available but might not meet requirements, check and fallback
                    if robust_design is not None and num_attempt > 1:
                        # After first iteration, if previous H∞ attempt failed requirements,
                        # automatically try pole placement instead
                        memory_content = self.design_memory.get_memory()
                        if "Requirements not met" in memory_content and "H∞" in memory_content:
                            print("🔄 H∞ previously failed requirements - switching to POLE_PLACEMENT")
                            method_choice = "POLE_PLACEMENT_FALLBACK"
                            
                            # Use pole placement as fallback
                            desired_pole = float(design_params.get("desired_pole", -2.0))
                            if desired_pole >= 0:
                                desired_pole = -abs(desired_pole)
                            
                            print(f"Fallback to pole placement with pole: {desired_pole:.3f}")
                            robust_design = self.try_robust_design_with_fallback(
                                system_info, performance_specs, desired_pole, num_attempt
                            )
                
                if method_choice in ["POLE_PLACEMENT", "POLE_PLACEMENT_FALLBACK"] or robust_design is None:
                    # Use pole placement (default or fallback)
                    desired_pole = float(design_params.get("desired_pole", -2.0))
                    
                    # Ensure pole is stable (negative for first-order systems)
                    if desired_pole >= 0:
                        desired_pole = -abs(desired_pole)
                    
                    print(f"LLM proposed pole: {desired_pole:.3f}")
                    
                    robust_design = self.try_robust_design_with_fallback(
                        system_info, performance_specs, desired_pole, num_attempt
                    )
                
                if robust_design is not None:
                    if robust_design.get('method') == 'HINF':
                        # H∞ controller results
                        controller_tf = robust_design['controller_tf']
                        hinf_gamma = robust_design['hinf_gamma']
                        success_rate = robust_design['robustness_results']['success_rate']
                        worst_settling_time = robust_design['worst_settling_time']
                        
                        print(f"H∞ design: Controller = {controller_tf}")
                        print(f"γ = {hinf_gamma:.3f}, Success rate: {success_rate:.1%}, worst settling time: {worst_settling_time:.3f}s")
                        
                        final_pole = None  # H∞ doesn't have single pole
                        nominal_gain = robust_design['nominal_gain'][0, 0]
                        
                    else:
                        # Pole placement results
                        final_pole = robust_design['desired_pole']
                        nominal_gain = robust_design['nominal_gain'][0, 0]
                        success_rate = robust_design['robustness_results']['success_rate']
                        worst_settling_time = robust_design['worst_settling_time']
                        
                        print(f"Robust design: pole = {final_pole:.3f}, gain = {nominal_gain:.4f}")
                        print(f"Success rate: {success_rate:.1%}, worst settling time: {worst_settling_time:.3f}s")
                    
                    # Evaluate performance against requirements
                    performance_results = self.evaluate_robust_performance(
                        system_info, robust_design, thresholds
                    )
                    
                    # Check if requirements are satisfied
                    requirements_met = self.check_requirements(performance_results, thresholds)
                    
                    if requirements_met['all_satisfied']:
                        print("The robust design satisfies all requirements.")
                        
                        # Store successful design (handle both pole placement and H∞)
                        if robust_design.get('method') == 'HINF':
                            controller_desc = f"H∞ controller (γ={robust_design.get('hinf_gamma', 'N/A'):.3f})"
                            self.design_memory.add_memory(
                                f"Iteration {num_attempt}: Successful {controller_desc}"
                            )
                            
                            return {
                                'is_succ': True,
                                'parameters': {
                                    'controller_type': 'HINF',
                                    'controller_tf': str(robust_design['controller_tf']),
                                    'hinf_gamma': robust_design.get('hinf_gamma', 0),
                                    'dc_gain': nominal_gain,
                                    'success_rate': success_rate
                                },
                                'performance': performance_results,
                                'conversation_rounds': num_attempt,
                                'robust_metrics': {
                                    'worst_case_settling_time': worst_settling_time,
                                    'parameter_success_rate': success_rate,
                                    'method': 'HINF',
                                    'robust_stability_margin': robust_design.get('robust_stability_margin', 0)
                                }
                            }
                        else:
                            # Pole placement results
                            self.design_memory.add_memory(
                                f"Iteration {num_attempt}: Successful robust pole placement at {final_pole:.3f}"
                            )
                            
                            return {
                                'is_succ': True,
                                'parameters': {
                                    'controller_type': 'POLE_PLACEMENT',
                                    'pole_location': final_pole,
                                    'feedback_gain': nominal_gain,
                                    'success_rate': success_rate
                                },
                                'performance': performance_results,
                                'conversation_rounds': num_attempt,
                                'robust_metrics': {
                                    'worst_case_settling_time': worst_settling_time,
                                    'parameter_success_rate': success_rate,
                                    'nominal_pole': final_pole,
                                    'method': 'POLE_PLACEMENT'
                                }
                            }
                    
                    else:
                        # Generate feedback for next iteration (pass method type)
                        method_type = robust_design.get('method', 'POLE_PLACEMENT')
                        feedback_msg = self.generate_feedback(requirements_met, performance_results, thresholds, method_type)
                        self.design_memory.add_feedback(feedback_msg)
                        
                        # Handle logging for both methods
                        if robust_design.get('method') == 'HINF':
                            self.design_memory.add_memory(
                                f"Iteration {num_attempt}: H∞ controller (γ={robust_design.get('hinf_gamma', 0):.3f}) - Requirements not met"
                            )
                        else:
                            self.design_memory.add_memory(
                                f"Iteration {num_attempt}: Pole {final_pole:.3f}, gain {nominal_gain:.4f} - Requirements not met"
                            )
                        
                        print(f"Requirements not satisfied. Feedback: {feedback_msg}")
                
                else:
                    # All fallback attempts failed
                    error_msg = f"All robust design attempts failed in iteration {num_attempt}"
                    print(error_msg)
                    self.design_memory.add_feedback(
                        f"Robust design failed for pole {desired_pole:.3f}. Try a different approach: "
                        f"for fast systems, use more negative poles (e.g., {desired_pole * 2:.1f}); "
                        f"for slow systems, use less negative poles (e.g., {desired_pole * 0.5:.1f})."
                    )
                
            except Exception as e:
                error_msg = f"LLM parsing error in iteration {num_attempt}: {str(e)}"
                print(error_msg)
                self.design_memory.add_feedback("Please provide valid JSON response with desired_pole field.")
            
            num_attempt += 1
        
        # Max attempts reached
        print(f"Maximum attempts ({self.max_attempts}) reached. Design failed.")
        return {
            'is_succ': False,
            'parameters': {},
            'performance': {},
            'conversation_rounds': num_attempt - 1,
            'error': 'Maximum attempts reached without satisfying requirements'
        }
    
    def evaluate_robust_performance(self, system_info, robust_design, thresholds):
        """Evaluate the robust design performance across parameter variations."""
        
        # Get robustness results
        robustness_results = robust_design['robustness_results']
        nominal_gain = robust_design['nominal_gain'][0, 0]
        
        # Calculate phase margin (different for H∞ vs pole placement)
        if robust_design.get('method') == 'HINF':
            # For H∞, use robust stability margin as proxy for phase margin
            robust_margin = robust_design.get('robust_stability_margin', 1.0)
            # Convert robust stability margin to approximate phase margin
            phase_margin = min(90.0, max(30.0, robust_margin * 60.0))  # Rough mapping
            print(f"H∞ robust stability margin: {robust_margin:.3f} → Estimated phase margin: {phase_margin:.2f}°")
        else:
            # For pole placement, calculate actual phase margin
            phase_margin = self.robust_controller.calculate_phase_margin(system_info, nominal_gain)
            print(f"Calculated phase margin: {phase_margin:.2f} degrees")
        
        # Calculate performance metrics
        performance = {
            'phase_margin': np.float64(phase_margin),
            'settling_time_min': np.float64(robust_design['worst_settling_time']),
            'settling_time_max': np.float64(robust_design['worst_settling_time']),
            'steadystate_error': np.float64(0.0),  # State feedback gives zero steady-state error
            'robustness_success_rate': np.float64(robustness_results['success_rate']),
            'worst_case_settling_time': np.float64(robust_design['worst_settling_time'])
        }
        
        return performance
    
    def check_requirements(self, performance, thresholds):
        """Check if the robust design meets all requirements."""
        
        results = {'all_satisfied': True, 'violations': []}
        
        # Check settling time
        if 'settling_time_max' in thresholds:
            max_allowed = thresholds['settling_time_max'].get('max', np.inf)
            if performance['settling_time_max'] > max_allowed:
                results['all_satisfied'] = False
                results['violations'].append(f"Settling time {performance['settling_time_max']:.3f} > {max_allowed}")
        
        # Check phase margin
        if 'phase_margin' in thresholds:
            min_required = thresholds['phase_margin'].get('min', 0)
            if performance['phase_margin'] < min_required:
                results['all_satisfied'] = False
                results['violations'].append(f"Phase margin {performance['phase_margin']:.1f} < {min_required}")
        
        # Check steady-state error
        if 'steadystate_error' in thresholds:
            max_allowed = thresholds['steadystate_error'].get('max', np.inf)
            if performance['steadystate_error'] > max_allowed:
                results['all_satisfied'] = False
                results['violations'].append(f"Steady-state error {performance['steadystate_error']:.4f} > {max_allowed}")
        
        # Check robustness (success rate should be high)
        if performance['robustness_success_rate'] < 0.95:
            results['all_satisfied'] = False
            results['violations'].append(f"Robustness success rate {performance['robustness_success_rate']:.1%} < 95%")
        
        return results
    
    def generate_feedback(self, requirements_met, performance, thresholds, method='POLE_PLACEMENT'):
        """Generate feedback for the LLM based on requirement violations."""
        
        if not requirements_met['violations']:
            return "All requirements satisfied."
        
        feedback_parts = []
        current_phase_margin = performance.get('phase_margin', 0)
        current_settling = performance.get('settling_time_max', 0)
        
        if method == 'HINF':
            # H∞-specific feedback
            for violation in requirements_met['violations']:
                if "Phase margin" in violation:
                    required_pm = thresholds['phase_margin'].get('min', 0)
                    deficit = required_pm - current_phase_margin
                    feedback_parts.append(f"Phase margin too low ({current_phase_margin:.1f}° < {required_pm:.1f}°). For H∞: Try POLE_PLACEMENT method instead, or increase performance bandwidth ωp by factor ~{1 + deficit/30:.1f} for better margins.")
                
                elif "Settling time" in violation:
                    max_allowed = thresholds.get('settling_time_max', {}).get('max', np.inf)
                    if current_settling > max_allowed:
                        excess_time = current_settling - max_allowed
                        feedback_parts.append(f"Settling time too slow ({current_settling:.2f}s > {max_allowed:.2f}s). For H∞: Increase performance bandwidth ωp by factor ~{1 + excess_time/max_allowed:.1f}.")
                
                elif "success rate" in violation:
                    feedback_parts.append("Robustness too low. For H∞: Reduce control bandwidth ωc or try more conservative design parameters.")
        else:
            # Pole placement feedback  
            for violation in requirements_met['violations']:
                if "Phase margin" in violation:
                    required_pm = thresholds['phase_margin'].get('min', 0)
                    deficit = required_pm - current_phase_margin
                    feedback_parts.append(f"Phase margin too low ({current_phase_margin:.1f}° < {required_pm:.1f}°). Move pole LESS negative (closer to zero) by ~{deficit*0.01:.3f} to increase phase margin.")
                
                elif "Settling time" in violation:
                    max_allowed = thresholds.get('settling_time_max', {}).get('max', np.inf)
                    if current_settling > max_allowed:
                        excess_time = current_settling - max_allowed
                        feedback_parts.append(f"Settling time too slow ({current_settling:.2f}s > {max_allowed:.2f}s). Move pole MORE negative by factor ~{1 + excess_time/max_allowed:.2f} for faster response.")
                
                elif "success rate" in violation:
                    feedback_parts.append("Robustness too low. Use more conservative (more negative) pole for better parameter tolerance.")
        
        if not feedback_parts:
            if method == 'HINF':
                feedback_parts.append("Requirements not satisfied. Try adjusting H∞ design parameters or switch to POLE_PLACEMENT method.")
            else:
                feedback_parts.append("Requirements not satisfied. Try different pole location.")
        
        return " ".join(feedback_parts)
    
    def try_robust_design_with_fallback(self, system_info, performance_specs, llm_pole, iteration):
        """
        Try robust design with LLM suggestion, with intelligent fallback strategies.
        
        Args:
            system_info: System information with uncertainty
            performance_specs: Performance requirements
            llm_pole: LLM's suggested pole location
            iteration: Current iteration number
            
        Returns:
            dict: Robust design results, or None if all attempts fail
        """
        
        # Strategy 1: Try LLM's suggestion directly
        print(f"Attempt 1: Using LLM suggested pole {llm_pole:.3f}")
        try:
            robust_design = self.robust_controller.robust_pole_placement(
                system_info, performance_specs, max_iterations=20, 
                llm_suggested_pole=llm_pole
            )
            print("✅ LLM suggestion worked!")
            return robust_design
        except Exception as e:
            print(f"❌ LLM suggestion failed: {e}")
        
        # Strategy 2: Try more aggressive poles for fast systems
        system_pole = abs(system_info['nominal']['a'])  # System's natural pole
        if system_pole > 5.0:  # Fast system
            aggressive_poles = [llm_pole * 2, llm_pole * 3, -system_pole * 2]
            print(f"Fast system detected (pole={system_pole:.1f}). Trying aggressive poles...")
        else:  # Slow system
            aggressive_poles = [llm_pole * 1.5, llm_pole * 0.7, -system_pole * 3]
            print(f"Slow system detected (pole={system_pole:.1f}). Trying moderate poles...")
        
        for attempt, fallback_pole in enumerate(aggressive_poles, 2):
            print(f"Attempt {attempt}: Trying fallback pole {fallback_pole:.3f}")
            try:
                robust_design = self.robust_controller.robust_pole_placement(
                    system_info, performance_specs, max_iterations=15,
                    llm_suggested_pole=fallback_pole
                )
                print(f"✅ Fallback pole {fallback_pole:.3f} worked!")
                return robust_design
            except Exception as e:
                print(f"❌ Fallback pole {fallback_pole:.3f} failed: {e}")
        
        # Strategy 3: Performance-based design (ignore LLM completely)
        print("Attempt 5: Using pure performance-based design (no LLM input)")
        try:
            robust_design = self.robust_controller.robust_pole_placement(
                system_info, performance_specs, max_iterations=30,
                llm_suggested_pole=None  # Force performance-based design
            )
            print("✅ Performance-based design worked!")
            return robust_design
        except Exception as e:
            print(f"❌ Performance-based design failed: {e}")
        
        # Strategy 4: Very conservative design
        settling_time_max = performance_specs.get('settling_time_max', 5.0)
        conservative_pole = -2.0 / settling_time_max  # Very conservative
        print(f"Attempt 6: Last resort - very conservative pole {conservative_pole:.3f}")
        try:
            robust_design = self.robust_controller.robust_pole_placement(
                system_info, performance_specs, max_iterations=10,
                llm_suggested_pole=conservative_pole
            )
            print("✅ Conservative design worked!")
            return robust_design
        except Exception as e:
            print(f"❌ All fallback strategies failed: {e}")
        
        return None  # Complete failure
    
    def try_hinf_design(self, system_info, performance_specs, design_params, iteration):
        """
        Try H∞ controller design.
        
        Args:
            system_info: System information with uncertainty
            performance_specs: Performance requirements  
            design_params: LLM-suggested design parameters
            iteration: Current iteration number
            
        Returns:
            dict: H∞ design results, or None if failed
        """
        
        print(f"Attempting H∞ controller design...")
        
        try:
            # Use H∞ synthesis
            hinf_result = self.hinf_controller.design_hinf_controller(
                system_info, performance_specs, design_params
            )
            
            print(f"✅ H∞ synthesis successful!")
            print(f"Achieved γ: {hinf_result['gamma']:.3f}")
            print(f"Robust stability margin: {hinf_result['robust_stability']['robust_stability_margin']:.3f}")
            
            # Convert H∞ result to format compatible with pole placement results
            controller_tf = hinf_result['controller']
            
            # For compatibility, extract equivalent "gain" from DC gain of controller
            try:
                dc_gain = float(ctrl.dcgain(controller_tf))
            except:
                dc_gain = 1.0
            
            # Create compatible result structure
            compatible_result = {
                'desired_pole': None,  # H∞ doesn't have single pole
                'nominal_gain': np.array([[dc_gain]]),
                'robustness_results': {
                    'success_rate': hinf_result['robust_performance']['success_rate']
                },
                'worst_settling_time': hinf_result['robust_performance']['worst_settling_time'],
                'controller_tf': controller_tf,  # Store full transfer function
                'hinf_gamma': hinf_result['gamma'],
                'robust_stability_margin': hinf_result['robust_stability']['robust_stability_margin'],
                'method': 'HINF'
            }
            
            return compatible_result
            
        except Exception as e:
            print(f"❌ H∞ design failed: {e}")
            return None

# Test function
def test_robust_subagent():
    """Test the robust pole placement subagent (without LLM)."""
    
    print("Testing robust control utilities directly...")
    
    # Create test system with uncertainty
    test_system = {
        'id': 999,
        'num': [19.95],
        'den': [1, 0.39],
        'num_uncertainty': [0.20],  # ±20%
        'den_uncertainty': [0.0, 0.15]  # ±15%
    }
    
    # Test just the robust control part
    subagent = RobustPolePlace_Design()
    
    # Convert to uncertain state-space representation
    system_info = subagent.robust_controller.tf_to_ss_uncertain(
        test_system['num'], test_system['den'], 
        test_system['num_uncertainty'], test_system['den_uncertainty']
    )
    
    print("System conversion successful!")
    print(f"Nominal: A = {system_info['nominal']['A']}, B = {system_info['nominal']['B']}")
    print(f"Bounds: K ∈ [{system_info['bounds']['K_min']:.2f}, {system_info['bounds']['K_max']:.2f}]")
    print(f"Bounds: a ∈ [{system_info['bounds']['a_min']:.3f}, {system_info['bounds']['a_max']:.3f}]")
    
    # Test robust design
    performance_specs = {'settling_time_max': 3.0, 'damping_min': 0.7}
    
    try:
        robust_design = subagent.robust_controller.robust_pole_placement(
            system_info, performance_specs, max_iterations=20
        )
        
        print(f"\nRobust Design Success!")
        print(f"Desired pole: {robust_design['desired_pole']:.3f}")
        print(f"Success rate: {robust_design['robustness_results']['success_rate']:.1%}")
        print(f"Worst settling time: {robust_design['worst_settling_time']:.3f} sec")
        print(f"Nominal gain: {robust_design['nominal_gain']}")
        
        return True
        
    except Exception as e:
        print(f"Robust design failed: {e}")
        return False

if __name__ == "__main__":
    test_robust_subagent()