from gpt4 import GPT4
import json
from util import loop_shaping, feedback_prompt, check_stability_pid
from instruction import response_format_PID, overall_instruction_PID, overall_instruction_RHP_Pole_higherorder_PID
from DesignMemory import design_memory
import control as ctrl
import numpy as np
import os


class higher_ord_Design:

    def __init__(self, engine='gemini-2.0-flash-exp', temperature=0.0, max_tokens=1024):
        self.gpt4 = GPT4(engine=engine, temperature=temperature, max_tokens=max_tokens)
        self.max_attempts = 30
        self.design_memory = design_memory()
        self.base_output_dir = "./outputs"
    
    def handle_task(self, system, thresholds, task_requirement, scenario):
        num_attempt = 1
        # Implement the task handling process
        print(f"Handling controller design for system {system['id']} in scenario {scenario}")

        # Construct the design prompt
        num = [system['num'][0]]
        den = system['den']
        G = ctrl.TransferFunction(num, den)
        poles = ctrl.poles(G)
        if np.all(np.real(poles) < -0.0001):
            print("This is a stable higher-order system")
            prompt = overall_instruction_PID
        else:
            print("This is an unstable higher-order system")
            prompt = overall_instruction_RHP_Pole_higherorder_PID

        new_problem = "Now consider the following design task:" + task_requirement + " Where the plant poles are:" + str(poles)

        problem_statement = prompt + new_problem + response_format_PID

        conversation_log = []

        while num_attempt <= self.max_attempts:
            print(f"Iteration {num_attempt} for system {system['id']} scenario {scenario}\n")
            # Call GPT4 to complete the prompt
            response = self.gpt4.complete(problem_statement)

            conversation_log.append({
                "Problem Statement": problem_statement,
                "Response": response
            })

            data = json.loads(response)

            # Extract the list of parameters
            parameters = data['parameter']
            omega_L = parameters[0]
            beta_b = parameters[1]
            beta_l = parameters[2]
            is_stable = check_stability_pid(omega_L, beta_b, beta_l, num, den)
            if is_stable:
                _, phase_margin, _, settlingtime, _, sse = loop_shaping(omega_L, beta_b, num, den)
                self.design_memory.add_design(
                    parameters={'omega_L': omega_L, 'beta_b': beta_b, 'beta_l': beta_l},
                    performance={'phase_margin': phase_margin, 'settling_time_min': settlingtime, 'settling_time_max': settlingtime,'steadystate_error': sse}
                )
                design = self.design_memory.get_latest_design()
                is_succ = True  # Assume success unless a metric fails
                for metric, specs in thresholds.items():
                    value = design['performance'].get(metric)
                    if value is not None:
                        if 'min' in specs and value < specs['min']:
                            is_succ = False  # Set success to False if any metric fails
                        elif 'max' in specs and value > specs['max']:
                            is_succ = False  # Set success to False if any metric fails
                if is_succ:
                    print("The current design satisfies the requirement.")
                    print(f"Phase Margin is {phase_margin}")
                    print(f"Settling Time is {settlingtime}")
                    print(f"Steady-state error is {sse}")

                    # Save success information and final design to the log
                    final_result = {
                        "is_succ": True,
                        "parameters": design['parameters'],
                        "performance": design['performance'],
                        "conversation_rounds": num_attempt
                    }

                    # Save success information and final design to the log
                    conversation_log.append({
                        "Design Success": True,
                        "Final Design Parameters": design['parameters'],
                        "Final Design Performance": design['performance']
                    })
                    break
                
                # abaltion 1: with or without feedback
                feedback = feedback_prompt(self.design_memory, thresholds)
                problem_statement = prompt + new_problem + "\n\n" + feedback + response_format_PID
            else:
                self.design_memory.add_design(
                    parameters={'omega_L': omega_L, 'beta_b': beta_b},
                    performance={'phase_margin': 'unstable', 'settling_time': 'unstable', 'steadystate_error': 'unstable'}
                )
                # Save unstable design information to the log
                conversation_log.append({
                    "Design Success": False,
                    "Failed Design Parameters": self.design_memory.get_latest_design()['parameters'],
                    "Failed Design Performance": self.design_memory.get_latest_design()['performance']
                })
                # Save unstable design information to the log
                feedback = feedback_prompt(self.design_memory, thresholds)
                problem_statement = prompt + new_problem + "\n\n" + feedback + response_format_PID
            num_attempt += 1

        if num_attempt == self.max_attempts + 1:
            design = self.design_memory.get_latest_design()
            # Save failure information and final design to the log
            final_result = {
                "is_succ": False,
                "parameters": design['parameters'],
                "performance": design['performance'],
                "conversation_rounds": num_attempt-1
            }
        
        # Save conversation log
        output_dir = os.path.join(self.base_output_dir, f"higher_ord")
        os.makedirs(output_dir, exist_ok=True)  
        output_file = os.path.join(output_dir, f"conversation_{system['id']}.json")
        with open(output_file, 'w') as f:
            json.dump(conversation_log, f, indent=4)


        # Extract the list of parameters
        parameters = data['parameter']
        
        # Parse and return the JSON response
        return final_result