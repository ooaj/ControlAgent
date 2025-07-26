from gpt4 import GPT4
import json
from DesignMemory import design_memory
from util import loop_shaping_pid, feedback_prompt, check_stability_pid
import os
from instruction import overall_instruction_PID, response_format_PID



class second_ord_stable_Design:

    def __init__(self, engine='gemini-2.0-flash-exp', base_output_dir = "./outputs/test", temperature=0.0, max_tokens=1024):
        self.gpt4 = GPT4(engine=engine, temperature=temperature, max_tokens=max_tokens)
        self.max_attempts = 10
        self.design_memory = design_memory()
        self.base_output_dir = "./outputs"

    def handle_task(self, system, thresholds, task_requirement, scenario):
        num_attempt = 1
        # Implement the task handling process
        print(f"Handling controller design for system {system['id']} in scenario {scenario}")

        # Construct the design prompt
        prompt = overall_instruction_PID
        new_problem = "Now consider the following design task:" + task_requirement

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

            # parameters for plant transfer function
            num = [system['num'][0]]
            den = system['den']

            # Extract the list of parameters
            parameters = data['parameter']
            omega_L = parameters[0]
            beta_b = parameters[1]
            beta_l = parameters[2]
            is_stable = check_stability_pid(omega_L, beta_b, beta_l, num, den)
            if is_stable:
                _, phase_margin, _, settlingtime, _, sse = loop_shaping_pid(omega_L, beta_b, beta_l, num, den)
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
        output_dir = os.path.join(self.base_output_dir, f"second_ord_stable_{scenario}")
        os.makedirs(output_dir, exist_ok=True)  
        output_file = os.path.join(output_dir, f"conversation_{system['id']}.json")
        with open(output_file, 'w') as f:
            json.dump(conversation_log, f, indent=4)


        # Extract the list of parameters
        parameters = data['parameter']
        
        # Parse and return the JSON response
        return final_result