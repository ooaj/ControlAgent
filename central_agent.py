import json
from gpt4 import GPT4
from subagents.first_ord_stable import first_ord_stable_Design
from subagents.first_ord_unstable import first_ord_unstable_Design
from subagents.second_ord_stable import second_ord_stable_Design
from subagents.second_ord_unstable import second_ord_unstable_Design
from subagents.first_ord_w_delay import first_ord_w_delay_Design
from subagents.higher_ord import higher_ord_Design
from subagents.robust_pole_placement import RobustPolePlace_Design

class CentralAgentLLM:

    def __init__(self, engine='gemini-2.0-flash-exp', temperature=0.0, max_tokens=1024):
        # Define the available sub-agents
        self.sub_agents = {
            1: first_ord_stable_Design(),
            2: first_ord_unstable_Design(),
            3: second_ord_stable_Design(),
            4: second_ord_unstable_Design(),
            5: first_ord_w_delay_Design(),
            6: higher_ord_Design(),
            7: RobustPolePlace_Design()
        }
        self.gpt4 = GPT4(engine=engine, temperature=temperature, max_tokens=max_tokens)


    def assign_task(self, system, input_prompt, thresholds, scenario):
        # Check for parameter uncertainty first - if present, use robust agent directly
        if self.has_parameter_uncertainty(system):
            print("Parameter uncertainty detected - routing to robust pole placement agent")
            sub_agent = self.sub_agents[7]  # Robust pole placement agent
            print(f"Activating sub-agent: 7 (Robust Pole Placement)")
            
            # Create task requirement for robust agent
            uncertainty_info = self.format_uncertainty_info(system)
            task_requirement = f"Design robust controller for system with parameter uncertainty. {uncertainty_info}"
            print(task_requirement)
            
            response = sub_agent.handle_task(system, thresholds, task_requirement, scenario)
        else:
            # Use standard LLM-based agent selection for systems without uncertainty
            response = self.gpt4.complete(input_prompt)
            parsed_response = json.loads(response)
            agent_number = int(parsed_response.get("Agent Number"))

            if agent_number in self.sub_agents and agent_number != 7:  # Exclude robust agent for non-uncertain systems
                sub_agent = self.sub_agents[agent_number]
                print(f"Activating sub-agent: {agent_number}")
                print(parsed_response['Task Requirement'])
                response = sub_agent.handle_task(system, thresholds, parsed_response['Task Requirement'], scenario)
            else:
                print("No suitable sub-agent found for this task.")
                response = {'is_succ': False, 'error': 'No suitable agent found'}

        return response
    
    def has_parameter_uncertainty(self, system):
        """Check if the system has parameter uncertainty information."""
        return ('num_uncertainty' in system and system['num_uncertainty'] is not None) or \
               ('den_uncertainty' in system and system['den_uncertainty'] is not None)
    
    def format_uncertainty_info(self, system):
        """Format uncertainty information for the robust agent."""
        info_parts = []
        
        if 'num_uncertainty' in system and system['num_uncertainty']:
            gain_uncertainty = system['num_uncertainty'][0] * 100
            info_parts.append(f"±{gain_uncertainty:.1f}% gain uncertainty")
        
        if 'den_uncertainty' in system and system['den_uncertainty'] and len(system['den_uncertainty']) > 1:
            if system['den_uncertainty'][1] > 0:
                pole_uncertainty = system['den_uncertainty'][1] * 100
                info_parts.append(f"±{pole_uncertainty:.1f}% pole uncertainty")
        
        if info_parts:
            return "System has " + " and ".join(info_parts) + "."
        else:
            return "System has unspecified parameter uncertainty."

