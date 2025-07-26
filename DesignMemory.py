


class design_memory:
    def __init__(self):
        self.buffer = []
        self.memory_log = []
        self.feedback_log = []

    def add_design(self, parameters, performance):
        """
        Adds a new design and its performance metrics to the memory buffer.
        :param parameters: Dictionary containing design parameters.
        :param performance: Dictionary containing performance metrics like gain margin, phase margin, etc.
        """
        entry = {
            'parameters': parameters,
            'performance': performance
        }
        self.buffer.append(entry)

    def get_all_designs(self):
        """
        Returns all designs and their performances.
        """
        return self.buffer

    def get_latest_design(self):
        """
        Returns the latest design and its performance.
        """
        if self.buffer:
            return self.buffer[-1]
        return None
    
    def add_memory(self, message):
        """Add a memory entry."""
        self.memory_log.append(message)
    
    def get_memory(self):
        """Get all memory entries as a string."""
        return "\n".join(self.memory_log)
    
    def add_feedback(self, feedback):
        """Add feedback entry.""" 
        self.feedback_log.append(feedback)
    
    def get_feedback(self):
        """Get all feedback entries as a string."""
        return "\n".join(self.feedback_log)