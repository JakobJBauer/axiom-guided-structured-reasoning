import itertools
from abc import ABC, abstractmethod


class Formula(ABC):
    """
    Base class for all formula operations.
    
    Formulas compute values from incoming node values and can auto-infer
    their valid_path_parents based on their structure.
    """
    
    @abstractmethod
    def compute(self, incoming_values):
        """
        Compute the formula value given incoming node values.
        
        Args:
            incoming_values: Dict mapping node IDs to their values
            
        Returns:
            The computed value (typically boolean, but can be any type)
        """
        pass
    
    @abstractmethod
    def get_required_keys(self):
        """
        Get the list of node IDs (keys) required for this formula.
        
        Returns:
            List of node ID strings
        """
        pass
    
    @abstractmethod
    def get_valid_path_parents(self):
        """
        Auto-infer valid_path_parents from the formula structure. Must be overridden when the formula is not 
        
        Returns:
            List of lists, where each inner list is a valid path of parent node IDs
        """
        pass
    
    def __call__(self, incoming_values):
        return self.compute(incoming_values)

