import itertools
from abc import ABC, abstractmethod


class Formula(ABC):
    """
    Base class for all formula operations.
    
    Formulas compute values from incoming node values and can auto-infer
    their valid_path_parents based on their structure.
    """

    @classmethod
    @abstractmethod
    def min_parameter_count(cls):
        """
        Return the minimum number of parameters required for this formula.
            
        Returns:
            Minimum number of parameters required
        """
        pass
    
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

    def rename_node_ids(self, node_map):
        """
        In-place rename of any node-id string references inside this Formula.

        This is used when graphs are renamed (node IDs change) and we need all
        formulas to keep referencing the correct nodes.
        """

        def transform(obj):
            if isinstance(obj, str):
                return node_map.get(obj, obj)
            if isinstance(obj, Formula):
                obj.rename_node_ids(node_map)
                return obj
            if isinstance(obj, list):
                return [transform(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(transform(x) for x in obj)
            if isinstance(obj, dict):
                return {transform(k): transform(v) for k, v in obj.items()}
            if isinstance(obj, set):
                return {transform(x) for x in obj}
                
            return obj

        # Rewrite any string references stored on this object
        for k, v in list(self.__dict__.items()):
            self.__dict__[k] = transform(v)

