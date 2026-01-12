import itertools
from .formula import Formula


class Not(Formula):
    def __init__(self, key_or_formula):
        """
        Args:
            key_or_formula: Node ID (string) or Formula object to negate
        """
        self.key_or_formula = key_or_formula
    
    def _get_value(self, incoming_values):
        if isinstance(self.key_or_formula, Formula):
            return self.key_or_formula(incoming_values)
        else:
            return incoming_values.get(self.key_or_formula)
    
    def compute(self, incoming_values):
        value = self._get_value(incoming_values)
        if value is None: return None
        return not bool(value)
    
    def get_required_keys(self):
        if isinstance(self.key_or_formula, Formula):
            return self.key_or_formula.get_required_keys()
        else:
            return [self.key_or_formula]

    def get_valid_path_parents(self):
        return [self.get_required_keys()]
    
    def __repr__(self):
        return f"Not({self.key_or_formula!r})"


class And(Formula):    
    def __init__(self, *keys_or_formulas):
        """
        Args:
            *keys_or_formulas: Variable number of node IDs (strings) or Formula objects to AND together
        """
        self.keys_or_formulas = list(keys_or_formulas)
    
    def _get_value(self, key_or_formula, incoming_values):
        if isinstance(key_or_formula, Formula):
            return key_or_formula(incoming_values)
        else:
            return incoming_values.get(key_or_formula)
    
    def compute(self, incoming_values):
        if not self.keys_or_formulas:
            return True
        
        values = [self._get_value(kf, incoming_values) for kf in self.keys_or_formulas]
        if any(v is None for v in values): return None
        
        return all(bool(v) for v in values)
    
    def get_required_keys(self):
        keys = []
        for kf in self.keys_or_formulas:
            if isinstance(kf, Formula):
                keys.extend(kf.get_required_keys())
            else:
                keys.append(kf)
        return keys

    def get_valid_path_parents(self):
        return [self.get_required_keys()]
    
    def __repr__(self):
        return f"And({', '.join(repr(kf) for kf in self.keys_or_formulas)})"


class Or(Formula):    
    def __init__(self, *keys_or_formulas):
        """
        Args:
            *keys_or_formulas: Variable number of node IDs (strings) or Formula objects to OR together
        """
        self.keys_or_formulas = list(keys_or_formulas)
    
    def _get_value(self, key_or_formula, incoming_values):
        if isinstance(key_or_formula, Formula):
            return key_or_formula(incoming_values)
        else:
            return incoming_values.get(key_or_formula)
    
    def compute(self, incoming_values):
        if not self.keys_or_formulas:
            return False
        
        values = [self._get_value(kf, incoming_values) for kf in self.keys_or_formulas]
        if any(v is None for v in values):
            if all(v is None for v in values): return None
            if any(v is True for v in values): return True
            return None
        
        return any(bool(v) for v in values)
    
    def get_required_keys(self):
        keys = []
        for kf in self.keys_or_formulas:
            if isinstance(kf, Formula):
                keys.extend(kf.get_required_keys())
            else:
                keys.append(kf)
        return keys
    
    def get_valid_path_parents(self):
        # return the power set of the required keys
        keys = self.get_required_keys()
        return [list(combo) for r in range(1, len(keys) + 1) for combo in itertools.combinations(keys, r)]
    
    def __repr__(self):
        return f"Or({', '.join(repr(kf) for kf in self.keys_or_formulas)})"


class Xor(Formula):    
    def __init__(self, *keys_or_formulas):
        """
        Args:
            *keys_or_formulas: Variable number of node IDs (strings) or Formula objects to XOR together
        """
        self.keys_or_formulas = list(keys_or_formulas)
    
    def _get_value(self, key_or_formula, incoming_values):
        if isinstance(key_or_formula, Formula):
            return key_or_formula(incoming_values)
        else:
            return incoming_values.get(key_or_formula)
    
    def compute(self, incoming_values):
        if not self.keys_or_formulas:
            return False
        
        values = [self._get_value(kf, incoming_values) for kf in self.keys_or_formulas]
        if any(v is None for v in values): return None
        
        bool_values = [bool(v) for v in values]
        return sum(bool_values) == 1
    
    def get_required_keys(self):
        keys = []
        for kf in self.keys_or_formulas:
            if isinstance(kf, Formula):
                keys.extend(kf.get_required_keys())
            else:
                keys.append(kf)
        return keys
    
    def get_valid_path_parents(self):
        """
        XOR valid paths: only single-key paths (exactly one input must be true).
        """
        keys = self.get_required_keys()
        if not keys: return None
        return [[key] for key in keys]
    
    def __repr__(self):
        return f"Xor({', '.join(repr(kf) for kf in self.keys_or_formulas)})"


class Equal(Formula):
    def __init__(self, key, value):
        """
        Args:
            key: Node ID to check
            value: Value to compare against
        """
        self.key = key
        self.value = value
    
    def compute(self, incoming_values):
        node_value = incoming_values.get(self.key)
        if node_value is None:
            return None
        return node_value == self.value
    
    def get_required_keys(self):
        return [self.key]
    
    def get_valid_path_parents(self):
        return [self.get_required_keys()]
    
    def __repr__(self):
        return f"Equal({self.key!r}, {self.value!r})"


class In(Formula):
    def __init__(self, key, values):
        """
        Args:
            key: Node ID to check
            values: List of values to check membership against
        """
        self.key = key
        self.values = list(values) if not isinstance(values, str) else [values]
    
    def compute(self, incoming_values):
        node_value = incoming_values.get(self.key)
        if node_value is None: return None
        return node_value in self.values
    
    def get_required_keys(self):
        return [self.key]
    
    def get_valid_path_parents(self):
        return [self.get_required_keys()]
    
    def __repr__(self):
        return f"In({self.key!r}, {self.values!r})"
