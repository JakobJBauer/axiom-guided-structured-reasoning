"""
Comprehensive tests for formula operations: Not, And, Or, Xor, Equal, In
"""

import unittest
import sys
import os

# Add parent directory to path to import graph modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph.formulas import Not, And, Or, Xor, Equal, In


class TestNot(unittest.TestCase):
    """Tests for Not operation"""
    
    def test_simple_not(self):
        """Test Not with a simple key"""
        formula = Not('a')
        self.assertEqual(formula.get_required_keys(), ['a'])
        self.assertEqual(formula.get_valid_path_parents(), [['a']])
        
        # Test computation
        self.assertEqual(formula({'a': True}), False)
        self.assertEqual(formula({'a': False}), True)
        self.assertIsNone(formula({'a': None}))
        self.assertIsNone(formula({}))
    
    def test_not_with_formula(self):
        """Test Not with a nested formula"""
        formula = Not(And('a', 'b'))
        self.assertEqual(formula.get_required_keys(), ['a', 'b'])
        self.assertEqual(formula.get_valid_path_parents(), [['a', 'b']])
        
        # Test computation
        self.assertEqual(formula({'a': True, 'b': True}), False)
        self.assertEqual(formula({'a': True, 'b': False}), True)
        self.assertEqual(formula({'a': False, 'b': False}), True)


class TestAnd(unittest.TestCase):
    """Tests for And operation"""
    
    def test_simple_and(self):
        """Test And with simple keys"""
        formula = And('a', 'b')
        self.assertEqual(set(formula.get_required_keys()), {'a', 'b'})
        self.assertEqual(formula.get_valid_path_parents(), [['a', 'b']])
        
        # Test computation
        self.assertEqual(formula({'a': True, 'b': True}), True)
        self.assertEqual(formula({'a': True, 'b': False}), False)
        self.assertEqual(formula({'a': False, 'b': True}), False)
        self.assertEqual(formula({'a': False, 'b': False}), False)
        self.assertIsNone(formula({'a': True, 'b': None}))
        self.assertIsNone(formula({'a': None, 'b': True}))
    
    def test_and_with_formulas(self):
        """Test And with nested formulas"""
        formula = And(Not('a'), 'b')
        required = set(formula.get_required_keys())
        self.assertEqual(required, {'a', 'b'})
        self.assertEqual(formula.get_valid_path_parents(), [['a', 'b']])
        
        # Test computation
        self.assertEqual(formula({'a': False, 'b': True}), True)
        self.assertEqual(formula({'a': True, 'b': True}), False)
    
    def test_and_empty(self):
        """Test And with no arguments"""
        formula = And()
        self.assertEqual(formula.get_required_keys(), [])
        self.assertEqual(formula({'a': True}), True)  # Empty And is True


class TestOr(unittest.TestCase):
    """Tests for Or operation"""
    
    def test_simple_or(self):
        """Test Or with simple keys"""
        formula = Or('a', 'b')
        self.assertEqual(set(formula.get_required_keys()), {'a', 'b'})
        paths = formula.get_valid_path_parents()
        self.assertIn(['a'], paths)
        self.assertIn(['b'], paths)
        self.assertIn(['a', 'b'], paths)
        self.assertEqual(len(paths), 3)  # Power set: 2^2 - 1 = 3
        
        # Test computation
        self.assertEqual(formula({'a': True, 'b': False}), True)
        self.assertEqual(formula({'a': False, 'b': True}), True)
        self.assertEqual(formula({'a': True, 'b': True}), True)
        self.assertEqual(formula({'a': False, 'b': False}), False)
    
    def test_or_with_formulas(self):
        """Test Or with nested formulas"""
        formula = Or(And('a', 'b'), And('c', 'd'))
        required = set(formula.get_required_keys())
        self.assertEqual(required, {'a', 'b', 'c', 'd'})
        paths = formula.get_valid_path_parents()
        # Should have power set: 2^4 - 1 = 15 paths
        self.assertEqual(len(paths), 15)
        self.assertIn(['a'], paths)
        self.assertIn(['a', 'b', 'c', 'd'], paths)
    
    def test_or_empty(self):
        """Test Or with no arguments"""
        formula = Or()
        self.assertEqual(formula.get_required_keys(), [])
        self.assertEqual(formula({'a': True}), False)  # Empty Or is False


class TestXor(unittest.TestCase):
    """Tests for Xor operation"""
    
    def test_simple_xor(self):
        """Test Xor with simple keys"""
        formula = Xor('a', 'b')
        self.assertEqual(set(formula.get_required_keys()), {'a', 'b'})
        paths = formula.get_valid_path_parents()
        # Xor should only have single-key paths
        self.assertEqual(paths, [['a'], ['b']])
        
        # Test computation
        self.assertEqual(formula({'a': True, 'b': False}), True)
        self.assertEqual(formula({'a': False, 'b': True}), True)
        self.assertEqual(formula({'a': True, 'b': True}), False)  # Both true = False
        self.assertEqual(formula({'a': False, 'b': False}), False)
    
    def test_xor_three_keys(self):
        """Test Xor with three keys"""
        formula = Xor('a', 'b', 'c')
        self.assertEqual(set(formula.get_required_keys()), {'a', 'b', 'c'})
        paths = formula.get_valid_path_parents()
        # Should only have single-key paths
        self.assertEqual(len(paths), 3)
        self.assertIn(['a'], paths)
        self.assertIn(['b'], paths)
        self.assertIn(['c'], paths)
        self.assertNotIn(['a', 'b'], paths)  # No combinations
        
        # Test computation (XOR: exactly one True = True)
        self.assertEqual(formula({'a': True, 'b': False, 'c': False}), True)
        self.assertEqual(formula({'a': True, 'b': True, 'c': False}), False)
        self.assertEqual(formula({'a': True, 'b': True, 'c': True}), False)
    
    def test_xor_with_formulas(self):
        """Test Xor with nested formulas"""
        formula = Xor(Not('a'), 'b')
        required = set(formula.get_required_keys())
        self.assertEqual(required, {'a', 'b'})
        paths = formula.get_valid_path_parents()
        self.assertEqual(paths, [['a'], ['b']])


class TestEqual(unittest.TestCase):
    """Tests for Equal operation"""
    
    def test_equal_string(self):
        """Test Equal with string values"""
        formula = Equal('category', 'fiction')
        self.assertEqual(formula.get_required_keys(), ['category'])
        self.assertEqual(formula.get_valid_path_parents(), [['category']])
        
        # Test computation
        self.assertEqual(formula({'category': 'fiction'}), True)
        self.assertEqual(formula({'category': 'non-fiction'}), False)
        self.assertIsNone(formula({'category': None}))
        self.assertIsNone(formula({}))
    
    def test_equal_boolean(self):
        """Test Equal with boolean values"""
        formula = Equal('flag', True)
        self.assertEqual(formula({'flag': True}), True)
        self.assertEqual(formula({'flag': False}), False)
    
    def test_equal_number(self):
        """Test Equal with numeric values"""
        formula = Equal('count', 5)
        self.assertEqual(formula({'count': 5}), True)
        self.assertEqual(formula({'count': 10}), False)


class TestIn(unittest.TestCase):
    """Tests for In operation"""
    
    def test_in_list(self):
        """Test In with a list"""
        formula = In('genre', ['sci-fi', 'fantasy', 'horror'])
        self.assertEqual(formula.get_required_keys(), ['genre'])
        self.assertEqual(formula.get_valid_path_parents(), [['genre']])
        
        # Test computation
        self.assertEqual(formula({'genre': 'sci-fi'}), True)
        self.assertEqual(formula({'genre': 'fantasy'}), True)
        self.assertEqual(formula({'genre': 'horror'}), True)
        self.assertEqual(formula({'genre': 'romance'}), False)
        self.assertIsNone(formula({'genre': None}))
        self.assertIsNone(formula({}))
    
    def test_in_single_value(self):
        """Test In with a single value (treated as list)"""
        formula = In('status', 'active')
        self.assertEqual(formula({'status': 'active'}), True)
        self.assertEqual(formula({'status': 'inactive'}), False)
    
    def test_in_numbers(self):
        """Test In with numeric values"""
        formula = In('score', [1, 2, 3, 4, 5])
        self.assertEqual(formula({'score': 3}), True)
        self.assertEqual(formula({'score': 10}), False)


class TestFormulaChaining(unittest.TestCase):
    """Tests for chaining formulas together"""
    
    def test_complex_chaining(self):
        """Test complex nested formula chains"""
        formula = And(Or('a', 'b'), Not('c'))
        required = set(formula.get_required_keys())
        self.assertEqual(required, {'a', 'b', 'c'})
        
        # Test computation
        self.assertEqual(formula({'a': True, 'b': False, 'c': False}), True)
        self.assertEqual(formula({'a': False, 'b': True, 'c': False}), True)
        self.assertEqual(formula({'a': True, 'b': False, 'c': True}), False)
    
    def test_deep_nesting(self):
        """Test deeply nested formulas"""
        formula = And(Or(And('a', 'b'), And('c', 'd')), Not('e'))
        required = set(formula.get_required_keys())
        self.assertEqual(required, {'a', 'b', 'c', 'd', 'e'})
        
        # Test computation
        self.assertEqual(formula({'a': True, 'b': True, 'c': False, 'd': False, 'e': False}), True)
        self.assertEqual(formula({'a': True, 'b': True, 'c': False, 'd': False, 'e': True}), False)
    
    def test_mixed_operations(self):
        """Test mixing different operations"""
        formula = Or(And('a', 'b'), Xor('c', 'd'))
        required = set(formula.get_required_keys())
        self.assertEqual(required, {'a', 'b', 'c', 'd'})
        
        paths = formula.get_valid_path_parents()
        # Or should generate power set
        self.assertGreater(len(paths), 4)  # Should have many combinations


class TestFormulaDirectExecution(unittest.TestCase):
    """Tests for direct execution (callable) of formulas"""
    
    def test_direct_call(self):
        """Test that formulas are directly callable"""
        formula = And('a', 'b')
        # Should be able to call directly, not just .compute()
        self.assertEqual(formula({'a': True, 'b': True}), True)
        self.assertEqual(formula({'a': True, 'b': False}), False)
    
    def test_all_operations_callable(self):
        """Test all operations are callable"""
        self.assertEqual(Not('a')({'a': True}), False)
        self.assertEqual(Or('a', 'b')({'a': True, 'b': False}), True)
        self.assertEqual(Xor('a', 'b')({'a': True, 'b': False}), True)
        self.assertEqual(Equal('x', 5)({'x': 5}), True)
        self.assertEqual(In('y', [1, 2, 3])({'y': 2}), True)


if __name__ == '__main__':
    unittest.main()

