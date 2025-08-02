# Test Suite

Comprehensive test suite for the Speedrun Optimization Framework.

## Running Tests

### Running all tests
```bash
python -m unittest discover -s Perfect/tests -p "test_*.py"  
```
### Running specific tests
```bash
python -m unittest discover -s Perfect/tests -p "test_example_case.py"  
```

## Writing Tests

### Test Naming Convention
- Test files: `test_*.py`

### Test Structure

```python
import unittest

from <module> import <function1>, <function2>

class TestProjectStructure(unittest.TestCase):
    """Example unit tests for functions in <module>."""

    def test_function1(self):
        # TODO: Replace with real assertions
        pass

    def test_function2(self):
        # TODO: Replace with real assertions
        pass

if __name__ == "__main__":
    unittest.main()
```
