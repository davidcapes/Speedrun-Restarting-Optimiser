# Test Suite

Test suite for the Speedrun Optimisation Framework.

## Running Tests

### Running all tests
```bash
python -m unittest discover -s tests -p "test_*.py"  
```
### Running specific tests
```bash
python -m unittest discover -s tests -p "test_example_case.py"  
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
        ...

    def test_function2(self):
        ...

if __name__ == "__main__":
    unittest.main()
```
