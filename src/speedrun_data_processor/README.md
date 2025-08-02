# Speedrun Data Processing

This module provides tools for processing real speedrun data, including the construction of task distribution estimates.  

## **File Descriptions**

### **1. Data Simulation**
#### `create_from_example`
Generates synthetic speedrun data using predefined distributions and game structure. 

For example:
```json
{
  "1": {
    "Completed": [<float>, ...],
    "Restarted": [<float>, ...],
    "Reachable": ["2", ...],
    "Start": true,
    "End": false
  },
  ...
}
```

### **2. Graph Processing**
#### `scan_json`
Reads the player or simulated JSON file and:
- Topologically sorts the game graph.
- Builds an **adjacency matrix** representation of the DAG.
- Validates that the graph is **acyclic**.

**Returns:**
- `graph_array` → NumPy adjacency matrix with an extra column for terminal states.
- `name_indices` → Mapping from task names to indices.

---

## **Example Usage**
```python
from src.speedrun_data_processor.graph_processor import create_from_example, scan_json
from src.speedrun_data_processor.distribution_estimator import kde_method

# Generate synthetic speedrun data
data = create_from_example(simulation_count=200, seed=123)

# Process the generated data into a DAG
graph_array, name_indices = scan_json()

print(graph_array)
print(name_indices)

# Create PDF table
D1 = data["1"]["Completed"]
D2 = data["1"]["Restarted"]
pdf_table = kde_method(D1, D2, 50)
print(pdf_table)
```