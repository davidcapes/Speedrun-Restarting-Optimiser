# Plots Directory

This directory contains generated visualizations from the scripts throughout the codebase.

## Directory Structure

```
plots/
└── task_distributions/   # Individual task probability distributions
    └── task*_pdf.png     # Task * probability density function

```

## Plot Types

### Task Distribution Plots (`task_distributions/`)

#### **Features:**
- **PDF curve**: Probability density function
- **Legend**: Relevant distribution statistics (mean, variance, std, median, iqr, mode)
- **Color Coding**: Each plot uses a separate distinct color for its graph.

#### **Generation:**
Located in `src/preset_distributions/task_distribution_plotter.py`:
