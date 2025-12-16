# Knapsack Challenge Information

## Overview

The Knapsack challenge is a variant of the classic knapsack optimization problem with an additional complexity: interactive values between items. 

## Task Description

Given a set of items, each with a weight and value, plus pairwise interaction values between items, select a subset of items that maximizes the total value while respecting a maximum weight constraint. The total value includes both individual item values and the sum of interaction values between all pairs of selected items.

## Input Format

Your algorithm receives a `Challenge` struct with the following fields:

- `seed: [u8; 32]` - Random seed for reproducible instance generation
- `num_items: usize` - Number of items (n)
- `weights: Vec<u32>` - Weight of each item (weights[i] ∈ [1, 50])
- `values: Vec<u32>` - Individual value of each item (values[i] ∈ [0, 100] with density probability, 0 otherwise)
- `interaction_values: Vec<Vec<i32>>` - Symmetric matrix of pairwise interaction values (interaction_values[i][j] ∈ [0, 100] with density probability, 0 otherwise, diagonal is always 0)
- `max_weight: u32` - Maximum allowed total weight (set to sum of all weights / 2)

## Output Format

Your algorithm must return a `Solution` struct containing:

- `items: Vec<usize>` - Indices of selected items (must be unique, sorted for consistency, 0-based indexing)

## Constraints

- Total weight of selected items ≤ max_weight
- No duplicate items in the solution

## Value Calculation

The total value is calculated as:

```
total_value = Σ(values[i] for i in selected_items) + Σ(interaction_values[i][j] for i,j in selected_items pairs)
```

Where the final result is `max(0, total_value)`.

## Scoring Metric

Solutions are scored using a quality metric that compares against a baseline greedy algorithm:

1. Calculate your solution's total value (V_your)
2. Calculate the greedy baseline solution's total value (V_greedy)
3. Quality = (V_your - V_greedy) / V_greedy
4. Quality is clamped to [-10.0, 10.0] and scaled by 1,000,000 for precision
5. Higher quality scores are better

The greedy baseline uses a simple heuristic: sort items by (value + sum of interaction values) / weight ratio, then greedily select items until the weight limit is reached.

## Challenge Tracks

The challenge supports different difficulty tracks that vary two parameters:

### Track Parameters
- `n_items`: Number of items in the knapsack instance
- `density`: Probability (as percentage) that values and interaction values are non-zero

### Available Tracks
Tracks are identified by strings in the format `n_items=X,density=Y` where:
- X is the number of items
- Y is the density percentage (0-100)

Common tracks include various combinations of:
- Small instances: n_items around 50-100
- Medium instances: n_items around 200-500
- Large instances: n_items around 1000+
- Low density: density = 10-30 (sparse interactions)
- Medium density: density = 40-60 (moderate interactions)
- High density: density = 70-90 (dense interactions)

## Algorithm Implementation

To implement a solution:

1. Use the provided `solve_challenge` function signature
2. Implement your optimization algorithm within the function
3. Periodically save intermediate solutions using `save_solution(&Solution { items: your_selected_items })`
4. Return `Ok(())` on success or `Err(anyhow!("error message"))` on failure

### Example Pseudocode

```rust
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // Your algorithm implementation here
    let mut selected_items = Vec::new();

    // Implement your selection logic
    // For example: greedy selection
    for i in 0..challenge.num_items {
        if would_fit_in_knapsack(i, &selected_items, challenge) {
            selected_items.push(i);
        }
    }

    // Sort for consistency
    selected_items.sort_unstable();

    // Save the solution
    save_solution(&Solution { items: selected_items })?;

    Ok(())
}

fn would_fit_in_knapsack(item: usize, selected: &[usize], challenge: &Challenge) -> bool {
    let current_weight: u32 = selected.iter().map(|&i| challenge.weights[i]).sum();
    current_weight + challenge.weights[item] <= challenge.max_weight
}
```

## Tips for Implementation

1. **Deterministic Behavior**: Use the challenge seed for any random number generation to ensure reproducible results
2. **Constraint Validation**: Always ensure weight constraints are respected
3. **Performance**: Consider the interaction matrix when making selection decisions
4. **Incremental Saves**: Save intermediate solutions during long-running optimizations
5. **Data Structures**: Use efficient data structures for large instances (n_items can be 1000+)

## Evaluation Details

- Weight violations result in invalid solutions
- Negative total values are clamped to 0
- Quality scores are compared across all submissions for ranking
- Runtime and solution quality both contribute to final scoring
