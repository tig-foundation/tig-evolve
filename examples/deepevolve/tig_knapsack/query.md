# Knapsack Challenge Information

Evolve a Rust algorithm that solves the Quadratic Knapsack problem


## Task Description


The Quadratic Knapsack challenge:  Given a set of items, each with a weight and value, and pairwise interaction values between items, select a subset of items that maximizes the total value while respecting a maximum weight constraint. The total value includes both individual item values and the sum of interaction values between all pairs of selected items.


## Input Format


Your algorithm receives a `Challenge` struct with the following fields:


- `seed: [u8; 32]` - Random seed for reproducible instance generation
- `num_items: usize` - Number of items (n)
- `weights: Vec<u32>` - Weight of each item (weights[i] ∈ [1, 50])
- `values: Vec<u32>` - Individual value of each item (values[i] ∈ [0, 100] with density probability, 0 otherwise)
- `interaction_values: Vec<Vec<i32>>` - Symmetric matrix of pairwise interaction values (interaction_values[i][j] ∈ [0, 100]
- `max_weight: u32` - Maximum allowed total weight (set to sum of all weights / 2)


## Output Format


Your algorithm must return a `Solution` struct containing:


- `items: Vec<usize>` - Indices of selected items (must be unique, 0-based indexing)


## Constraints


- Total weight of selected items ≤ max_weight
- No duplicate items in the solution


## Value Calculation


The total value is calculated as:


```
total_value = Σ(values[i] for i in selected_items) + Σ(interaction_values[i][j] for i,j in selected_items pairs)
```




## Scoring Metric


Solutions are scored using a quality metric that compares against a baseline greedy algorithm:


Takes your solution's total_value, compares it to the value of the greedy baseline solution's (V_greedy), outputting the 


Quality = (total_value - V_greedy) / V_greedy


Higher quality scores are better




## Algorithm Implementation


To implement a solution:


1. Use the provided `solve_challenge` function signature to implement your algorithm
2. Periodically save intermediate solutions using `save_solution(&Solution)` function
3. Return `Ok(())` on success or `Err(anyhow!("error message"))` on failure


## Tips for Implementation


1. **Deterministic Behavior**: Use the challenge seed for any random number generation to ensure reproducible results
2. **Incremental Saves**: Save intermediate solutions during long-running searches
3. **Data Structures**: Use efficient data structures.
4. **Imports**: If you want to add new imports they must be from the standard library std.
5. **No Training**: Do not implement an algorithm that requires training, we do not want machine learning models.
6. **Using Hyperparameters**: I)  Define the struct: Add fields to the Hyperparameters struct in the template: 


`#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub param1: usize,
    pub param2: f64,
    // Add more fields as needed
}`




ii) Parse in solve_challenge: Uncomment and modify the hyperparameter parsing code: 


`let hyperparameters = match hyperparameters {
    Some(hyperparameters) => {
        serde_json::from_value::<Hyperparameters>(Value::Object(hyperparameters.clone()))
            .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
    }
    None => Hyperparameters {
        param1: default_value,
        param2: default_value,
        // Set defaults for all fields
    },
};`


ii) Use the parsed struct: Access hyperparameters like hyperparameters.param1 in your algorithm logic.








