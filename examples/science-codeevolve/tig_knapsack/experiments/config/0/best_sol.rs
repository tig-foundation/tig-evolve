// DO NOT CHANGE THESE IMPORTS
use crate::challenge::{Challenge, Solution};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use rand::{rngs::SmallRng, Rng, SeedableRng};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // EVOLVE-BLOCK-START    
    // A more sophisticated approach: a greedy start followed by a best-improvement local search.
    // This directly handles the quadratic interaction values.
    let num_items = challenge.values.len();

    // --- 1. Initial Solution: Greedy Construction ---
    // Sort items by value/weight ratio. This provides a good starting point.
    let mut items: Vec<usize> = (0..num_items).collect();
    items.sort_by(|&a, &b| {
        let ratio_a = challenge.values[a] as f64 / challenge.weights[a] as f64;
        let ratio_b = challenge.values[b] as f64 / challenge.weights[b] as f64;
        ratio_b.partial_cmp(&ratio_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut in_knapsack = vec![false; num_items];
    let mut current_weight: u64 = 0;
    let max_weight = challenge.max_weight as u64;

    for &item_idx in &items {
        if current_weight + challenge.weights[item_idx] as u64 <= max_weight {
            in_knapsack[item_idx] = true;
            current_weight += challenge.weights[item_idx] as u64;
        }
    }

    // --- 2. Initialize State for Local Search ---
    let mut current_value: i64 = 0;
    // interaction_sum[i] = sum_{j in knapsack} p_ij for any item i
    let mut interaction_sum = vec![0i64; num_items];

    // Calculate initial value and interaction_sum based on the greedy solution
    {
        let selected_indices: Vec<usize> = (0..num_items).filter(|&i| in_knapsack[i]).collect();

        // Calculate initial interaction_sum for ALL items
        for i in 0..num_items {
            for &j in &selected_indices {
                interaction_sum[i] += challenge.interaction_values[i][j] as i64;
            }
        }
        
        // Calculate initial total value using the precomputed sums
        let mut total_interaction_value = 0i64;
        for &i in &selected_indices {
            current_value += challenge.values[i] as i64;
            total_interaction_value += interaction_sum[i];
        }
        // Each pair (i, j) is counted twice (p_ij and p_ji), so divide by 2.
        current_value += total_interaction_value / 2;
    }

    // --- 3. Local Search (Hill Climbing with Best Improvement) ---
    loop {
        let mut best_delta: i64 = 0;
        // Store move as (item_to_remove, item_to_add).
        // Use num_items as a sentinel for no-item (for add/remove moves).
        let mut best_move: Option<(usize, usize)> = None;

        let selected_indices: Vec<usize> = (0..num_items).filter(|&i| in_knapsack[i]).collect();
        let unselected_indices: Vec<usize> = (0..num_items).filter(|&i| !in_knapsack[i]).collect();

        // A) Evaluate 1-Swap moves
        for &item_out in &selected_indices {
            for &item_in in &unselected_indices {
                if current_weight - challenge.weights[item_out] as u64 + challenge.weights[item_in] as u64 <= max_weight {
                    let delta = (challenge.values[item_in] as i64 - challenge.values[item_out] as i64)
                              + (interaction_sum[item_in] - challenge.interaction_values[item_in][item_out] as i64)
                              - interaction_sum[item_out];
                    if delta > best_delta {
                        best_delta = delta;
                        best_move = Some((item_out, item_in));
                    }
                }
            }
        }

        // B) Evaluate 1-Add moves
        for &item_in in &unselected_indices {
            if current_weight + challenge.weights[item_in] as u64 <= max_weight {
                let delta = challenge.values[item_in] as i64 + interaction_sum[item_in];
                if delta > best_delta {
                    best_delta = delta;
                    best_move = Some((num_items, item_in)); // Sentinel for add
                }
            }
        }

        // C) Evaluate 1-Remove moves
        for &item_out in &selected_indices {
            let delta = -(challenge.values[item_out] as i64) - interaction_sum[item_out];
            if delta > best_delta {
                best_delta = delta;
                best_move = Some((item_out, num_items)); // Sentinel for remove
            }
        }

        if let Some((item_out, item_in)) = best_move {
            // Apply the best move found
            current_value += best_delta;

            match (item_out < num_items, item_in < num_items) {
                (true, true) => { // Swap move
                    current_weight = current_weight - challenge.weights[item_out] as u64 + challenge.weights[item_in] as u64;
                    in_knapsack[item_out] = false;
                    in_knapsack[item_in] = true;
                    for i in 0..num_items {
                        interaction_sum[i] = interaction_sum[i] - challenge.interaction_values[i][item_out] as i64 + challenge.interaction_values[i][item_in] as i64;
                    }
                },
                (false, true) => { // Add move
                    current_weight += challenge.weights[item_in] as u64;
                    in_knapsack[item_in] = true;
                    for i in 0..num_items {
                        interaction_sum[i] += challenge.interaction_values[i][item_in] as i64;
                    }
                },
                (true, false) => { // Remove move
                    current_weight -= challenge.weights[item_out] as u64;
                    in_knapsack[item_out] = false;
                    for i in 0..num_items {
                        interaction_sum[i] -= challenge.interaction_values[i][item_out] as i64;
                    }
                },
                _ => {} // Should not happen
            }
        } else {
            // No improving move found, local optimum reached
            break;
        }
    }

    // --- 4. Finalize Solution ---
    let selected: Vec<usize> = (0..num_items)
        .filter(|&i| in_knapsack[i])
        .collect();
    // EVOLVE-BLOCK-END
    
    save_solution(&Solution { items: selected })?;
    Ok(())
}

