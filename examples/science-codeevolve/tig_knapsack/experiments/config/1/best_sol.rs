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
    use rand::seq::SliceRandom;

    #[derive(Serialize, Deserialize)]
    pub struct Hyperparameters {
        // Optionally define hyperparameters here
    }

    let num_items = challenge.values.len();
    let mut rng = SmallRng::from_seed(challenge.seed);

    // --- 1. Initial Greedy Solution ---
    // A better greedy heuristic that considers the potential from positive interactions.
    let mut items_with_density: Vec<(usize, f64)> = (0..num_items)
        .map(|i| {
            let positive_interactions: i32 = challenge.interaction_values[i]
                .iter()
                .filter(|&&v| v > 0)
                .sum();
            // The 0.5 factor is a heuristic to temper optimism, as not all partners will be chosen.
            let density = (challenge.values[i] as f64 + 0.5 * positive_interactions as f64)
                / challenge.weights[i] as f64;
            (i, density)
        })
        .collect();

    items_with_density.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut in_knapsack = vec![false; num_items];
    let mut current_weight = 0u64;

    for (item_idx, _) in items_with_density {
        if current_weight + challenge.weights[item_idx] as u64 <= challenge.max_weight as u64 {
            in_knapsack[item_idx] = true;
            current_weight += challenge.weights[item_idx] as u64;
        }
    }

    // --- 2. Local Search (Iterated Hill Climbing) ---
    // Pre-calculate interaction contributions for efficient delta evaluation.
    // interaction_contrib[i] = sum_{j in knapsack} interaction(i, j)
    let mut interaction_contrib = vec![0i64; num_items];
    for i in 0..num_items {
        for j in 0..num_items {
            if in_knapsack[j] {
                interaction_contrib[i] += challenge.interaction_values[i][j] as i64;
            }
        }
    }

    loop {
        let mut improvement_found = false;

        // --- Neighborhood 1: Add an item (1-0 moves) if space allows ---
        // Find the best item to add that provides the most improvement.
        let mut best_add_delta = 0i64;
        let mut best_add_idx = None;

        for i in 0..num_items {
            if !in_knapsack[i] && current_weight + challenge.weights[i] as u64 <= challenge.max_weight as u64 {
                let delta = challenge.values[i] as i64 + interaction_contrib[i];
                if delta > best_add_delta {
                    best_add_delta = delta;
                    best_add_idx = Some(i);
                }
            }
        }

        if let Some(add_idx) = best_add_idx {
            // Apply the best "add" move
            in_knapsack[add_idx] = true;
            current_weight += challenge.weights[add_idx] as u64;
            // Update contribution scores for all items based on the new addition
            for i in 0..num_items {
                interaction_contrib[i] += challenge.interaction_values[i][add_idx] as i64;
            }
            improvement_found = true;
            continue; // Restart search from the new, improved state
        }

        // --- Neighborhood 2: Swap items (1-1 moves) ---
        // Find the best swap (one item in, one item out).
        let mut best_swap_delta = 0i64;
        let mut best_in_idx = None;
        let mut best_out_idx = None;

        let mut out_indices: Vec<usize> = (0..num_items).filter(|&i| !in_knapsack[i]).collect();
        let mut in_indices: Vec<usize> = (0..num_items).filter(|&i| in_knapsack[i]).collect();
        
        // Shuffle to introduce randomness, helping to escape shallow local optima.
        out_indices.shuffle(&mut rng);
        in_indices.shuffle(&mut rng);

        for &out_idx in &out_indices {
            for &in_idx in &in_indices {
                if current_weight - challenge.weights[in_idx] as u64 + challenge.weights[out_idx] as u64 > challenge.max_weight as u64 {
                    continue;
                }

                // O(1) delta calculation for the swap move
                let delta = (challenge.values[out_idx] as i64 - challenge.values[in_idx] as i64)
                    + (interaction_contrib[out_idx] - challenge.interaction_values[out_idx][in_idx] as i64)
                    - interaction_contrib[in_idx];

                if delta > best_swap_delta {
                    best_swap_delta = delta;
                    best_in_idx = Some(in_idx);
                    best_out_idx = Some(out_idx);
                }
            }
        }

        if let (Some(in_idx), Some(out_idx)) = (best_in_idx, best_out_idx) {
            // Apply the best swap move
            in_knapsack[in_idx] = false;
            in_knapsack[out_idx] = true;

            current_weight = current_weight - challenge.weights[in_idx] as u64 + challenge.weights[out_idx] as u64;
            
            // Update contribution scores for all items based on the swap (O(n))
            for i in 0..num_items {
                interaction_contrib[i] -= challenge.interaction_values[i][in_idx] as i64;
                interaction_contrib[i] += challenge.interaction_values[i][out_idx] as i64;
            }
            improvement_found = true;
        }

        if !improvement_found {
            break; // Local optimum reached
        }
    }

    // --- 3. Finalize Solution ---
    let selected: Vec<usize> = (0..num_items)
        .filter(|&i| in_knapsack[i])
        .collect();
    // EVOLVE-BLOCK-END
    
    save_solution(&Solution { items: selected })?;
    Ok(())
}

