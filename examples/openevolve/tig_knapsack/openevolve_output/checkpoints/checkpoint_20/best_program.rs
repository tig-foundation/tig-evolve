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
    #[derive(Serialize, Deserialize)]
    pub struct Hyperparameters {
        // Optionally define hyperparameters here
    }

    // Phase 1: Iterative greedy construction heuristic.
    // At each step, it adds the item with the highest marginal gain-to-weight ratio.
    let num_items = challenge.values.len();
    let mut selected = Vec::new();

    if num_items > 0 {
        let mut total_weight: u32 = 0;
        let mut is_selected = vec![false; num_items];
        let mut marginal_gains: Vec<f64> = challenge.values.iter().map(|&v| v as f64).collect();

        loop {
            let mut best_item: Option<usize> = None;
            let mut best_ratio = f64::NEG_INFINITY;

            for i in 0..num_items {
                if !is_selected[i] {
                    let item_weight = challenge.weights[i];
                    if total_weight + item_weight <= challenge.max_weight {
                        let ratio = if item_weight > 0 {
                            marginal_gains[i] / item_weight as f64
                        } else {
                            if marginal_gains[i] > 0.0 {
                                f64::INFINITY
                            } else {
                                f64::NEG_INFINITY
                            }
                        };

                        if ratio > best_ratio {
                            best_ratio = ratio;
                            best_item = Some(i);
                        }
                    }
                }
            }

            if let Some(item_to_add) = best_item {
                selected.push(item_to_add);
                is_selected[item_to_add] = true;
                total_weight += challenge.weights[item_to_add];

                for i in 0..num_items {
                    if !is_selected[i] {
                        marginal_gains[i] += challenge.interaction_values[item_to_add][i] as f64;
                    }
                }
            } else {
                break;
            }
        }

        // Phase 2: Local search (1-opt swap) to improve the greedy solution.
        // This phase iterates and performs the best possible swap (one item in, one item out)
        // until no further improvements can be made.
        if !selected.is_empty() {
            let mut interaction_sums = vec![0i64; num_items];
            for i in 0..num_items {
                for &k in &selected {
                    interaction_sums[i] += challenge.interaction_values[i][k] as i64;
                }
            }

            loop {
                let mut improvement_found = false;
                let mut best_swap: Option<(usize, usize, i64)> = None; // (selected_vec_idx, item_j_idx, delta_value)

                // Best-improvement strategy: find the best possible swap in the entire 1-opt neighborhood.
                for i_vec_idx in 0..selected.len() {
                    let item_i = selected[i_vec_idx];
                    for item_j in 0..num_items {
                        if is_selected[item_j] { continue; }

                        let delta_weight = challenge.weights[item_j] as i64 - challenge.weights[item_i] as i64;
                        if (total_weight as i64 + delta_weight) > challenge.max_weight as i64 {
                            continue;
                        }

                        let delta_value = (challenge.values[item_j] as i64 - challenge.values[item_i] as i64)
                                        + (interaction_sums[item_j] - challenge.interaction_values[item_j][item_i] as i64)
                                        - interaction_sums[item_i];

                        if delta_value > 0 {
                            if best_swap.is_none() || delta_value > best_swap.unwrap().2 {
                                best_swap = Some((i_vec_idx, item_j, delta_value));
                            }
                        }
                    }
                }

                if let Some((i_vec_idx_to_swap, item_j_to_swap, _)) = best_swap {
                    let item_i_to_swap = selected[i_vec_idx_to_swap];

                    let delta_weight = challenge.weights[item_j_to_swap] as i64 - challenge.weights[item_i_to_swap] as i64;
                    total_weight = (total_weight as i64 + delta_weight) as u32;
                    is_selected[item_i_to_swap] = false;
                    is_selected[item_j_to_swap] = true;
                    selected[i_vec_idx_to_swap] = item_j_to_swap;

                    // Update interaction_sums efficiently in O(N).
                    for k in 0..num_items {
                        interaction_sums[k] += (challenge.interaction_values[k][item_j_to_swap] - challenge.interaction_values[k][item_i_to_swap]) as i64;
                    }
                    improvement_found = true;
                }

                if !improvement_found {
                    break; // Local optimum reached.
                }
            }
        }
    }
    // EVOLVE-BLOCK-END
    
    save_solution(&Solution { items: selected })?;
    Ok(())
}