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
    let num_items = challenge.values.len();

    #[derive(Serialize, Deserialize)]
    pub struct Hyperparameters {
        pub tabu_iterations: usize,
        pub tabu_tenure: usize,
        pub non_improving_limit: usize,
    }

    let hyperparameters = match _hyperparameters {
        Some(params) => {
            serde_json::from_value::<Hyperparameters>(Value::Object(params.clone()))
                .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
        }
        None => Hyperparameters {
            tabu_iterations: 10000,
            // A dynamic tenure based on problem size often works well.
            tabu_tenure: ((num_items as f64).sqrt().round() as usize).max(5),
            non_improving_limit: 1000,
        },
    };

    // Phase 1: Iterative greedy construction heuristic.
    // At each step, it adds the item with the highest marginal gain-to-weight ratio.
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

            // Calculate initial value
            let mut current_value: i64 = 0;
            for &i in &selected {
                current_value += challenge.values[i] as i64;
                // Summing interaction_sums[i] over i in selected counts interactions twice.
                current_value += interaction_sums[i] / 2;
            }

            let mut best_value = current_value;
            let mut best_selected = selected.clone();
            let mut best_total_weight = total_weight;

            let mut rng = SmallRng::from_seed(challenge.seed);

            // Phase 2: Tabu Search (1-opt swap)
            // Tabu List: stores the iteration when an item's removal restriction expires.
            let mut tabu_list = vec![0usize; num_items];
            let tabu_tenure = hyperparameters.tabu_tenure;
            let max_iterations = hyperparameters.tabu_iterations;
            let mut iterations_since_best_update = 0;

            // Define move types for clarity
            enum Move {
                Swap { i_vec_idx: usize, j_idx: usize },
                Add { j_idx: usize },
                Remove { i_vec_idx: usize },
            }

            for iteration in 1..=max_iterations {
                iterations_since_best_update += 1;
                let mut best_move: Option<(Move, i64)> = None;
                let mut best_delta = i64::MIN;

                // --- 1. Search Swap neighborhood (i out, j in) ---
                for i_vec_idx in 0..selected.len() {
                    let item_i = selected[i_vec_idx]; // Item to remove

                    for item_j in 0..num_items { // Item to add
                        if is_selected[item_j] { continue; }

                        // 1. Check weight constraint
                        let delta_weight = challenge.weights[item_j] as i64 - challenge.weights[item_i] as i64;
                        if (total_weight as i64 + delta_weight) > challenge.max_weight as i64 {
                            continue;
                        }

                        // 2. Calculate Delta Value
                        // Delta V = (v_j - v_i) + (interaction_sums[j] - c_{ji}) - interaction_sums[i]
                        let delta_value = (challenge.values[item_j] as i64 - challenge.values[item_i] as i64)
                                        + (interaction_sums[item_j] - challenge.interaction_values[item_j][item_i] as i64)
                                        - interaction_sums[item_i];

                        let new_value = current_value + delta_value;

                        // 3. Tabu Check & Aspiration Criterion
                        // Item i (removed) is tabu from being added back.
                        let is_tabu = tabu_list[item_i] > iteration;
                        let is_aspirated = is_tabu && (new_value > best_value);

                        if !is_tabu || is_aspirated {
                            if delta_value > best_delta || (delta_value == best_delta && rng.gen_bool(0.1)) {
                                best_delta = delta_value;
                                best_move = Some((Move::Swap { i_vec_idx, j_idx: item_j }, delta_value));
                            }
                        }
                    }
                }

                // --- 2. Search Add neighborhood (j in) ---
                for item_j in 0..num_items {
                    if is_selected[item_j] { continue; }
                    let item_weight = challenge.weights[item_j];

                    if total_weight + item_weight <= challenge.max_weight {
                        // Delta V = v_j + interaction_sums[j]
                        let delta_value = challenge.values[item_j] as i64 + interaction_sums[item_j];
                        
                        if delta_value > best_delta || (delta_value == best_delta && rng.gen_bool(0.1)) {
                            best_delta = delta_value;
                            best_move = Some((Move::Add { j_idx: item_j }, delta_value));
                        }
                    }
                }

                // --- 3. Search Remove neighborhood (i out) ---
                for i_vec_idx in 0..selected.len() {
                    let item_i = selected[i_vec_idx]; // Item to remove

                    // Delta V = -v_i - interaction_sums[i]
                    let delta_value = -(challenge.values[item_i] as i64 + interaction_sums[item_i]);
                    let new_value = current_value + delta_value;

                    // Tabu Check: Item i (removed) is tabu from being added back.
                    let is_tabu = tabu_list[item_i] > iteration;
                    let is_aspirated = is_tabu && (new_value > best_value);

                    if !is_tabu || is_aspirated {
                        if delta_value > best_delta || (delta_value == best_delta && rng.gen_bool(0.1)) {
                            best_delta = delta_value;
                            best_move = Some((Move::Remove { i_vec_idx }, delta_value));
                        }
                    }
                }

                // If no feasible move (respecting weight constraint) is found, stop.
                if best_move.is_none() {
                    break;
                }

                // Perform the best move found
                let (move_type, delta) = best_move.unwrap();
                current_value += delta;

                match move_type {
                    Move::Swap { i_vec_idx, j_idx } => {
                        let item_i = selected[i_vec_idx];

                        // 1. Update weight
                        let delta_weight = challenge.weights[j_idx] as i64 - challenge.weights[item_i] as i64;
                        total_weight = (total_weight as i64 + delta_weight) as u32;

                        // 2. Update selection status and list
                        is_selected[item_i] = false;
                        is_selected[j_idx] = true;
                        selected[i_vec_idx] = j_idx;

                        // 3. Update interaction_sums efficiently in O(N).
                        // interaction_sums[k] += c_{k, j_new} - c_{k, i_old}
                        for k in 0..num_items {
                            interaction_sums[k] += (challenge.interaction_values[k][j_idx] - challenge.interaction_values[k][item_i]) as i64;
                        }

                        // 4. Update Tabu List: Tabu the item removed (i)
                        tabu_list[item_i] = iteration + tabu_tenure;
                    }
                    Move::Add { j_idx } => {
                        // 1. Update weight
                        total_weight += challenge.weights[j_idx];

                        // 2. Update selection status and list
                        is_selected[j_idx] = true;
                        selected.push(j_idx);

                        // 3. Update interaction_sums efficiently in O(N).
                        // interaction_sums[k] += c_{k, j_new}
                        for k in 0..num_items {
                            interaction_sums[k] += challenge.interaction_values[k][j_idx] as i64;
                        }
                        // No item removed, no tabu restriction applied.
                    }
                    Move::Remove { i_vec_idx } => {
                        let item_i = selected[i_vec_idx];

                        // 1. Update weight
                        total_weight -= challenge.weights[item_i];

                        // 2. Update selection status and list
                        is_selected[item_i] = false;
                        selected.swap_remove(i_vec_idx); // O(1) removal, order doesn't matter

                        // 3. Update interaction_sums efficiently in O(N).
                        // interaction_sums[k] -= c_{k, i_old}
                        for k in 0..num_items {
                            interaction_sums[k] -= challenge.interaction_values[k][item_i] as i64;
                        }

                        // 4. Update Tabu List: Tabu the item removed (i)
                        tabu_list[item_i] = iteration + tabu_tenure;
                    }
                }

                // 5. Update Best Solution
                if current_value > best_value {
                    best_value = current_value;
                    best_selected = selected.clone();
                    best_total_weight = total_weight;
                    iterations_since_best_update = 0;
                }

                // 6. Intensification on Stagnation
                if iterations_since_best_update > hyperparameters.non_improving_limit {
                    // Return to the best known solution to intensify the search in that area
                    selected = best_selected.clone();
                    current_value = best_value;
                    total_weight = best_total_weight;

                    // Recalculate state based on the restored best solution
                    is_selected.fill(false);
                    for &item in &selected {
                        is_selected[item] = true;
                    }

                    interaction_sums.fill(0);
                    for i in 0..num_items {
                        for &k in &selected {
                            interaction_sums[i] += challenge.interaction_values[i][k] as i64;
                        }
                    }
                    
                    // Clear the tabu list to allow free exploration from this point
                    tabu_list.fill(0);
                    iterations_since_best_update = 0;
                }
            }

            // Restore the best solution found during the Tabu Search run
            selected = best_selected;
            total_weight = best_total_weight;
        }
    }
    // EVOLVE-BLOCK-END
    
    save_solution(&Solution { items: selected })?;
    Ok(())
}