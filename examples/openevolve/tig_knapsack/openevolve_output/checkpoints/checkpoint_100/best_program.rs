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
        pub max_restarts: usize,
        pub base_perturbation_size: usize,
        pub stagnation_limit: usize,
        pub perturbation_increase_factor: f64,
    }

    let hyperparameters = match _hyperparameters {
        Some(params) => {
            serde_json::from_value::<Hyperparameters>(Value::Object(params.clone()))
                .unwrap_or(Hyperparameters {
                    max_restarts: 250, // Increased restarts for better exploration
                    base_perturbation_size: 20,
                    stagnation_limit: 30, // Slightly increased limit
                    perturbation_increase_factor: 1.25,
                })
        }
        None => Hyperparameters {
            max_restarts: 250,
            base_perturbation_size: 20,
            stagnation_limit: 30,
            perturbation_increase_factor: 1.25,
        },
    };

    let num_items = challenge.values.len();
    let weights = &challenge.weights;
    let values = &challenge.values;
    let interactions = &challenge.interaction_values;
    let max_weight = challenge.max_weight;

    let mut rng = SmallRng::from_seed(challenge.seed);

    // --- State Variables (Current Solution) ---
    let mut is_selected = vec![false; num_items];
    let mut current_weight: u32 = 0;
    // Marginal gains initialized with base values
    let mut marginal_gains: Vec<i64> = values.iter().map(|&v| v as i64).collect();
    let mut current_value: i64 = 0;

    // --- Best Solution Tracking ---
    let mut best_solution = vec![false; num_items];
    let mut best_value = i64::MIN;

    #[derive(Debug, Clone, Copy)]
    enum LS_Move {
        Add(usize),
        Remove(usize),
        Swap(usize, usize), // (item_to_remove, item_to_add)
    }

    // Helper function to run Best Improvement Local Search (BILS) until convergence
    // Modifies state variables in place.
    let mut run_bils = |
        is_selected: &mut Vec<bool>,
        current_weight: &mut u32,
        marginal_gains: &mut Vec<i64>,
        current_value: &mut i64
    | {
        loop {
            let mut best_delta_v = 0i64;
            let mut best_move: Option<LS_Move> = None;

            // O(N^2) neighborhood search
            for j in 0..num_items {
                if is_selected[j] {
                    // Item j is selected: Check Remove and Swap moves involving j

                    // --- Remove Move (j out) ---
                    // Delta V = -MG[j]
                    let delta_v_remove_j = -marginal_gains[j];
                    if delta_v_remove_j > best_delta_v {
                        best_delta_v = delta_v_remove_j;
                        best_move = Some(LS_Move::Remove(j));
                    }

                    // --- Swap Moves (j out, i in) ---
                    let w_j = weights[j];
                    
                    for i in 0..num_items {
                        if !is_selected[i] {
                            let w_i = weights[i];
                            
                            // Check weight constraint for swap
                            let delta_w = w_i as i64 - w_j as i64;
                            if (*current_weight as i64 + delta_w) <= max_weight as i64 {
                                
                                // Delta V(j out, i in) = MG[i] - interactions[i][j] - MG[j]
                                let delta_v_swap = marginal_gains[i] - interactions[i][j] as i64 - marginal_gains[j];

                                if delta_v_swap > best_delta_v {
                                    best_delta_v = delta_v_swap;
                                    best_move = Some(LS_Move::Swap(j, i));
                                }
                            }
                        }
                    }
                } else {
                    // Item i is unselected: Check Add moves
                    let i = j;
                    let w_i = weights[i];

                    // --- Add Move (i in) ---
                    // Delta V = MG[i]
                    if current_weight.checked_add(w_i).map_or(false, |w| w <= max_weight) {
                        let delta_v_add_i = marginal_gains[i];

                        if delta_v_add_i > best_delta_v {
                            best_delta_v = delta_v_add_i;
                            best_move = Some(LS_Move::Add(i));
                        }
                    }
                }
            }

            if best_delta_v > 0 {
                *current_value += best_delta_v;

                match best_move.unwrap() {
                    LS_Move::Add(i) => {
                        is_selected[i] = true;
                        *current_weight += weights[i];
                        for k in 0..num_items {
                            marginal_gains[k] += interactions[i][k] as i64;
                        }
                    }
                    LS_Move::Remove(j) => {
                        is_selected[j] = false;
                        *current_weight -= weights[j];
                        for k in 0..num_items {
                            marginal_gains[k] -= interactions[j][k] as i64;
                        }
                    }
                    LS_Move::Swap(j, i) => {
                        is_selected[j] = false;
                        is_selected[i] = true;
                        *current_weight = *current_weight - weights[j] + weights[i];
                        for k in 0..num_items {
                            marginal_gains[k] = marginal_gains[k] - interactions[j][k] as i64 + interactions[i][k] as i64;
                        }
                    }
                }
            } else {
                break; // Local optimum reached
            }
        }
    };



    // --- 1. Initial Greedy Construction Phase (O(N^2)) ---
    loop {
        let mut best_item: Option<usize> = None;
        let mut best_density: f64 = f64::NEG_INFINITY;

        for i in 0..num_items {
            if !is_selected[i] {
                let item_weight = weights[i];
                if current_weight.checked_add(item_weight).map_or(false, |w| w <= max_weight) {
                    let gain = marginal_gains[i];
                    
                    let density = if item_weight == 0 {
                        if gain > 0 { f64::INFINITY } else { f64::NEG_INFINITY }
                    } else {
                        gain as f64 / item_weight as f64
                    };

                    if density > best_density {
                        best_density = density;
                        best_item = Some(i);
                    }
                }
            }
        }

        if let Some(idx_to_add) = best_item {
            // Update value incrementally
            current_value += marginal_gains[idx_to_add]; 
            
            is_selected[idx_to_add] = true;
            current_weight += weights[idx_to_add];

            // Update marginal gains for ALL items (O(N) step). 
            for i in 0..num_items {
                marginal_gains[i] += interactions[idx_to_add][i] as i64;
            }
        } else {
            break;
        }
    }
    
    // Initialize best solution tracking (current_value is already correct)
    best_value = current_value;
    best_solution = is_selected.clone();

    // --- 2. Iterated Local Search (ILS) with Adaptive Perturbation ---
    let mut iterations_since_best_update = 0;
    let mut dynamic_perturbation_size = hyperparameters.base_perturbation_size;

    for restart in 0..hyperparameters.max_restarts {
        
        // A. Run BILS on the current state until local optimum is reached
        run_bils(&mut is_selected, &mut current_weight, &mut marginal_gains, &mut current_value);

        // B. Update Best Solution and adapt perturbation strength
        if current_value > best_value {
            best_value = current_value;
            best_solution = is_selected.clone();
            iterations_since_best_update = 0;
            // On improvement, reset perturbation strength to base to focus search locally
            dynamic_perturbation_size = hyperparameters.base_perturbation_size;
        } else {
            iterations_since_best_update += 1;
        }

        if iterations_since_best_update >= hyperparameters.stagnation_limit {
            // Stagnation detected: increase perturbation strength to escape
            dynamic_perturbation_size = (dynamic_perturbation_size as f64 * hyperparameters.perturbation_increase_factor).round() as usize;
            // Cap the size to avoid destroying the solution completely
            let max_p_size = (num_items as f64 * 0.4) as usize; // Don't remove more than 40% of items
            dynamic_perturbation_size = dynamic_perturbation_size.min(max_p_size);
            iterations_since_best_update = 0; // Reset counter after strengthening
        }

        if restart == hyperparameters.max_restarts - 1 {
            break; // No need to perturb after the last optimization run
        }

        // C. Perturbation Phase (Biased removal + Incremental Repair)
        let mut selected_indices: Vec<usize> = (0..num_items).filter(|&i| is_selected[i]).collect();
        let num_selected = selected_indices.len();
        
        let p_size = dynamic_perturbation_size.min(num_selected);
        
        // 1. & 2. Biasedly remove P_size items and update state incrementally (O(P*N))
        for _ in 0..p_size {
            if selected_indices.is_empty() { break; }

            // Calculate MG_max for normalization (O(N_selected))
            let mg_max = selected_indices.iter()
                .map(|&i| marginal_gains[i])
                .max()
                .unwrap_or(1); 

            // Calculate weights: S_j = MG_max - MG_j + 1 (Bias towards low MG items)
            let mut total_weight = 0i64;
            let weights_and_indices: Vec<(i64, usize)> = selected_indices.iter().map(|&j| {
                // Ensure weight is positive and non-zero. 
                let weight = mg_max - marginal_gains[j] + 1; 
                total_weight += weight;
                (weight, j)
            }).collect();

            // Select item j using roulette wheel selection (O(N_selected))
            let target = rng.gen_range(0..total_weight);
            let mut cumulative_weight = 0i64;
            
            let mut removal_index_in_vec = 0; // Index in selected_indices vector
            
            for (idx, &(weight, _)) in weights_and_indices.iter().enumerate() {
                cumulative_weight += weight;
                if cumulative_weight > target {
                    removal_index_in_vec = idx;
                    break;
                }
            }
            
            // Retrieve item index j and remove it from selected_indices (O(1) using swap_remove)
            let j = selected_indices.swap_remove(removal_index_in_vec);

            // Update value incrementally using the current marginal gain
            current_value -= marginal_gains[j];
            
            is_selected[j] = false;
            current_weight -= weights[j];

            // Update marginal gains for all other items due to j's removal (O(N))
            for k in 0..num_items {
                marginal_gains[k] -= interactions[j][k] as i64;
            }
        }

        // 3. Greedy Repair (Add items back based on marginal density)
        loop {
            let mut best_item: Option<usize> = None;
            let mut best_density: f64 = f64::NEG_INFINITY;

            for i in 0..num_items {
                if !is_selected[i] {
                    let item_weight = weights[i];
                    if current_weight.checked_add(item_weight).map_or(false, |w| w <= max_weight) {
                        let gain = marginal_gains[i];
                        
                        let density = if item_weight == 0 {
                            if gain > 0 { f64::INFINITY } else { f64::NEG_INFINITY }
                        } else {
                            gain as f64 / item_weight as f64
                        };

                        if density > best_density {
                            best_density = density;
                            best_item = Some(i);
                        }
                    }
                }
            }

            if let Some(idx_to_add) = best_item {
                is_selected[idx_to_add] = true;
                current_weight += weights[idx_to_add];
                current_value += marginal_gains[idx_to_add]; // Update value incrementally

                // Update marginal gains for ALL items (O(N) step). 
                for i in 0..num_items {
                    marginal_gains[i] += interactions[idx_to_add][i] as i64;
                }
            } else {
                break;
            }
        }
    }

    // Finalize solution indices from the best solution found
    let selected: Vec<usize> = (0..num_items).filter(|&i| best_solution[i]).collect();
    // EVOLVE-BLOCK-END
    
    save_solution(&Solution { items: selected })?;
    Ok(())
}