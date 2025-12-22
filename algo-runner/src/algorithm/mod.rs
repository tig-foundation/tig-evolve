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
    
    // --- QKP Solver using Simulated Annealing with Delta Evaluation ---

    let num_items = challenge.values.len();
    let max_weight = challenge.max_weight;
    let mut rng = SmallRng::from_seed(challenge.seed);

    // Helper structure for state management and delta calculation
    struct QkpState {
        is_selected: Vec<bool>,
        current_value: i64,
        current_weight: u32,
        // interaction_sums[i] = sum_{j in S, j != i} I_{ij}. Used for O(N) delta updates.
        interaction_sums: Vec<i64>, 
        n: usize,
    }

    impl QkpState {
        fn new(n: usize) -> Self {
            QkpState {
                is_selected: vec![false; n],
                current_value: 0,
                current_weight: 0,
                interaction_sums: vec![0; n],
                n,
            }
        }

        // Initialize state based on an initial selection (e.g., greedy start)
        fn initialize(&mut self, challenge: &Challenge, initial_selection: &[usize]) {
            self.current_weight = initial_selection.iter().map(|&i| challenge.weights[i]).sum();
            
            // 1. Set membership and calculate interaction sums
            for &i in initial_selection {
                self.is_selected[i] = true;
            }

            for i in 0..self.n {
                if self.is_selected[i] {
                    for j in 0..self.n {
                        if i != j && self.is_selected[j] {
                            self.interaction_sums[i] += challenge.interaction_values[i][j] as i64;
                        }
                    }
                }
            }
            
            // 2. Calculate total value
            let base_value: i64 = initial_selection.iter().map(|&i| challenge.values[i] as i64).sum();
            // Total interactions = 1/2 * sum(interaction_sums)
            let interaction_value: i64 = self.interaction_sums.iter().sum::<i64>() / 2;
            self.current_value = base_value + interaction_value;
        }

        // Delta for adding L (unselected)
        fn delta_add(&self, challenge: &Challenge, l: usize) -> (i64, i32) {
            let v_l = challenge.values[l] as i64;
            let w_l = challenge.weights[l] as i32;

            // 1. Base value change
            let mut delta_v = v_l;

            // 2. Interaction change: L interacts with all currently selected items S
            for i in 0..self.n {
                if self.is_selected[i] {
                    delta_v += challenge.interaction_values[l][i] as i64;
                }
            }
            
            (delta_v, w_l)
        }

        // Delta for removing K (selected)
        fn delta_remove(&self, challenge: &Challenge, k: usize) -> (i64, i32) {
            let v_k = challenge.values[k] as i64;
            let w_k = challenge.weights[k] as i32;

            // 1. Base value change
            let delta_v_base = -v_k;

            // 2. Interaction change: K stops interacting with all currently selected items S \ {k}
            let delta_v_interaction = -self.interaction_sums[k];
            
            let delta_v = delta_v_base + delta_v_interaction;
            let delta_w = -(w_k);
            
            (delta_v, delta_w)
        }

        // Apply an add move (L in)
        fn apply_add(&mut self, challenge: &Challenge, l: usize, delta_v: i64, delta_w: i32) {
            self.is_selected[l] = true;
            self.current_weight = ((self.current_weight as i32) + delta_w) as u32;
            self.current_value += delta_v;

            // O(N) update of interaction sums

            // 1. Update sums for existing items i in S_new \ {l}
            let mut sum_l: i64 = 0;
            for i in 0..self.n {
                if self.is_selected[i] && i != l { 
                    let i_li = challenge.interaction_values[l][i] as i64;
                    
                    // i gains interaction with l
                    self.interaction_sums[i] += i_li;
                    
                    // Calculate sum for L simultaneously
                    sum_l += i_li;
                }
            }
            
            // 2. Update sum for L (newly added)
            self.interaction_sums[l] = sum_l; 
        }

        // Apply a remove move (K out)
        fn apply_remove(&mut self, challenge: &Challenge, k: usize, delta_v: i64, delta_w: i32) {
            self.is_selected[k] = false;
            self.current_weight = ((self.current_weight as i32) + delta_w) as u32;
            self.current_value += delta_v;

            // O(N) update of interaction sums

            // 1. Update sums for existing items i in S_new
            for i in 0..self.n {
                if self.is_selected[i] { 
                    let i_ki = challenge.interaction_values[k][i] as i64;
                    
                    // i loses interaction with k
                    self.interaction_sums[i] -= i_ki;
                }
            }
            
            // 2. Update sum for K (newly removed)
            self.interaction_sums[k] = 0; 
        }

        // Delta for swapping K (out, selected) and L (in, unselected)
        fn delta_swap(&self, challenge: &Challenge, k: usize, l: usize) -> (i64, i32) {
            let v_k = challenge.values[k] as i64;
            let v_l = challenge.values[l] as i64;
            let w_k = challenge.weights[k] as i32;
            let w_l = challenge.weights[l] as i32;

            // 1. Base value change
            let mut delta_v = v_l - v_k;

            // 2. Interaction change: Gain - Loss
            // Loss: S_k = sum_{i in S_old \ {k}} I_ki. This is self.interaction_sums[k]
            // Gain: sum_{i in S_old \ {k}} I_li.

            let mut interaction_gain: i64 = 0;
            
            // Calculate gain term: O(N) loop over items i in S_old \ {k}
            for i in 0..self.n {
                // Check if i is selected AND i is not k (k is being removed)
                if self.is_selected[i] && i != k {
                    interaction_gain += challenge.interaction_values[l][i] as i64;
                }
            }

            delta_v += interaction_gain - self.interaction_sums[k];
            
            let delta_w = w_l - w_k;
            (delta_v, delta_w)
        }

        // Apply a swap move (K out, L in)
        fn apply_swap(&mut self, challenge: &Challenge, k: usize, l: usize, delta_v: i64, delta_w: i32) {
            // Update membership
            self.is_selected[k] = false;
            self.is_selected[l] = true;

            // Update global weight and value
            self.current_weight = ((self.current_weight as i32) + delta_w) as u32;
            self.current_value += delta_v;

            // O(N) update of interaction sums

            // 1. Update sums for remaining items i in S_new \ {l}
            // S_new \ {l} = S_old \ {k}
            let mut sum_l: i64 = 0;
            for i in 0..self.n {
                // If i is currently selected (i.e., i in S_old \ {k})
                if self.is_selected[i] && i != l { 
                    let i_li = challenge.interaction_values[l][i] as i64;
                    let i_ki = challenge.interaction_values[k][i] as i64;
                    
                    // i loses interaction with k, gains interaction with l
                    self.interaction_sums[i] = self.interaction_sums[i] - i_ki + i_li;
                    
                    // Calculate sum for L simultaneously
                    sum_l += i_li;
                }
            }
            
            // 2. Update sum for L (newly added)
            self.interaction_sums[l] = sum_l; 

            // 3. Update sum for K (newly removed)
            self.interaction_sums[k] = 0; 
        }
    }
    
    // Enum to represent the type of move for application
    enum MoveType {
        Swap(usize, usize),    // (k_out, l_in)
        Remove(usize),         // (k_out)
        Add(usize),            // (l_in)
    }

    // --- 1. Initial Greedy Solution (V/W ratio) ---
    
    let mut items_sorted: Vec<(usize, f64)> = (0..num_items)
        .map(|i| {
            // Handle zero weight
            let weight = challenge.weights[i].max(1) as f64; 
            let ratio = challenge.values[i] as f64 / weight;
            (i, ratio)
        })
        .collect();
    items_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut initial_selection = Vec::new();
    let mut current_weight: u32 = 0;
    
    for (item_idx, _) in items_sorted {
        if current_weight + challenge.weights[item_idx] <= max_weight {
            initial_selection.push(item_idx);
            current_weight += challenge.weights[item_idx];
        }
    }
    
    // 2. Initialize QKP State
    let mut state = QkpState::new(num_items);
    state.initialize(challenge, &initial_selection);
    
    // 3. Initialize Best Solution Tracker
    let mut best_value = state.current_value;
    let mut best_is_selected = state.is_selected.clone(); 
    
    // --- 4. Simulated Annealing Parameters ---
    
    // Parameters tuned for higher quality (increased iteration count)
    // Target 5 million iterations total (5000 steps * 1000 moves/step)
    const TOTAL_STEPS: u64 = 5000; 
    const MOVES_PER_STEP: u64 = 1000; 
    
    // Estimate initial temperature T0 based on average change magnitude.
    // Use 1% of the initial value, bounded reasonably, to ensure a high initial acceptance rate (e.g., 80%).
    let t_initial: f64 = (state.current_value as f64 * 0.01).max(1000.0);
    let t_final: f64 = 1e-4; 
    
    // Calculate cooling rate alpha for geometric cooling
    let alpha = (t_final / t_initial).powf(1.0 / (TOTAL_STEPS as f64));
    
    let mut temperature = t_initial;
    let mut step_count: u64 = 0;

    // --- 5. SA Loop ---
    
    while step_count < TOTAL_STEPS {
        step_count += 1;
        
        for _ in 0..MOVES_PER_STEP {
            
            // Select move type: Swap (80%), Remove (10%), Add (10%)
            let move_type_rand = rng.gen_range(0..10);
            
            let (dv, dw, move_applied) = match move_type_rand {
                0..=7 => { // Swap attempt (80%)
                    // Find k_out (selected) and l_in (unselected)
                    let mut k_out;
                    let mut l_in;

                    // O(1) expected sampling
                    loop {
                        k_out = rng.gen_range(0..num_items);
                        if state.is_selected[k_out] { break; }
                    }
                    loop {
                        l_in = rng.gen_range(0..num_items);
                        if !state.is_selected[l_in] { break; }
                    }
                    
                    let (dv, dw) = state.delta_swap(challenge, k_out, l_in);
                    let move_applied = MoveType::Swap(k_out, l_in);
                    (dv, dw, move_applied)
                },
                8 => { // Remove attempt (10%)
                    // Find k_out (selected)
                    let mut k_out;
                    let mut attempts = 0;
                    loop {
                        k_out = rng.gen_range(0..num_items);
                        if state.is_selected[k_out] || attempts > num_items { break; } 
                        attempts += 1;
                    }
                    // If no item is selected, skip
                    if !state.is_selected[k_out] { continue; } 

                    let (dv, dw) = state.delta_remove(challenge, k_out);
                    (dv, dw, MoveType::Remove(k_out))
                },
                _ => { // Add attempt (10%)
                    // Find l_in (unselected)
                    let mut l_in;
                    let mut attempts = 0;
                    loop {
                        l_in = rng.gen_range(0..num_items);
                        if !state.is_selected[l_in] || attempts > num_items { break; }
                        attempts += 1;
                    }
                    // If all items are selected, skip
                    if state.is_selected[l_in] { continue; } 

                    let (dv, dw) = state.delta_add(challenge, l_in);
                    (dv, dw, MoveType::Add(l_in))
                }
            };

            // Check feasibility (Weight constraint)
            let new_weight = (state.current_weight as i32) + dw;
            
            if new_weight <= max_weight as i32 && new_weight >= 0 { 
                
                let mut accepted = false;
                
                if dv >= 0 {
                    accepted = true;
                } else {
                    // Metropolis criterion: P = exp(Delta V / T)
                    if temperature > 1e-9 {
                        let prob = (-dv as f64 / temperature).exp();
                        if rng.gen::<f64>() < prob {
                            accepted = true;
                        }
                    }
                }
                
                if accepted {
                    match move_applied {
                        MoveType::Swap(k, l) => state.apply_swap(challenge, k, l, dv, dw),
                        MoveType::Remove(k) => state.apply_remove(challenge, k, dv, dw),
                        MoveType::Add(l) => state.apply_add(challenge, l, dv, dw),
                    }

                    // Update best solution found so far
                    if state.current_value > best_value {
                        best_value = state.current_value;
                        best_is_selected.copy_from_slice(&state.is_selected);
                    }
                }
            }
        }
        
        // Cooling
        temperature *= alpha;
    }
    
    // Convert best_is_selected (Vec<bool>) back to selected (Vec<usize>)
    let mut selected = Vec::with_capacity(num_items);
    for i in 0..num_items {
        if best_is_selected[i] {
            selected.push(i);
        }
    }
    // EVOLVE-BLOCK-END
    
    save_solution(&Solution { items: selected })?;
    Ok(())
}

