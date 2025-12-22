// DO NOT CHANGE THESE IMPORTS
use crate::challenge::{Challenge, Solution};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use rand::{rngs::SmallRng, Rng, SeedableRng};

pub fn help() {
    // Print help information about your algorithm here. It will be invoked with `help_algorithm` script
    println!("Greedy nearest-neighbor heuristic for VRPTW");
}

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

    // Helper struct to store precalculated time profile data
    struct TimeProfile {
        // T_start[i] = Service start time at route[i]
        start_times: Vec<i32>, 
        // T_dep[i] = Departure time from route[i]
        departure_times: Vec<i32>,
    }

    /// Calculates the time profile for a given route. Assumes route is feasible.
    fn calculate_time_profile(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
    ) -> TimeProfile {
        let N = route.len();
        let mut start_times: Vec<i32> = vec![0; N];
        let mut departure_times: Vec<i32> = vec![0; N];
        
        let mut curr_time = 0; 
        let mut curr_node = 0;

        for pos in 1..N {
            let next_node = route[pos];
            curr_time += distance_matrix[curr_node][next_node]; // arrival
            let start_time = ready_times[next_node].max(curr_time);
            
            let dep_time = start_time + service_time; 
            
            start_times[pos] = start_time;
            departure_times[pos] = dep_time;
            curr_time = dep_time;
            curr_node = next_node;
        }

        TimeProfile { start_times, departure_times }
    }


    fn find_best_insertion(
        route: &Vec<usize>,
        remaining_nodes: Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Option<(usize, usize)> {
        
        // 1. Precalculate time profile for the existing feasible route R (O(N))
        let profile = calculate_time_profile(route, distance_matrix, service_time, ready_times);
        let N = route.len();

        let mut best_c2: Option<i32> = None;
        let mut best: Option<(usize, usize)> = None;

        // Solomon parameters: alpha1 controls distance impact, alpha2 controls time impact.
        let alpha1 = 1; 
        let alpha2 = 3; 
        let lambda = 1; 
    
        for insert_node in remaining_nodes {
            
            // Iterate over insertion positions (i, j), where i=route[pos-1], j=route[pos]
            for pos in 1..N {
                let j_node = route[pos]; 
                let i_node = route[pos - 1]; 
                
                let T_i_dep = profile.departure_times[pos - 1]; // Original departure time from i

                // --- 1. Calculate time profile at u (i -> u) ---
                let arrival_u = T_i_dep + distance_matrix[i_node][insert_node];
                let start_u = ready_times[insert_node].max(arrival_u);
                
                if start_u > due_times[insert_node] {
                    continue; 
                }
                let departure_u = start_u + service_time;

                // --- 2. Calculate time profile at j if u is inserted (u -> j) ---
                let arrival_j_u = departure_u + distance_matrix[insert_node][j_node];
                let start_j_u = ready_times[j_node].max(arrival_j_u);
                
                if start_j_u > due_times[j_node] {
                    continue; 
                }
                
                // --- 3. Check delay propagation feasibility ---
                
                let original_start_j = profile.start_times[pos];
                let c12 = start_j_u - original_start_j; // Delay propagation at j
                
                let mut feasible_time = true;

                if c12 > 0 {
                    // Delay propagates. Check subsequent nodes k = route[k_pos], k_pos > pos
                    let mut propagated_delay = c12;
                    
                    // Start checking from node k = route[pos + 1] onwards
                    for k_pos in (pos + 1)..N {
                        let k_node = route[k_pos];
                        
                        // Calculate waiting time W_k = T_start[k] - A_k
                        let prev_node = route[k_pos - 1];
                        let arrival_k = profile.departure_times[k_pos - 1] + distance_matrix[prev_node][k_node];
                        let waiting_time = profile.start_times[k_pos] - arrival_k;
                        
                        // Delay is absorbed by waiting time
                        if waiting_time >= propagated_delay {
                            propagated_delay = 0;
                            break; 
                        } else {
                            // Remaining delay
                            propagated_delay -= waiting_time;
                            
                            // Check due time constraint: T_start[k] + remaining_delay <= DT_k
                            if profile.start_times[k_pos] + propagated_delay > due_times[k_node] {
                                feasible_time = false;
                                break;
                            }
                        }
                    }
                }

                if feasible_time {
                    // --- Cost Calculation (Maximize score) ---
                    
                    // C11: Distance increase
                    let c11 = distance_matrix[i_node][insert_node]
                        + distance_matrix[insert_node][j_node]
                        - distance_matrix[i_node][j_node];
        
                    // C1: Local insertion score (maximize savings/minimize cost)
                    let c1 = -(alpha1 * c11 + alpha2 * c12);
                    
                    // C2: Global selection score (maximize)
                    let c2 = lambda * distance_matrix[0][insert_node] + c1;
        
                    // Update global best based strictly on C2 score
                    if best_c2.is_none() || c2 > best_c2.unwrap() {
                        best_c2 = Some(c2);
                        best = Some((insert_node, pos));
                    }
                }
            }
        }
        best
    }
    
    let mut routes = Vec::new();

    let mut nodes: Vec<usize> = (1..challenge.num_nodes).collect();
    nodes.sort_by(|&a, &b| challenge.distance_matrix[0][a].cmp(&challenge.distance_matrix[0][b]));

    let mut remaining: Vec<bool> = vec![true; challenge.num_nodes];
    remaining[0] = false;

    // popping furthest node from depot
    while let Some(node) = nodes.pop() {
        if !remaining[node] {
            continue;
        }
        remaining[node] = false;
        let mut route = vec![0, node, 0];
        let mut route_demand = challenge.demands[node];

        while let Some((best_node, best_pos)) = find_best_insertion(
            &route,
            remaining
                .iter()
                .enumerate()
                .filter(|(n, &flag)| {
                    flag && route_demand + challenge.demands[*n] <= challenge.max_capacity
                })
                .map(|(n, _)| n)
                .collect(),
            &challenge.distance_matrix,
            challenge.service_time,
            &challenge.ready_times,
            &challenge.due_times,
        ) {
            remaining[best_node] = false;
            route_demand += challenge.demands[best_node];
            route.insert(best_pos, best_node);
        }

        routes.push(route);
    }
    
    // EVOLVE-BLOCK-END
    
    save_solution(&Solution { routes })?;
    Ok(())
}

