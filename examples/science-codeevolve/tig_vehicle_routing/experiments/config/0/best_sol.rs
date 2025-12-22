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
        pub alpha1: i32, // Distance weight
        pub alpha2: i32, // Time delay weight
        pub lambda: i32, // Depot proximity weight
    }

    // Helper trait to simplify option checks (used in the original code structure)
    trait OptionExt<T> {
        fn is_none_or<F>(&self, f: F) -> bool
        where
            F: FnOnce(T) -> bool,
            T: Copy + PartialOrd;
    }
    
    impl<T: Copy + PartialOrd> OptionExt<T> for Option<T> {
        fn is_none_or<F>(&self, f: F) -> bool
        where
            F: FnOnce(T) -> bool,
        {
            match self {
                None => true,
                Some(x) => f(*x),
            }
        }
    }

    // Structure to hold pre-calculated timing data for route nodes for O(1) feasibility check
    #[derive(Clone, Copy)]
    struct NodeTiming {
        service_start: i32, // S_i (Earliest possible service start time)
        latest_arrival: i32, // LA_i (Latest feasible arrival time to maintain downstream deadlines)
    }

    fn find_best_insertion(
        route: &Vec<usize>,
        remaining_nodes: Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Option<(usize, usize)> {
        // Parameters for VRPTW insertion heuristic (Solomon R2 style)
        let alpha1 = 1;
        let alpha2 = 5; // Time delay weight
        let lambda = 1; // Depot proximity weight
    
        let mut best_c2: Option<i32> = None;
        let mut best: Option<(usize, usize)> = None;

        let route_len = route.len();
        let mut timings: Vec<NodeTiming> = Vec::with_capacity(route_len);

        // 1. Forward Pass: Calculate S_i (Service Start Time) and D_i (Departure Time)
        // This calculates the original route timing profile.
        let mut current_time = 0; // Represents D_{i-1} (Departure from previous node)
        let mut current_node = 0;
        let mut departure_times = Vec::with_capacity(route_len);

        for &next_node in route.iter() {
            let arrival_time = current_time + distance_matrix[current_node][next_node];
            
            let service_start_time = arrival_time.max(ready_times[next_node]);
            
            // Calculate departure time D_i. Depot service time (node 0) is 0.
            let effective_service_time = if next_node == 0 { 0 } else { service_time };
            current_time = service_start_time + effective_service_time;
            
            timings.push(NodeTiming {
                service_start: service_start_time,
                latest_arrival: 0, // Placeholder
            });
            departure_times.push(current_time);
            current_node = next_node;
        }

        // 2. Backward Pass: Calculate LA_i (Latest Feasible Arrival Time)
        // This allows O(1) check for time window violations downstream.
        
        let mut latest_arrival_successor = due_times[0]; // LA for the final depot (node 0)
        let mut successor_node = 0;
        
        // Iterate backwards from the final depot (route_len - 1) to the starting depot (0)
        for i in (0..route_len).rev() {
            let node = route[i];
            
            if i == route_len - 1 {
                timings[i].latest_arrival = due_times[node];
            } else {
                // Effective service time at node i 
                let effective_service_time = if node == 0 { 0 } else { service_time };

                let latest_departure_i = latest_arrival_successor 
                    - distance_matrix[node][successor_node];
                
                // LA_i must satisfy LA_i + service_time <= Latest_Departure_i
                let latest_arrival_i = latest_departure_i - effective_service_time;
                
                // LA_i must also respect the node's own due time D_i
                let la_i = due_times[node].min(latest_arrival_i);
                
                timings[i].latest_arrival = la_i;
                latest_arrival_successor = la_i;
                successor_node = node;
            }
        }
        
        // 3. Search for the best insertion (u between i and j)
        
        for insert_node in remaining_nodes {
            // Iterate over all insertion positions (i, j) where i = route[pos - 1], j = route[pos]
            for pos in 1..route_len {
                let curr_node = route[pos - 1]; // Node i
                let next_node = route[pos];     // Node j
                
                let curr_time = departure_times[pos - 1]; // Departure time D_i
                
                let timing_j = timings[pos];

                // --- Step 3a: Local Feasibility Check (i -> u -> j) ---
                
                // A_u (Arrival at u)
                let arrival_u = curr_time + distance_matrix[curr_node][insert_node];
                
                // Check due time constraint at u 
                if arrival_u > due_times[insert_node] {
                    continue;
                }
                
                // D_u (Departure from u)
                let service_start_u = arrival_u.max(ready_times[insert_node]);
                let departure_u = service_start_u + service_time;

                // A_j^new (Arrival at j after insertion)
                let arrival_j_new = departure_u + distance_matrix[insert_node][next_node];
                
                // Check due time constraint at j 
                if arrival_j_new > due_times[next_node] {
                    continue;
                }
                
                // S_j^new (Service Start at j after insertion)
                let service_start_j_new = arrival_j_new.max(ready_times[next_node]);

                // S_j^old (Service Start at j before insertion)
                let service_start_j_old = timing_j.service_start;
    
                // --- Step 3b: Calculate Insertion Cost C1 ---
                
                // C11: Distance cost 
                let c11 = distance_matrix[curr_node][insert_node]
                    + distance_matrix[insert_node][next_node]
                    - distance_matrix[curr_node][next_node];
    
                // C12: Time delay cost 
                let c12 = service_start_j_new - service_start_j_old;
    
                // C1: Total Insertion Cost (Maximized)
                let c1 = -(alpha1 * c11 + alpha2 * c12);
                
                // C2: Composite cost (R2 style)
                let c2 = lambda * distance_matrix[0][insert_node] + c1; 
                
                // --- Step 3c: Check Global Feasibility (j to depot) ---
                
                // Time shift delta_t at node j:
                let delta_t = service_start_j_new - service_start_j_old;

                // Max allowed delay at j is LA_j - S_j^old.
                // If delta_t > max_allowed_delay, the time shift propagates and violates a downstream constraint.
                let max_allowed_delay = timing_j.latest_arrival - timing_j.service_start;
                
                if delta_t <= max_allowed_delay {
                    // Downstream feasibility maintained in O(1)
                    
                    // Optimization: Maximize C2 (primary goal)
                    if best_c2.is_none_or(|x| c2 > x) {
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

