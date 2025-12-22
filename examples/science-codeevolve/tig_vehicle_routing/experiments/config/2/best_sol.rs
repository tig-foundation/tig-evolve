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

    // Helper trait for clearer option comparisons (used for tracking min/max costs)
    trait OptionExt<T> {
        fn is_none_or<F>(&self, f: F) -> bool
        where
            F: FnOnce(T) -> bool,
            T: Copy;
    }

    impl OptionExt<i32> for Option<i32> {
        fn is_none_or<F>(&self, f: F) -> bool
        where
            F: FnOnce(i32) -> bool,
        {
            match self {
                None => true,
                Some(&x) => f(x),
            }
        }
    }

    // Calculates the Earliest Arrival Time (EAT) and Earliest Departure Time (EDT) for all nodes 
    // in the route (forward pass feasibility check).
    // Returns None if the route is infeasible due to time windows.
    fn calculate_route_timing(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Option<(Vec<i32>, Vec<i32>)> {
        // Returns (arrival_times, departure_times) indexed by route position
        let N = route.len();
        let mut arrival_times = vec![0; N];
        let mut departure_times = vec![0; N];
        
        let mut curr_time = 0; // Departure time from previous node
        let mut curr_node = 0; // Previous node index
        
        // Node 0 (Depot departure, index 0)
        departure_times[0] = 0;

        for i in 1..N {
            let next_node = route[i];
            
            // Arrival time
            curr_time += distance_matrix[curr_node][next_node];
            arrival_times[i] = curr_time;
            
            // Check due time (Constraint 5 & 8)
            if curr_time > due_times[next_node] {
                return None;
            }
            
            // Departure time (Constraint 6 & 7)
            curr_time = curr_time.max(ready_times[next_node]) + service_time;
            departure_times[i] = curr_time;
            
            curr_node = next_node;
        }
        
        Some((arrival_times, departure_times))
    }

    // Checks feasibility of inserting node U between route[pos-1] (i) and route[pos] (j).
    // Uses the precalculated timing profile of the original route and checks for delay propagation.
    fn check_insertion_feasibility(
        route: &Vec<usize>,
        u: usize,           // Node to insert
        pos: usize,         // Insertion index (where j is currently)
        timing: &(Vec<i32>, Vec<i32>), // (arrival_times, departure_times) of current route
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> bool {
        let (arrival_times, departure_times) = timing;
        
        let i = route[pos - 1]; // Predecessor node index
        let j = route[pos];     // Successor node index
        
        let d_i = departure_times[pos - 1]; // Departure time from i (original route)
        let d_j = departure_times[pos];     // Original departure time from j (original route)
        
        // 1. Check feasibility at U (i -> u)
        let a_u = d_i + distance_matrix[i][u];
        if a_u > due_times[u] {
            return false;
        }
        let d_u = a_u.max(ready_times[u]) + service_time;
        
        // 2. Check feasibility at J (u -> j)
        let a_j_prime = d_u + distance_matrix[u][j];
        
        if a_j_prime > due_times[j] {
            return false;
        }
        
        let d_j_prime = a_j_prime.max(ready_times[j]) + service_time;
        
        // 3. Check downstream propagation
        let mut delay = d_j_prime - d_j;
        
        if delay <= 0 {
            return true; // No positive delay introduced
        }
        
        // Positive delay introduced. Propagate check. O(L) worst case, but often faster due to early exit.
        let N = route.len();
        for k_idx in (pos + 1)..N {
            let k = route[k_idx];
            
            // New arrival time at k: A_k' = A_k + Delay_{k-1}
            let a_k_prime = arrival_times[k_idx] + delay;
            
            if a_k_prime > due_times[k] {
                return false;
            }
            
            // Original departure time D_k
            let d_k = departure_times[k_idx];
            
            // New departure time D_k'
            let d_k_prime = a_k_prime.max(ready_times[k]) + service_time;
            
            // New delay introduced at k
            delay = d_k_prime - d_k;
            
            if delay <= 0 {
                return true; // Delay absorbed
            }
        }
        
        true
    }
    
    // Uses minimum added distance (C11) as the primary insertion criterion among feasible spots.
    fn find_best_insertion(
        route: &Vec<usize>,
        remaining_nodes: Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Option<(usize, usize)> {
        
        // Calculate timing profile for the current route O(L)
        let Some(timing) = calculate_route_timing(
            route,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) else {
            // If the base route itself is somehow infeasible (shouldn't happen during construction), stop.
            return None; 
        };
        
        // Track minimum added distance C11
        let mut min_added_distance = None;
        let mut best_insertion = None;
        
        let route_len = route.len();

        for insert_node in remaining_nodes {
            // Iterate through all possible insertion points (i, j), where i = route[pos-1], j = route[pos]
            for pos in 1..route_len {
                let i = route[pos - 1]; 
                let j = route[pos];     

                // Calculate added distance C11 = d(i,u) + d(u,j) - d(i,j)
                let added_distance = distance_matrix[i][insert_node]
                    + distance_matrix[insert_node][j]
                    - distance_matrix[i][j];

                // Check time window feasibility using the precalculated timing profile
                let feasible = check_insertion_feasibility(
                    route,
                    insert_node,
                    pos,
                    &timing,
                    distance_matrix,
                    service_time,
                    ready_times,
                    due_times,
                );
                
                // 3. If feasible, update best insertion based on minimum added distance
                if feasible {
                    // Use OptionExt::is_none_or to check if current added_distance is better (smaller)
                    if min_added_distance.is_none_or(|x| added_distance < x) {
                        min_added_distance = Some(added_distance);
                        best_insertion = Some((insert_node, pos));
                    }
                }
            }
        }
        best_insertion
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

