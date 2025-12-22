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

    // Helper trait to provide comparison on Option types
    trait OptionExt<T> {
        fn is_none_or<F>(&self, f: F) -> bool
        where
            F: FnOnce(&T) -> bool;
    }

    impl<T> OptionExt<T> for Option<T> {
        fn is_none_or<F>(&self, f: F) -> bool
        where
            F: FnOnce(&T) -> bool,
        {
            match self {
                None => true,
                Some(x) => f(x),
            }
        }
    }

    fn is_feasible(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
        mut curr_node: usize,
        mut curr_time: i32,
        start_pos: usize,
    ) -> bool {
        let mut valid = true;
        for pos in start_pos..route.len() {
            let next_node = route[pos];
            curr_time += distance_matrix[curr_node][next_node];
            if curr_time > due_times[route[pos]] {
                valid = false;
                break;
            }
            curr_time = curr_time.max(ready_times[next_node]) + service_time;
            curr_node = next_node;
        }
        valid
    }
    
    fn find_best_insertion(
        route: &Vec<usize>,
        remaining_nodes: Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Option<(usize, usize)> {
        let alpha1 = 1;
        let alpha2 = 0;
        let lambda = 1;
    
        let mut best_c1 = None; // Moved initialization outside the loop
        let mut best_c2 = None;
        let mut best = None;
        for insert_node in remaining_nodes {
            let mut curr_time = 0;
            let mut curr_node = 0;
            for pos in 1..route.len() {
                let next_node = route[pos];
                let new_arrival_time =
                    ready_times[insert_node].max(curr_time + distance_matrix[curr_node][insert_node]);
                if new_arrival_time > due_times[insert_node] {
                    continue;
                }
                // T_start(j | i -> j)
                let old_start_time_j =
                    ready_times[next_node].max(curr_time + distance_matrix[curr_node][next_node]);

                // T_end(u)
                let end_time_u = new_arrival_time + service_time;

                // T_start(j | i -> u -> j)
                let new_start_time_j =
                    ready_times[next_node].max(end_time_u + distance_matrix[insert_node][next_node]);
    
                // Distance criterion: c11 = d(i,u) + d(u,j) - d(i,j)
                let c11 = distance_matrix[curr_node][insert_node]
                    + distance_matrix[insert_node][next_node]
                    - distance_matrix[curr_node][next_node];
    
                // Time criterion: c12 = T_start(j | new) - T_start(j | old). Measures the delay propagation.
                let c12 = new_start_time_j - old_start_time_j;
    
                let c1 = -(alpha1 * c11 + alpha2 * c12);
                let c2 = lambda * distance_matrix[0][insert_node] + c1;
    
                let update = match (best_c2, best_c1) {
                    (None, _) => true,
                    (Some(bc2), Some(bc1)) => {
                        // Maximize C2 (Primary), then C1 (Tie-breaker)
                        c2 > bc2 || (c2 == bc2 && c1 > bc1)
                    }
                    // Should not happen as C1 and C2 are updated atomically
                    _ => false, 
                };

                if update
                    && is_feasible(
                        route,
                        distance_matrix,
                        service_time,
                        ready_times,
                        due_times,
                        insert_node,
                        new_arrival_time + service_time,
                        pos,
                    )
                {
                    best_c1 = Some(c1);
                    best_c2 = Some(c2);
                    best = Some((insert_node, pos));
                }
    
                curr_time = ready_times[next_node]
                    .max(curr_time + distance_matrix[curr_node][next_node])
                    + service_time;
                curr_node = next_node;
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

