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
            curr_time += distance_matrix[curr_node][next_node]; // Arrival time A_k
            
            // Service start time S_k = max(R_k, A_k)
            let service_start = curr_time.max(ready_times[next_node]);

            // Check constraint 5: S_k <= L_k (due_time, latest service start time)
            if service_start > due_times[next_node] {
                valid = false;
                break;
            }
            
            curr_time = service_start + service_time; // Departure time D_k
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
        // Parameters for Solomon's C1/C2 insertion criterion.
        // alpha1 weights distance cost (c11).
        // alpha2 weights time delay cost (c12).
        // lambda weights distance from depot (c2 criterion).
        let alpha1 = 1;
        let alpha2 = 10; // Increased weight for time delay, critical for VRPTW
        let lambda = 1; 
    
        let mut best_c2 = None;
        let mut best = None;
        for insert_node in remaining_nodes {
            let mut best_c1 = None;
    
            let mut curr_time = 0; // Departure time D_i
            let mut curr_node = 0; // i
            for pos in 1..route.len() {
                let next_node = route[pos]; // j

                // 2. Calculate state at j before insertion (S_j^old) for cost baseline and state update
                let arrival_j_old = curr_time + distance_matrix[curr_node][next_node];
                let service_start_j_old = ready_times[next_node].max(arrival_j_old);

                // 1. Calculate state for insertion node u
                let arrival_u = curr_time + distance_matrix[curr_node][insert_node];
                let service_start_u = ready_times[insert_node].max(arrival_u);

                let mut potential_best = None;

                // Check feasibility at u (S_u <= L_u)
                if service_start_u <= due_times[insert_node] {
                    let departure_u = service_start_u + service_time;

                    // 3. Calculate state at j after insertion (S_j^new)
                    let arrival_j_new = departure_u + distance_matrix[insert_node][next_node];
                    let service_start_j_new = ready_times[next_node].max(arrival_j_new);

                    // Distance criterion: c11 = d(i,u) + d(u,j) - d(i,j). Cost increase.
                    let c11 = distance_matrix[curr_node][insert_node]
                        + distance_matrix[insert_node][next_node]
                        - distance_matrix[curr_node][next_node];

                    // Time criterion: c12 = S_j^new - S_j^old. Delay caused at j.
                    let c12 = service_start_j_new - service_start_j_old;

                    let c1 = -(alpha1 * c11 + alpha2 * c12);
                    let c2 = lambda * distance_matrix[0][insert_node] + c1;

                    if best_c1.is_none_or(|x| c1 > x)
                        && best_c2.is_none_or(|x| c2 > x)
                        && is_feasible(
                            route,
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                            insert_node,
                            departure_u,
                            pos,
                        )
                    {
                        potential_best = Some((c1, c2));
                    }
                }
                
                // Update best overall if we found a better candidate in this position
                if let Some((c1, c2)) = potential_best {
                    best_c1 = Some(c1);
                    best_c2 = Some(c2);
                    best = Some((insert_node, pos));
                }

                // Update state for next iteration (i becomes j). 
                // We use the time window constraints of j to calculate D_j.
                curr_time = service_start_j_old + service_time;
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

