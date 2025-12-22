// DO NOT CHANGE THESE IMPORTS
use crate::challenge::{Challenge, Solution};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::collections::HashSet;

pub fn help() {
    // Print help information about your algorithm here. It will be invoked with `help_algorithm` script
    println!("Initial program");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // EVOLVE-BLOCK-START
    // Save initial solution
    let _ = save_solution(&Solution {
        variables: vec![false; challenge.num_variables],
    });

    // Deterministic RNG from the challenge seed
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(
        challenge.seed[..8].try_into().unwrap(),
    ));

    // Randomly assign initial variables
    let mut variables: Vec<bool> = (0..challenge.num_variables)
        .map(|_| rng.gen::<bool>())
        .collect();

    // Hill climbing optimization
    let mut best_variables = variables.clone();
    let mut best_satisfied_clauses = count_satisfied_clauses(&challenge.clauses, &variables);

    for _ in 0..1000 { // Iterate for a fixed number of steps
        let mut current_variables = best_variables.clone();
        let variable_index = rng.gen_range(0..challenge.num_variables);
        current_variables[variable_index] = !current_variables[variable_index]; // Flip a bit

        let satisfied_clauses = count_satisfied_clauses(&challenge.clauses, &current_variables);

        if satisfied_clauses > best_satisfied_clauses {
            best_satisfied_clauses = satisfied_clauses;
            best_variables = current_variables.clone();
            let _ = save_solution(&Solution { variables: best_variables.clone() }); // Save improved solution
        }
    }

    variables = best_variables;

fn count_satisfied_clauses(clauses: &Vec<Vec<i32>>, variables: &Vec<bool>) -> usize {
    clauses
        .iter()
        .filter(|clause| {
            clause.iter().any(|literal| {
                let variable_index = literal.abs() as usize - 1;
                let variable_value = variables[variable_index];
                (literal > &0 && variable_value) || (literal < &0 && !variable_value)
            })
        })
        .count()
}


    // EVOLVE-BLOCK-END

    let _ = save_solution(&Solution { variables  });
    Ok(())
}
