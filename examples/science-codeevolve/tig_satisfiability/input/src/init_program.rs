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

    // Randomly assign all variables
    let variables: Vec<bool> = (0..challenge.num_variables)
        .map(|_| rng.gen::<bool>())
        .collect();


    // EVOLVE-BLOCK-END

    let _ = save_solution(&Solution { variables  });
    Ok(())
}
