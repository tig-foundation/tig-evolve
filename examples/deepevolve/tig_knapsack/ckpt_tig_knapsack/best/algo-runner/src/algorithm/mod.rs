### >>> DEEPEVOLVE-BLOCK-START: Evolving a Genetic Algorithm with Interaction Scoring
use rand::seq::SliceRandom; // Import random choice for genetic operations
use rand::Rng; // For random number generation
use std::collections::HashSet;

// Genetic Algorithm Constants
const MAX_GENERATIONS: usize = 1000;
const POPULATION_SIZE: usize = 100;
const MUTATION_RATE: f64 = 0.1; // Mutation probability

/// Helper function to evaluate individuals in the population.
fn evaluate_individual(
    individual: &[usize],
    challenge: &Challenge,
    pos_interactions: &[i64],
) -> i64 {
    let mut total_value = 0;
    let mut total_weight = 0;

    for &item in individual {
        total_value += challenge.values[item] as i64;
        total_weight += challenge.weights[item];
    }

    // Calculate interaction values
    for i in 0..individual.len() {
        for j in (i + 1)..individual.len() {
            total_value += challenge.interaction_values[individual[i]][individual[j]] as i64;
        }
    }

    if total_weight <= challenge.max_weight as i64 {
        total_value
    } else {
        0 // Invalid solution
    }
}

/// Initialize a random population of solutions
fn initialize_population(challenge: &Challenge) -> Vec<Vec<usize>> {
    (0..POPULATION_SIZE)
        .map(|_| {
            let mut rng = rand::thread_rng();
            let mut items: Vec<usize> = (0..challenge.num_items).collect();
            items.shuffle(&mut rng);
            items.truncate(1 + rng.gen_range(0..challenge.num_items)); // Random number of items
            items
        })
        .collect()
}

/// Crossover function to create new offspring
fn crossover(parent1: &[usize], parent2: &[usize]) -> Vec<usize> {
    let cut = parent1.len() / 2;
    let mut child = parent1[..cut].to_vec();
    child.extend(parent2[cut..].iter().cloned());
    child.into_iter().collect::<HashSet<_>>().into_iter().collect() // Ensure uniqueness
}

/// Mutate an individual solution
fn mutate(individual: &mut Vec<usize>, challenge: &Challenge) {
    let mut rng = rand::thread_rng();
    if rng.gen::<f64>() < MUTATION_RATE {
        let swap_idx1 = rng.gen_range(0..individual.len());
        let swap_idx2 = rng.gen_range(0..individual.len());
        individual.swap(swap_idx1, swap_idx2);
    }
}

/// Main genetic algorithm function
fn genetic_algorithm(challenge: &Challenge) -> Vec<usize> {
    let pos_interactions: Vec<i64> = (0..challenge.num_items)
        .map(|i| {
            challenge.interaction_values[i]
                .iter()
                .filter(|&&v| v > 0)
                .map(|&v| v as i64)
                .sum::<i64>()
        })
        .collect();

    let mut population = initialize_population(challenge);
    let mut best_solution = Vec::new();
    let mut best_value = 0;

    for _ in 0..MAX_GENERATIONS {
        let scores: Vec<i64> = population
            .iter()
            .map(|individual| evaluate_individual(individual, challenge, &pos_interactions))
            .collect();

        // Selection based on scores
        population = population
            .iter()
            .enumerate()
            .filter(|&(i, _)| scores[i] > 0)
            .map(|(_, individual)| individual.clone())
            .collect();

        // Crossover and mutation
        population = (0..POPULATION_SIZE)
            .map(|_| {
                let (parent1, parent2) = population
                    .choose_multiple(&mut rand::thread_rng(), 2).unwrap(); // Random parents
                let mut child = crossover(parent1, parent2);
                mutate(&mut child, challenge);
                child
            })
            .collect();

        // Keep track of the best solution
        for (individual, &score) in population.iter().zip(scores.iter()) {
            if score > best_value {
                best_value = score;
                best_solution = individual.clone();
            }
        }
    }
    best_solution
}

/// Updated solve_challenge to integrate genetic algorithm instead of greedy
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let _params = hyperparameters.as_ref().unwrap_or(&Map::new());

    // Use genetic algorithm for challenge solving
    let selection = genetic_algorithm(challenge);

    let mut solution = Solution::new();
    solution.items = selection;
    save_solution(&solution)?;
    
    Ok(())
}
### <<< DEEPEVOLVE-BLOCK-END
}


