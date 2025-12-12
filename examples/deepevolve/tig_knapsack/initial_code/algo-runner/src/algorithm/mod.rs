// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::challenge::*;
use anyhow::{Result, anyhow};
use serde_json::{Map, Value};

/// Simple greedy seed: rank items by (value + 0.5 * positive interaction sum) / weight.
/// This is intentionally lightweight so DeepEvolve can iterate and improve it.
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let _params = hyperparameters.as_ref().unwrap_or(&Map::new());

    let n = challenge.num_items;
    if n == 0 {
        return Err(anyhow!("Empty challenge"));
    }

    // Precompute positive interaction contributions per item (approximation).
    let mut pos_interactions: Vec<i64> = Vec::with_capacity(n);
    for i in 0..n {
        let sum = challenge.interaction_values[i]
            .iter()
            .filter(|&&v| v > 0)
            .map(|&v| v as i64)
            .sum::<i64>();
        pos_interactions.push(sum);
    }

    // Rank items by approximate value density.
    let mut ranked: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let weight = challenge.weights[i].max(1) as f64;
            let approx_value = challenge.values[i] as f64 + 0.5 * pos_interactions[i] as f64;
            let ratio = approx_value / weight;
            (i, ratio)
        })
        .collect();

    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selection = Vec::new();
    let mut total_weight: u32 = 0;

    for (idx, _) in ranked {
        let w = challenge.weights[idx];
        if total_weight + w <= challenge.max_weight {
            total_weight += w;
            selection.push(idx);
        }
    }

    let mut solution = Solution::new();
    solution.items = selection;
    save_solution(&solution)
}


