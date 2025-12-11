#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
import requests
import os
import shutil
import subprocess
import tempfile
import argparse

API_URL = "https://mainnet-api.tig.foundation"
CHALLENGES = ["satisfiability", "vehicle_routing", "knapsack"]

def download_contents(path: str, ref: str) -> dict:
    print(f"Fetching contents for {path} at ref {ref}")
    resp = requests.get(f"https://api.github.com/repos/tig-foundation/tig-monorepo/contents/{path}?ref={ref}")
    if resp.status_code != 200:
        raise Exception(f"Error fetching {path}': {resp.status_code} {resp.text}")
    ret = {}
    for x in resp.json():
        if x["type"] == "file":
            print(f"Downloading {x['download_url']}")
            r = requests.get(x["download_url"])
            if r.status_code != 200:
                raise Exception(f"Error downloading {x['download_url']}': {r.status_code} {r.text}")
            ret[x["path"].removeprefix(path + "/")] = r.text
        elif x["type"] == "dir":
            ret[x["path"].removeprefix(path + "/")] = download_contents(x["path"], ref)
        else:
            raise Exception(f"Unsupported type {x['type']} for {x['path']}")
    return ret

def write_contents(base_path: str, contents: dict):
    for name, content in contents.items():
        if isinstance(content, str):
            full_path = os.path.join(base_path, name)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            print(f"Writing file {full_path}")
            with open(full_path, "w") as f:
                f.write(content)
        elif isinstance(content, dict):
            write_contents(os.path.join(base_path, name), content)
        else:
            raise Exception(f"Unsupported content type for {name}: {type(content)}")

def reset_folder(path: str, force: bool = False):
    if os.path.exists(path):
        if os.listdir(path):
            if not force:
                raise Exception(f"Directory {path} is not empty. Use --force to overwrite.")
            print(f"Clearing folder {path}")
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def init_challenge(challenge: str, force: bool = False):
    base_dir = "algo-runner/src/challenge"
    reset_folder(base_dir, force)
    contents = download_contents(f"tig-challenges/src/{challenge}", "blank_slate")
    contents["mod.rs"] = contents["mod.rs"].replace(
        "use crate::QUALITY_PRECISION;",
        "use algo_runner::*;"
    )
    write_contents(base_dir, contents)

def build_algorithm():
    if not os.path.exists("algo-runner/src/challenge/mod.rs"):
        raise Exception("Challenge not initialized. Please run init_challenge first.")
    if not os.path.exists("algo-runner/src/algorithm/mod.rs"):
        raise Exception("No algorithm downloaded. Please run download_algorithm first.")
    result = subprocess.run(["cargo", "--version"])
    if result.returncode != 0:
        raise Exception("Cargo is not installed or not found in PATH. Please install Rust and Cargo from https://www.rust-lang.org/tools/install")
    subprocess.run(["cargo", "build", "--release"], cwd="algo-runner", check=True)

def download_algorithm(challenge: str, algorithm: str, force: bool = False):
    base_dir = "algo-runner/src/algorithm"
    reset_folder(base_dir, force)
    contents = download_contents(f"tig-algorithms/src/{challenge}/{algorithm}", f"{challenge}/{algorithm}")
    write_contents(base_dir, contents)

def list_algorithms(challenge: str) -> list[dict]:
    block = requests.get(f"{API_URL}/get-block").json()["block"]
    challenges = requests.get(f"{API_URL}/get-challenges?block_id={block['id']}").json()["challenges"]
    data = requests.get(f"{API_URL}/get-algorithms?block_id={block['id']}").json()
    algorithms = data["codes"]
    compile_success = {x['algorithm_id']: x['details']['compile_success'] for x in data['binarys']}
    c_id = next(
        (
            c['id'] for c in challenges
            if c['details']['name'] == challenge
        ),
        None
    )
    if c_id is None:
        raise Exception(f"Challenge '{challenge}' not found.")
    algorithms = sorted([
        a for a in algorithms if a['details']['challenge_id'] == c_id
    ], key=lambda x: x['id'])
    for a in algorithms:
        if a["id"] not in compile_success:
            status = f"pending compilation"
        elif not compile_success[a["id"]]:
            status = f"failed to compile"
        elif (pushed_in := a["state"]["round_pushed"] - block["details"]["round"]) > 0:
            status = f"pushed in {pushed_in} rounds"
        else:
            adoption = int((a["block_data"] or {}).get("adoption", 0)) / 1e16
            status = f"adoption {adoption:.2f}%"
        print(f"id: {a['id']:<12} name: {a['details']['name']:<25} status: {status}")
    return algorithms

def run_test(
    track_id: str,
    seed: int,
    hyperparameters: str = None,
    timeout: int = 60,
    debug: bool = False,
) -> tuple:
    try:
        quality, time, memory = None, None, None
        with tempfile.NamedTemporaryFile() as solution_file:
            cmd = [
                "/usr/bin/time",
                "-f", "Memory: %M",
                "target/release/algo-runner",
                "solve",
                track_id, # track_id
                str(seed), # seed
                solution_file.name,
            ]
            if hyperparameters:
                cmd += ["--hyperparameters", hyperparameters]
            if debug:
                print("Running command:", " ".join(cmd))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="algo-runner",
            )
            for line in result.stdout.strip().split("\n"):
                if line.startswith("Time:"):
                    time = float(line.split(":")[1].strip())
            for line in result.stderr.strip().split("\n"):
                if line.startswith("Memory:"):
                    memory = int(line.split(":")[1].strip())

            cmd = [
                "target/release/algo-runner",
                "eval",
                track_id, # track_id
                str(seed), # seed
                solution_file.name,
            ]
            if debug:
                print("Running command:", " ".join(cmd))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="algo-runner",
            )
            if result.returncode != 0:
                raise ValueError(f"Evaluation failed: {result.stderr.strip()}")
            for line in result.stdout.strip().split("\n"):
                if line.startswith("Quality:"):
                    quality = line.split(":")[1].strip()
        print(f"Seed: {seed}, Quality: {quality}, Time: {time}, Memory: {memory}KB")
        return quality, time, memory
    except Exception as e:
        print(f"Seed: {seed}, Error: {e}")
        return None, None, None

def test_algorithm(
    track_id: str,
    num_tests: int = 1,
    num_workers: int = 1,
    hyperparameters: str = None,
    timeout: int = 60,
    debug: bool = False,
) -> list:
    pool = ThreadPoolExecutor(max_workers=num_workers)
    return list(pool.map(
        lambda seed: run_test(track_id, seed, hyperparameters, timeout, debug),
        range(num_tests)
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Evolve CLI Tool")    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # init_challenge subcommand
    init_parser = subparsers.add_parser("init_challenge", help="Initialize a challenge")
    init_parser.add_argument("challenge", choices=CHALLENGES, help="Challenge name")
    init_parser.add_argument("--force", action="store_true", help="Force overwrite existing files")
    
    # download_algorithm subcommand
    download_parser = subparsers.add_parser("download_algorithm", help="Download an algorithm")
    download_parser.add_argument("challenge", choices=CHALLENGES, help="Challenge name")
    download_parser.add_argument("algorithm", help="Algorithm name")
    download_parser.add_argument("--force", action="store_true", help="Force overwrite existing files")
    
    # build_algorithm subcommand
    build_parser = subparsers.add_parser("build_algorithm", help="Build the algorithm")
    
    # list_algorithms subcommand
    list_parser = subparsers.add_parser("list_algorithms", help="List available algorithms")
    list_parser.add_argument("challenge", help="Challenge name")
    
    # test_algorithm subcommand
    test_parser = subparsers.add_parser("test_algorithm", help="Test the algorithm")
    test_parser.add_argument("track_id", help="Track ID")
    test_parser.add_argument("--tests", type=int, default=1, help="Number of tests to run")
    test_parser.add_argument("--workers", type=int, default=1, help="Number of worker threads")
    test_parser.add_argument("--hyperparameters", help="Hyperparameters string")
    test_parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    test_parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    if args.command == "init_challenge":
        init_challenge(args.challenge, args.force)
    elif args.command == "download_algorithm":
        download_algorithm(args.challenge, args.algorithm, args.force)
    elif args.command == "build_algorithm":
        build_algorithm()
    elif args.command == "list_algorithms":
        list_algorithms(args.challenge)
    elif args.command == "test_algorithm":
        test_algorithm(
            args.track_id,
            args.num_tests,
            args.num_workers,
            args.hyperparameters,
            args.timeout,
            args.debug
        )
    else:
        parser.print_help()