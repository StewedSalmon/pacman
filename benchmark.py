import sys
import os
import csv
import argparse
from pathlib import Path
import statistics


# Point Python to the untouched src directory
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now we can import the original framework classes
from arena import Arena

def run_native_benchmark(pacman_id, ghost_id, num_runs=50, save_csv=False):
    print(f"Starting {num_runs} native runs: [{pacman_id}] vs [{ghost_id}]...\n")
    
    wins = 0
    losses = 0
    draws = 0
    steps_list = []

    # Optional: Suppress standard print statements from the framework to keep the console clean
    # old_stdout = sys.stdout
    # sys.stdout = open(os.devnull, 'w')

    for i in range(num_runs):
        # CRITICAL: Create a completely fresh Arena object every single loop.
        # This acts as our "memory wipe" so agents don't share state between games.
        arena = Arena(
            pacman_id=pacman_id,
            ghost_id=ghost_id,
            submissions_dir="submissions", # Point to your submissions folder
            max_steps=200,
            visualize=True,               # Disable visualizer for speed
            delay=0.0,
            capture_distance_threshold=2,
            deterministic_starts=True,    
            pacman_speed=2                 # Match your agent's speed
        )
        
        # Load fresh agent instances
        arena.load_agents()
        
        # Run the game and grab the returned stats dictionary directly
        result, stats = arena.run_game()
        
        # Tally the results
        if result == 'pacman_wins':
            wins += 1
        elif result == 'ghost_wins':
            losses += 1
        else:
            draws += 1
            
        steps_list.append(stats['total_steps'])

    # Restore console printing if you suppressed it
    # sys.stdout = old_stdout

    # Calculate statistics
    win_rate = (wins / num_runs) * 100
    avg_steps = statistics.mean(steps_list) if steps_list else 0
    std_dev = statistics.stdev(steps_list) if len(steps_list) > 1 else 0

    # print("\n" + "="*40)
    # print("🏆 NATIVE BENCHMARK RESULTS 🏆")
    # print("="*40)
    # print(f"Pacman Win Rate: {win_rate:.1f}% ({wins} W / {losses} L / {draws} D)")
    # print(f"Average Steps to Catch: {avg_steps:.1f}")
    # print(f"Step Variance (Std Dev): ±{std_dev:.1f}")
    # print("="*40)
    
    # Export to CSV if requested
    if save_csv:
        file_exists = os.path.isfile('benchmark_results.csv')
        with open('benchmark_results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write headers if the file is brand new
                writer.writerow(['Pacman', 'Ghost', 'Runs', 'Win Rate (%)', 'Wins', 'Losses', 'Draws', 'Avg Steps', 'Std Dev'])
            writer.writerow([pacman_id, ghost_id, num_runs, round(win_rate, 2), wins, losses, draws, round(avg_steps, 2), round(std_dev, 2)])
        print("\n Results saved to benchmark_results.csv")

if __name__ == "__main__":
    # Set up argparse to handle command line inputs
    parser = argparse.ArgumentParser(description="Run automated benchmarks for Pacman vs Ghost.")
    parser.add_argument('--seek', type=str, required=True, help="Folder name of the Pacman agent.")
    parser.add_argument('--hide', type=str, required=True, help="Folder name of the Ghost agent.")
    parser.add_argument('--runs', type=int, default=50, help="Number of games to simulate (default: 50).")
    parser.add_argument('--save', action='store_true', help="Flag to save results to a CSV file.")
    
    args = parser.parse_args()

    run_native_benchmark(
        pacman_id=args.seek, 
        ghost_id=args.hide, 
        num_runs=args.runs,
        save_csv=args.save
    )