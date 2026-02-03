"""Run the sampling + model experiments and save results.

Usage:
    python run_experiment.py

This script expects the dataset CSV `Creditcard_data.csv` to be located
one level above the project folder (the workspace root). You can adjust
`DATA_PATH` if needed.
"""
from pathlib import Path
from src.data_loader import load_creditcard_data
from src.sampling import get_samplers
from src.models import get_models
from src.evaluation import run_experiments, save_results


# Default paths (assumes workspace layout used in this project)
PROJECT_DIR = Path(__file__).resolve().parent
# Prefer dataset in repo under data/raw; fall back to old location for backward compatibility.
DATA_PATH = PROJECT_DIR / "data" / "raw" / "Creditcard_data.csv"
if not DATA_PATH.exists():
    DATA_PATH = PROJECT_DIR / "Creditcard_data.csv"
RESULTS_PATH = PROJECT_DIR / "results" / "final_results.csv"


def main(data_path: Path = DATA_PATH, results_path: Path = RESULTS_PATH):
    print(f"Loading data from: {data_path}")
    X, y = load_creditcard_data(str(data_path))

    samplers = get_samplers()
    models = get_models()

    print("Running experiments...")
    results = run_experiments(X, y, samplers, models)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(results_path))

    print("Results saved to:", results_path)
    print(results)


if __name__ == "__main__":
    main()