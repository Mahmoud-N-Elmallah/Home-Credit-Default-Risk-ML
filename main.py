import yaml
from pathlib import Path
import sys
import argparse

# Ensure src module can be found
sys.path.append(str(Path(__file__).parent))
from src.data_processing.run_pipeline import run_pipeline
from src.model_training.run_training import run_training

def main():
    parser = argparse.ArgumentParser(description="Home Credit Default Risk Pipeline")
    parser.add_argument('--process', action='store_true', help="Run the data processing pipeline")
    parser.add_argument('--train', action='store_true', help="Run the model training pipeline")
    parser.add_argument('--all', action='store_true', help="Run both processing and training")
    args = parser.parse_args()

    if not any([args.process, args.train, args.all]):
        print("Please specify a step: --process, --train, or --all")
        return

    print("Loading config...")
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("Config file not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.process or args.all:
        print("\n--- Starting Data Processing Pipeline ---")
        run_pipeline(config)
        
    if args.train or args.all:
        print("\n--- Starting Model Training Pipeline ---")
        run_training(config)
        
    print("\nPipeline finished.")

if __name__ == "__main__":
    main()