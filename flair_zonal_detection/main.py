import argparse
import sys
import os
# Add the project root folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from flair_zonal_detection.inference import run_inference

def main() -> None:
    parser = argparse.ArgumentParser(description="Run zonal detection inference.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the detection config file"
    )
    args = parser.parse_args()
    run_inference(args.config)

if __name__ == '__main__':
    main()
