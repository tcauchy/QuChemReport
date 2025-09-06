import argparse
import traceback
import sys

import quchemreport.execution_engine.engine as engine
from quchemreport.config.YAML_loader import load_yaml_config

def main():
    parser = argparse.ArgumentParser(description="Command-line interface for running quchemreport.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        config = load_yaml_config(args.config)
    except Exception as e:
        print(e)
        sys.exit(1)
    
    try:
        engine.run(config)
    except Exception as e:
        print("engine error: ",e)
        if config.logging.level == "debug":
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

