from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402,F401
    experiment_path,
    expected_preprocessor_input_columns,
    load_input_frame,
    load_threshold,
    load_hydra_config,
    main,
    output_path,
    parse_args,
    prepare_features,
    resolve_path,
    run_inference,
)


if __name__ == "__main__":
    main()
