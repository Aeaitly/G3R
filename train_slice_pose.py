"""Run G3R reconstruction from user-supplied stacks and optional masks."""
# G3R modifications to the Apache-2.0 GSVR implementation. See LICENSE and NOTICE.

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "third_party" / "nesvor"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "g3r.yaml",
        help="YAML configuration (default: configs/g3r.yaml in this repository).",
    )
    parser.add_argument(
        "--exp_name", default=None, help="Override the output case name."
    )
    parser.add_argument(
        "--model", choices=("v1", "v2"),
        help="Override the model variant and select its matching pretrained checkpoint.",
    )
    args = parser.parse_args()
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    if args.model:
        config["svort"]["model_variant"] = args.model
        config["svort"]["checkpoint_path"] = f"pretrained/g3r_{args.model}.pt"

    from g3r_slice_pose import g3r

    output_root = Path(config["experiment"]["output_root"])
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    start = time.time()
    for subject in config["data"]["subjects"]:
        if not subject.get("enabled", True):
            continue
        stack_paths = subject["input_stacks"]
        if not stack_paths:
            raise ValueError("At least one input stack is required.")
        name = args.exp_name or subject["name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_root / name / f"{name}_{timestamp}.nii.gz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.with_suffix("").with_suffix(".yaml").open("w") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        print(f"Reconstructing {name} from {len(stack_paths)} stack(s).")
        print(f"Output directory: {output_path.parent}")
        g3r(
            stack_paths=stack_paths,
            mask_paths=subject.get("input_masks", []),
            config=config,
            output_file_path=str(output_path),
            subject=subject,
        )
    print(f"Total elapsed time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
