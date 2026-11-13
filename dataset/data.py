"""Load the stack and mask paths explicitly supplied in the configuration."""
# G3R modifications to the Apache-2.0 GSVR implementation. See LICENSE and NOTICE.

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "nesvor"))

from nesvor.image import load_stack


class Subject:
    def __init__(self, name, input_stacks, input_masks=None, device="cpu"):
        self.name = name
        self.file_list = input_stacks
        self.mask_list = input_masks or []
        self.device = device
        self._input_stacks = None

    @property
    def input_stacks(self):
        if self._input_stacks is None:
            self._input_stacks = []
            for index, path in enumerate(self.file_list):
                mask = self.mask_list[index] if index < len(self.mask_list) else None
                self._input_stacks.append(load_stack(path, mask, device=self.device))
        return self._input_stacks
