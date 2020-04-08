from dataclasses import dataclass
from hepathlib import Path


@dataclass
class Project:
    base_dir: Path = Path(__file__).parents[0]
    data_dir: base_dir / 'dataset'
    checkpoint_dir : base_dir / 'checkpoint'

    def __phost_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)