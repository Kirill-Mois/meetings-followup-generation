import yaml
from dataclasses import dataclass

# constants

@dataclass
class SummarizerConfig:
    chunk_size: int = 20000
    chunk_overlap: int = 4000
    breakpoint_threshold_type: str = "standard_deviation"
    chain_name: str
    map_template: str
    reduce_template: str
    question_template: str
    refine_template: str

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, 'r') as f:
            return SummarizerConfig(**yaml.load(f))