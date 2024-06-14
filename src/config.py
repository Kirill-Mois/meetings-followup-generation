import yaml
from dataclasses import dataclass

# constants

@dataclass
class SummarizerConfig:
    chain_name: str
    map_template: str
    reduce_template: str
    question_template: str
    refine_template: str

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, 'r') as f:
            return SummarizerConfig(**yaml.load(f))