from typing import Any

import torch
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

import app.constants as app_constants


class MutantRequestData(BaseModel):
    target_protein_id: str
    mutant: list[str]
    mutated_sequence: list[str]
    checkpoint: str = "zika_proteingym"
    task_name: str = "zika"
    device: str = "cuda"


    @model_validator(mode="after")
    def check_input_lists(self) -> Self:
        if not all((self.mutant, self.mutated_sequence)):
            raise ValueError("Mutant and sequence lists cannot be empty")
        
        if not all(self.mutant) or not all(self.mutated_sequence):
            raise ValueError("Mutant and sequence lists cannot contain empty values")
            
        if len(self.mutant) != len(self.mutated_sequence):
            raise ValueError(
                f"Mutants and sequences lists must have the same length, got {len(self.mutant)} and {len(self.mutated_sequence)}"
            )
        return self
    
    @field_validator("target_protein_id", mode="before")
    @classmethod
    def check_target_protein_not_empty(cls, v) -> Any:
        if not v:
            raise ValueError("Target protein cannot be empty")
        return v
    
    @field_validator("checkpoint", mode="before")
    @classmethod
    def validate_checkpoint(cls, v) -> Any:
        if not v in app_constants.CHECKPOINT_MAPPING.keys():
            raise ValueError(f"Got unsupported checkpoint: {v}")
        return v
    
    @field_validator("device", mode="before")
    @classmethod
    def validate_checkpoint(cls, v) -> Any:
        if not v in {"cuda", "cpu"}:
            raise ValueError(f"Device must be one of: cuda, cpu; got: {v}")
        if v == "cuda" and not torch.cuda.is_available():
            raise ValueError("GPU requested but not available")
        return v
