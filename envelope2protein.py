import argparse
import pandas as pd

from constants import ENVELOPE_INDEX_LEFT, ENVELOPE_INDEX_RIGHT, ZIKA_ENVELOPE, ZIKA_PROTEIN


ZIKA_PROTEIN_LEFT = ZIKA_PROTEIN[:ENVELOPE_INDEX_LEFT]
ZIKA_PROTEIN_RIGHT = ZIKA_PROTEIN[ENVELOPE_INDEX_RIGHT:]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i",
                        help="Path to input CSV file containing mutated envelope sequences in `mutated_sequence` column")
    return parser.parse_args()


def paste_envelope_into_zika_protein(envelope: str) -> str:
    """A helper function just in case it's needed"""
    result = ZIKA_PROTEIN_LEFT + envelope + ZIKA_PROTEIN_RIGHT
    assert len(result) == len(ZIKA_PROTEIN)
    return result

def main(input_file: str) -> None:
    df = pd.read_csv(input_file)

    assert df["mutated_sequence"].str.len().nunique() == 1, "Mutated sequences have varied length!"
    mutated_seq_len = df["mutated_sequence"].str.len().unique()[0]
    assert mutated_seq_len == len(ZIKA_ENVELOPE), f"Mutated sequences have wrong length: required {len(ZIKA_ENVELOPE)}, got {mutated_seq_len}"

    df["mutated_sequence"] = ZIKA_PROTEIN_LEFT + df["mutated_sequence"] + ZIKA_PROTEIN_RIGHT
    df.to_csv(args.input_file, index=False)


# def shift_mutant_code(mutant_code: str) -> str:
#     before, site, after = mutant_code[0], int(mutant_code[1:-1]), mutant_code[-1]
#     return before + str(site + ENVELOPE_INDEX_LEFT) + after

if __name__ == "__main__":
    args = parse_args()
    main(args.input_file)
