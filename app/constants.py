INPUT_DIR = "data/pred_data"
OUTPUT_DIR = "predictions"
OUTPUT_FNAME = f"{OUTPUT_DIR}/current_predictions.csv"

BASE_MODEL = "esm2"
CHECKPOINT_MAPPING = {
    "zika_proteingym": "checkpoints/meta-transfer/esm2/A0A140D2T1_ZIKV_Sourisseau_2019/r16_ts40_cv4_cosine_mt3_GEMME",
    "zika_kikawa": "checkpoints/meta-transfer/esm2/A0A140D2T1_ZIKV_Kikawa/r16_ts40_cv4_cosine_mt3_kikawa",
}
