This very simple API provides a way to run predictions of fine-tuned Pro-FSFP models. It has a single endpoint that accepts a JSON payload and returns a prediction.

## Predict

By default, the app runs on port 8080 and provides a single endpoint `/predict` to run predictions.

A POST request to `/predict` should be made with a JSON payload structured as follows:

```json
{
  "target_protein_id": "string",
  "mutant": ["string"],
  "mutated_sequence": ["string"],
  "checkpoint": "string",
  "task_name": "string",
  "device": "string",
}
```

Here's a brief explanation of each field:

- `target_protein_id`: A string representing the target protein; should be a Uniprot ID; currently it is set to `A0A140D2T1_ZIKV` by default, as the inference endpoint only supports a Zika model. Note that **you don't need to specify this if the default is used**.
- `mutant`: An array of strings representing the mutants. Each element should either be a single mutation in ProteinGym 1-indexed mutation format, like `M1T`, or multiple mutations in the same format separated by colons, like `M1T:A42I`.
- `mutated_sequence`: An array of strings representing the mutated amino-acid sequences.
- `checkpoint`: A string specifying the checkpoint to use (default is "zika_proteingym", trained on data from Sourisseau et al./ProteinGym; there is a less reliable checkpoint called "zika_kikawa", trained on additional data from Kikawa et al.). Note that you should generally use the default checkpoint, so no need to specify that field in your POST request.
- `task_name`: A string specifying the task name (default is "zika"). Task name will be used in file naming for prediction outputs and generally can be left as is.
- `device`: A string specifying the device to run predictions on (default is "cpu"). Must be one of: "cpu" or "cuda". Note that **if `cuda` is specified but no GPU is available to the inference container, the predictions will run on CPU instead and a warning will be logged.

**Constraints:**
The `mutant` and `mutated_sequence` arrays must have the same length and cannot be empty. Very important: `mutated_sequence` should have the length of Zika envelope, as it will be pasted into the full Zika reference protein at appropriate positions for inference. The `target_protein_id` cannot be empty and for now should generally be set to `A0A140D2T1_ZIKV`, and `checkpoint` must be one of the supported values (`zika_proteingym`, `zika_kikawa`).

**Returns:**
The prediction endpoint returns a JSON response with the following structure:

```json
{
  "message": "Prediction pipeline completed successfully.",
  "mutant": ["M1T", "A42I", "M1T:A42I"],
  "prediction": [0.75, 0.82, 0.68]
}
```

In this structure:

- `message` provides a status update on the prediction process.
- `mutant` is an array of the input mutants, matching the input exactly.
- `prediction` is an array of corresponding prediction scores, one score for each mutant.

**Code snippet** making a request to the app running on localhost, illustrating the typical usage:

```python
import requests

payload = {
    "mutant": ["I291A", "I291Y"],
    "mutated_sequence": ["ARCIGVSNRDFVEGMSGGTWVDVVLEHGGCVTVMAQDKPTVDIELVTTTVSNMAEVRSYCYEASISDMASDSRCPTQGEAYLDKQSDTQYVCKRTLVDRGWGNGCGLFGKGSLVTCAKFTCSKKMTGKSIQPENLEYRIMLSVHGSQHSGMIVNDTGYETDENRAKVEVTPNSPRAEATLGGFGSLGLDCEPRTGLDFSDLYYLTMNNKHWLVHKEWFHDIPLPWHAGADTGTPHWNNKEALVEFKDAHAKRQTVVVLGSQEGAVHTALAGALEAEMDGAKGKLFSGHLKCRLKMDKLRLKGVSYSLCTAAFTFTKVPAETLHGTVTVEVQYAGTDGPCKIPVQMAVDMQTLTPVGRLITANPVITESTENSKMMLELDPPFGDSYIVIGVGDKKITHHWHRSG", "YRCIGVSNRDFVEGMSGGTWVDVVLEHGGCVTVMAQDKPTVDIELVTTTVSNMAEVRSYCYEASISDMASDSRCPTQGEAYLDKQSDTQYVCKRTLVDRGWGNGCGLFGKGSLVTCAKFTCSKKMTGKSIQPENLEYRIMLSVHGSQHSGMIVNDTGYETDENRAKVEVTPNSPRAEATLGGFGSLGLDCEPRTGLDFSDLYYLTMNNKHWLVHKEWFHDIPLPWHAGADTGTPHWNNKEALVEFKDAHAKRQTVVVLGSQEGAVHTALAGALEAEMDGAKGKLFSGHLKCRLKMDKLRLKGVSYSLCTAAFTFTKVPAETLHGTVTVEVQYAGTDGPCKIPVQMAVDMQTLTPVGRLITANPVITESTENSKMMLELDPPFGDSYIVIGVGDKKITHHWHRSG"],
    "devide": "cuda",
}
response = requests.post("http://localhost:8080/predict", json=payload)
```
