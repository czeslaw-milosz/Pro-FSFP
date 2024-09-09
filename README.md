# How to run predictions on new data (RJ)

0. You will need to setup an environment with Python 3.10. When inside the environment, execute the following steps (**Note** that this configuration will only work for inference and not for model fitting; training is a very different matter due to code quality issues and package conflicts):
 - `git clone https://github.com/czeslaw-milosz/Pro-FSFP.git`
 - from `Pro-FSFP`, run `pip install -r requirements.txt`
 - (optional) if you run into torch trouble when running predictions, `pip uninstall torch_xla` might help you.

1. Download `checkpoints.zip` (130MB) from S3 bucket `zika-tmp` and extract it to `Pro-FSFP/checkpoints`.

2. Prepare your mutants file in CSV; see `data/demo` for an example of the correct data format. Place your csv file under `Pro-FSFP/data/pred_data`. **Please ensure** you filename starts with `A0A140D2T1_ZIKV` (a not-very-clever requirement of the original codebase).

3. **Switch from envelope to full protein format** by running
```python envelope2protein --input_file {YOUR_INPUT_CSV}```
(this will convert your csv file in-place from envelope format to full protein format)

4. Run `python preprocess.py`. Make sure a file called `merged.pkl` appeared in `data/`.

5. To run predictions on new data, cd to `Pro-FSFP` and run:

**For the model trained on Sourisseau et al. 2019 data from ProteinGym (more reliable):**
```bash
python main.py -ckpt checkpoints/meta/esm2/A0A140D2T1_ZIKV_Sourisseau_2019/r16_ts40_cv4_cosine_mt3_GEMME --model esm2 --protein A0A140D2T1_ZIKV --predict
```
You will find results in the `predictions/` directory.

**For the model trained on Kikawa et al. data (potentially not reliable at all, use with caution):**
```bash
python main.py -ckpt checkpoints/meta-transfer/esm2/A0A140D2T1_ZIKV_Kikawa/r16_ts40_cv4_cosine_mt3_kikawa --model esm2 --protein A0A140D2T1_ZIKV --save_prefix kikawa --predict
```
You will find results in the `predictions/` directory.

------------------------------------------
------------------------------------------

# Pro-FSFP: Few-Shot Protein Fitness Prediction
Supported PLMs: **ESM-1b, ESM-1v, ESM-2, and SaProt**

## Requirements
### Software
The code has been tested on Windows 10 and Ubuntu 22.04.3 LTS, with Anaconda3. The package dependencies are listed as follows:
```
cudatoolkit 11.8.0
learn2learn 0.2.0
pandas 1.5.3
peft 0.4.0
python 3.10
pytorch 2.0.1
scipy 1.10.1
scikit-learn 1.3.0
tqdm 4.65.0
transformers 4.29.2
```
### Hardware
The code has been tested on RTX 3090 GPU.
### Installation
- Install transformers and peft according to [HuggingFace](https://huggingface.co/docs)
- Install the gpu version of pytorch according to [Pytorch](https://pytorch.org/get-started/locally/)
- Install learn2learn according to [learn2learn](https://learn2learn.net/tutorials/getting_started/)
- Other packages can be easily installed by `conda install xxx`
- The installation should finish in 10-20 minutes.

## Config file
The config file `fsfp/config.json` defines the paths of model checkpoints, input and output.

## Data preprocessing
The datasets of ProteinGym should be put under `data/substitutions/`. Run `python preprocess.py -s` to preprocess the raw datasets and pack them to `data/merged.pkl`.

## Search for similar datasets for meta-training (not necessary if using LTR only)
- Run `python retrieve.py -m vectorize -md esm2` to compute and cache the embedding vectors of the proteins in ProteinGym, using ESM-2 for example.
- Run `python retrieve.py -m retrieve -md esm2 -b 16 -k 71 -mt cosine -cpu` to measure and save the similarities between proteins from the cached vectors.

## Training and inference
Run `main.py` for model training and inference. The default hyper-parameters may not be optimal, so it is recommended to perform hyper-parameter search for each protein via cross-validation.
Important hyper-parmeters are listed as follows (abbreviations in parentheses):
- --mode (-m): perform LTR finetuning, meta-learning or transfer learning using the mear-learned model
- --test (-t): whether to load the trained models from checkpoints and test them
- --model (-md): name of the PLM to train
- --protein (-p): name of the target protein (UniProt ID)
- --train_size (-ts): few-shot training set size, can be a float number less than 1 to indicate a proportion
- --train_batch (-tb): batch size for training (outer batch size in the case of meta-learning)
- --eval_batch (-eb): batch size for evaluation
- --lora_r (-r): hyper-parameter r of LORA
- --optimizer (-o): optimizer for training (outer loop optimization in the case of meta-learning)
- --learning_rate (-lr): learning rate
- --epochs (-e): maximum training epochs
- --max_grad_norm (-gn): maximum gradient norm to clip to
- --list_size (-ls): list size for ranking
- --max_iter (-mi): maximum number of iterations per training epoch, useless during meta-training
- --eval_metric (-em): evaluation metric
- --augment (-a): specify one or more models to use their zero-shot scores for data augmentation
- --meta_tasks (-mt): number of tasks used for meta-training
- --meta_train_batch (-mtb): inner batch size for meta-training
- --meta_eval_batch (-meb): inner batch size for meta-testing
- --adapt_lr (-alr): learning rate for inner loop during meta-learning
- --patience (-pt): number of epochs to wait until the validation score improves
- --cross_validation (-cv): number of splits for cross validation (shuffle & split) on the training set
- --force_cpu (-cpu): use cpu for training and evaluation even if gpu is available

## Run the benchmark using a single script
Put the csv files of ProteinGym to `data/substitutions/`, go to the root directory of this project, and then simply run `run.sh`. This will automatically benchmark ESM-2 (FSFP) on all 87 datasets in ProteinGym, with the training size of 40.

### Demo
- Use LTR and LoRA to train PLMs for specific protein (SYUA_HUMAN for example) without meta-learning: <br>
`python main.py -md esm2 -m finetune -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -p SYUA_HUMAN`. This may take several minutes, and the trained model will be saved to `checkpoints/finetune`.
- Test the trained model, print results, and save predictions: <br>
`python main.py -md esm2 -m finetune -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -p SYUA_HUMAN -t`. This may take a few seconds, and the predictions will be saved to `predictions/`.
- Meta-train PLMs on the auxiliary tasks: <br>
`python main.py -md esm2 -m meta -ts 40 -tb 1 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -a GEMME -p SYUA_HUMAN`. This may take 10-20 minutes, and the trained model will be saved to `checkpoints/meta`.
- Transfer the meta-trained model to the target task: <br>
`python main.py -md esm2 -m meta-transfer -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -a GEMME -p SYUA_HUMAN`. This may take several minutes, and the trained model will be saved to `checkpoints/meta-transfer`.
- Test the trained model, print results, and save predictions: <br>
`python main.py -md esm2 -m meta-transfer -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -a GEMME -p SYUA_HUMAN -t`. This may take a few seconds, and the predictions will be saved to `predictions/`.
- Other datasets can also be used as long as they have the same file format as the ones in ProteinGym and are in the correct directory.
