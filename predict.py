import argparse
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import EsmTokenizer, EsmForMaskedLM

from fsfp import config
from fsfp.dataset.saprot import SaProtMutantData, saprot_zero_shot
from fsfp.pipeline import Pipeline
from fsfp.utils.score import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", "-ckpt", type=str,
                        default="checkpoints/meta-transfer/esm2/A0A140D2T1_ZIKV_Sourisseau_2019/r16_ts40_cv4_cosine_mt3_GEMME/")
    parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--mode", "-m", type=str, choices=["finetune", "transfer", "meta", "meta-transfer"],
                        default="meta-transfer", help="perform finetuning, meta learning or meta-transfer")
    parser.add_argument("--test", "-t", action="store_true",
                        help="load the trained models from checkpoints and test them")
    parser.add_argument("--model", "-md", type=str, choices=config.model_dir.keys(),
                        default="esm2", help="name of the foundation model")
    parser.add_argument("--protein", "-p", type=str, default="A0A140D2T1_ZIKV",
                        help="name of the target protein")
    parser.add_argument("--train_size", "-ts", type=float, default=40,
                        help="few-shot training set size, can be a float number less than 1 to indicate a proportion")
    parser.add_argument("--train_batch", "-tb", type=int, default=10,
                        help="batch size for training (outer batch size in the case of meta learning)")
    parser.add_argument("--eval_batch", "-eb", type=int, default=1000,
                        help="batch size for evaluation")
    parser.add_argument("--lora_r", "-r", type=int, default=16,
                        help="hyper-parameter r of LORA")
    parser.add_argument("--optimizer", "-o", type=str, choices=["sgd", "nag", "adagrad", "adadelta", "adam"],
                        default="adam", help="optimizer for training (outer loop optimization in the case of meta learning)")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="maximum training epochs")
    parser.add_argument("--max_grad_norm", "-gn", type=float, default=3,
                        help="maximum gradient norm to clip to")
    parser.add_argument("--mask", "-mk", type=str, choices=["train", "eval", "all", "none"], default="none",
                        help="whether to compute masked 0-shot scores")
    parser.add_argument("--list_size", "-ls", type=int, default=5,
                        help="list size for ranking")
    parser.add_argument("--max_iter", "-mi", type=int, default=10,
                        help="maximum number of iterations per training epoch, useless during meta training")
    parser.add_argument("--eval_metric", "-em", type=str, choices=metrics, default="spearmanr",
                        help="evaluation metric")
    parser.add_argument("--retr_metric", "-rm", type=str, default="cosine",
                        help="similarity metric used for retrieving proteins for meta training")
    parser.add_argument("--augment", "-a", nargs="*", type=str, default=[],
                        help="specify one or more models to use their zero-shot scores for data augmentation")
    parser.add_argument("--meta_tasks", "-mt", type=int, default=3,
                        help="number of tasks used for meta training")
    parser.add_argument("--meta_train_batch", "-mtb", type=int, default=10,
                        help="inner batch size for meta training")
    parser.add_argument("--meta_eval_batch", "-meb", type=int, default=64,
                        help="inner batch size for meta testing")
    parser.add_argument("--adapt_lr", "-alr", type=float, default=5e-3,
                        help="learning rate for inner loop")
    parser.add_argument("--adapt_steps", "-as", type=int, default=4,
                        help="number of iterations for inner loop")
    parser.add_argument("--patience", "-pt", type=int, default=15,
                        help="number of epochs to wait until the validation score improves")
    parser.add_argument("--n_sites", "-ns", nargs="+", type=int, default=[1],
                        help="possible numbers of mutation sites in the training data. \
                              setting to 0 means no constraint")
    parser.add_argument("--negative_train", "-neg", action="store_true",
                        help="whether to constraint the training data to negative examples")
    parser.add_argument("--cross_validation", "-cv", type=int, default=5,
                        help="number of splits for cross validation (shuffle & split) on the training set. \
                              if set to 1, the test set will be used for validation; \
                              if set to 0, no testing or validation will be performed.")
    parser.add_argument("--seed", "-s", type=int, default=666666,
                        help="random seed for training")
    parser.add_argument("--save_postfix", "-sp", type=str, default="",
                        help="a custom string to append to all data paths (data, checkpoints and predictions)")
    parser.add_argument("--force_cpu", "-cpu", action="store_true",
                        help="use cpu for training and evaluation even if gpu is available")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    assert args.model == "esm2"
    assert args.mode == "meta-transfer"

    path = config.data_path.replace(".pkl", f"{args.save_postfix}.pkl")
    proteins = torch.load(path)
    pipeline = Pipeline(args)
    checkpoint_dir = args.checkpoint_dir
    model, tokenizer = pipeline.get_base_model(checkpoint_dir)
    model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True).to(args.device)


    args = self.args
    proteins = self.select_datasets(all_proteins)
    if args.mode == 'meta':
        database, topk = self.get_meta_database(all_proteins)
    
    reports = {}
    for protein in proteins:
        print(f'**********************Current dataset: {protein["name"]}**********************')
        if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
            eval_metric = args.eval_metric
            args.eval_metric = 'ndcg' # in case of nan spearmanr
        
        train, test = split_data(protein, args.train_size, n_sites=args.n_sites,
                                    neg_train=args.negative_train, scale=args.list_size == 1)
        if args.test:
            report = self.test_single(train, test)
        elif args.mode != 'meta':
            if args.mode == 'finetune' and args.augment:
                protein = self.augment_data(protein)[0]
            report = self.finetune_single_cv(train, test)
        else:
            src_name = '_'.join(protein['name'].split('_')[:2])
            tgt_names = topk[src_name]['tgt_names'][:args.meta_tasks]
            meta_train = [database[name] for name in tgt_names]
            if args.augment:
                meta_train[-len(args.augment):] = self.augment_data(protein)
            if args.meta_tasks < 4:
                meta_train *= 2
            report = self.meta_single(meta_train, train, test)
        reports[protein['name']] = report
        torch.cuda.empty_cache()
        
        if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
            args.eval_metric = eval_metric
    
    if args.test and args.protein in {'single-site', 'multi-site', 'all'}:
        save_path = self.get_save_dir(args.mode, args.protein, prediction=True)
        save_path += '_base.pkl' if args.epochs == 0 else '.pkl'
        make_dir(save_path)
        reports = summarize_scores(reports, save_path)
        print('**********************Score summary**********************')
        print(reports[args.eval_metric])


    pipeline(proteins)
