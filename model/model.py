""" Model based on 🤗 (HuggingFace) API

Much of my code is based on the below Medium article so read it for a deeper explanation on how my code works: 
https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
"""


import sys
import os

from dataclasses import dataclass, field
import logging
from datasets import load_dataset, Dataset, DatasetDict, load_metric
import datasets
import numpy as np
from transformers import (
    HfArgumentParser, 
    T5ForConditionalGeneration,  
    Seq2SeqTrainingArguments,  DataCollatorForSeq2Seq, 
    EvalPrediction, set_seed,
    AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM)
from typing import Dict, List, Optional 
import unicodedata
import nltk
import transformers
from transformers.trainer_utils import get_last_checkpoint
import wandb
from model.custom_trainer import CustomTrainer
logger = logging.getLogger(__name__)


metric = load_metric("rouge")

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(

        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    # can add different padding strategies as well
    padding_strategy: bool = field(
        default=None,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
                " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    """
    Args:
        sortish_sampler (`bool`, *optional*, defaults to `False`):
            Whether to use a *sortish sampler* or not. Only possible if the underlying datasets are *Seq2SeqDataset*
            for now but will become generally available in the near future.

            It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness
            for the training set.
        predict_with_generate (`bool`, *optional*, defaults to `False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
        generation_max_length (`int`, *optional*):
            The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `max_length` value of the model configuration.
        generation_num_beams (`int`, *optional*):
            The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `num_beams` value of the model configuration.
    """
    # wandb params
    project_name: Optional[str] = field(
        default="qa-bot"
    )
    run_name: Optional[str] = field(
        default=None
    )
    tags: Optional[List[str]] = field(
        default=None
    )
    group: Optional[str] = field(
        default=None
    )
    notes: Optional[str] = field(
        default=None
    )

   # answer generation params 
    entity: Optional[str] = field(
        default= "piazza-qabot"
    )

    temperature: Optional[float] = field(
        default=None
    )
    top_k: Optional[int] = field(
        default=None
    )
    top_p: Optional[float] = field(
        default=None
    )
    repetition_penalty: Optional[float] = field(
        default=None
    )
    length_penalty: Optional[float] = field(
        default=None
    )
    no_repeat_ngram_size: Optional[int] = field(
        default=None
    )

    do_sample: Optional[bool] = field(
        default=False
    )
    num_return_sequences: Optional[int] = field(
        default= 1
    )



def clean_text(text: str) -> str: return unicodedata.normalize("NFKD", text.strip().replace('\n', ''))


def print_dataset(dataset: DatasetDict):
    print('PRINTING DATASET')
    for s in dataset.keys():
        print(s)
        d:Dataset = dataset[s]
        print(d.format)

    

def train(config_path: str):

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args: ModelArguments = None 
    data_args: DataTrainingArguments = None 
    training_args: MyTrainingArguments = None


    if config_path.endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(config_path)
        
    else:
       raise AssertionError('Please provide path to json config file')
    

    # setup wandb tracking
    # if "wandb" in training_args.report_to:
    #     wandb.init(project=training_args.project_name, name=training_args.run_name, tags=training_args.tags, group=training_args.group, notes=training_args.notes,
    #     entity=training_args.entity)


    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    
    
    log_level = training_args.get_process_log_level()
    #log_level = transformers.logging.WARNING
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    datasets.utils.logging.set_verbosity(log_level)
    

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Logging level is {logging.getLevelName(logger.getEffectiveLevel())}")
  
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print(f'last checkpoint is {last_checkpoint}')
        # stuff in output_dir that is not a checkpoint -> don't want to overwrite
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed before initializing model.
    set_seed(training_args.seed) 


    # dataset loading

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]

    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        usecols=['question', 'answer'],
        cache_dir=model_args.cache_dir
    )
    print('PRINTING RAW DATASETS')
    print(raw_datasets)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )


    # tokenize cleaned datasets

    def tokenize(batch: Dict[str, List['str']]):
        # print(batch)
        questions = batch['question']
        answers = batch['answer']
        for i in range(len(questions)):
            questions[i] = clean_text(questions[i])
            answers[i] = clean_text(answers[i])

        # Max token length of t5-small is 512  
    
        # padding done dynamically, per-batch through data collator
        q_batch = tokenizer(
            batch['question'],
            truncation = True,
            #padding = 'longest', 
        )

        with tokenizer.as_target_tokenizer(): 
            a_batch = tokenizer(
                batch['answer'],
                truncation = True,
                #padding = 'longest',
            )

        q_batch['labels'] = a_batch['input_ids']

        return q_batch

    def find_max(score_list: List[tuple]):
        max_score = None
        for pred, score in score_list:
            # if score[1].
            rogue1_fmeasure = score["rouge1"].mid.fmeasure
            if not max_score or rogue1_fmeasure > max_score[1]: # OR short-circuits
                max_score = (pred, rogue1_fmeasure, score_list.index((pred, score)))
        return max_score


    """
    Rogue1 evaluation metric.
    Can also track several eval metrics here such as avg prediction length. These get logged to wandb.
    Currently tracked metrics: rouge1, rogue2, rogueL, gen_len


    Account for beams here. i.e. num_return_sequences > 1

    (400, 400) num_beams = 4
    use best rogue1 score per set of beams

    """
    def compute_metrics(eval_pred:EvalPrediction):
        log_level = transformers.logging.WARNING
        datasets.utils.logging.set_verbosity(log_level)

        # contains a prediction for each beam. For instance, if there are 20 questions and 4 beams, then there will be 
        # 20*4=80 total predictions
        predictions, labels = eval_pred
        num_return_sequences = training_args.num_return_sequences
        print('IN COMPUTE METRICS')
        # print(predictions.shape)
        # print(labels.shape)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                        for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                        for label in decoded_labels]


        decoded_labels = np.asarray(decoded_labels) 
        # repeat labels once per num_return_sequences to match it up with predictions
        decoded_labels = np.repeat(decoded_labels, num_return_sequences)

        decoded_preds = np.asarray(decoded_preds)
        # print(f'decoded_labels is {decoded_labels.shape}')
        # print(f'decoded predictions is {decoded_preds.shape}')

        num_samples = decoded_preds.shape[0] // num_return_sequences
        rogue1 = 0
        chosen_preds = []


        # print a max of x samples in eval dataset
        # TODO: Change this for printing beams here instead of in custom_trainer()
        NUM_PRINT_SAMPLES = -1
        print(f'num samples is {num_samples}')
        for i in range(num_samples):
            #question = raw_datasets['validation']['question'][i]

            # uncomment below for printing question, answer pairs  
            # if i < NUM_PRINT_SAMPLES:
            #     print(f"\n\nQUESTION: {question}\n") 
            #     print(f'ACTUAL ANSWER : {decoded_labels[i * num_return_sequences]}\n')

            beam_preds = decoded_preds[i * num_return_sequences : (i * num_return_sequences) + num_return_sequences]
            # labels should be the same for all samples in the beam group
            beam_labels = decoded_labels[i * num_return_sequences : (i * num_return_sequences) + num_return_sequences]
            rogue_scores = []
            for j in range(num_return_sequences):
                # print(f'beam preds is {beam_preds[j]}')
                # print(f'beam labels is {beam_labels[j]}')
                result = metric.compute(predictions=[beam_preds[j]], references=[beam_labels[j]],
                                use_stemmer=False)
                rogue_scores.append((beam_preds[j], result))

                # if i < NUM_PRINT_SAMPLES:
                #     print(f'BEAM {j} : {beam_preds[j]}\n')
                #     print(f'SCORE {j} : {round(result["rouge1"].mid.fmeasure * 100, 4)}\n')

            # use the beam with the highest rogue1 mid fmeasure 
            pred, score, beam_number  = find_max(rogue_scores)

            # if i < NUM_PRINT_SAMPLES:
            #     print(f"\nCHOSE BEAM {beam_number}: {pred}\n")            
            #     print(f'SCORE {beam_number}: {round(score * 100, 4)}')

            rogue1 += score
            # save the encoded chosen prediction for computation of gen_len
            chosen_pred_idx = i * num_return_sequences 
            chosen_pred_idx += beam_number
            chosen_preds.append(predictions[chosen_pred_idx])

        # compute avg rogue1 score across all samples in the batch
        rogue1 /= num_samples 
        rogue1 *= 100

        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                        for pred in chosen_preds]


        # print(prediction_lens)
        gen_len = np.mean(prediction_lens)
        return_dict =  {"eval_rouge1" : round(rogue1, 4), "gen_len" : round(gen_len, 4)}
        print(return_dict)
        
        
        log_level =  training_args.get_process_log_level()
        datasets.utils.logging.set_verbosity(log_level)

        return return_dict



    # batch tokenization 

    tokenized_datasets = {} # ensure no package name clash with datasets
    for split_alt, split in zip(['train', 'eval', 'predict'], ['train', 'validation', 'test']):
        do_split = f"do_{split_alt}"
        if getattr(training_args, do_split):
            if split not in raw_datasets:
                raise ValueError(f"--do_{do_split} requires a {split} dataset") 
            ds = raw_datasets[split]

            max_samples = getattr(data_args, f'max_{split_alt}_samples')
            if max_samples is not None:
                 max_samples = min(len(ds), max_samples)
                 ds = ds.select(range(max_samples))

            ds = ds.map(
                tokenize,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Running tokenizer on {split} dataset",
            )

            tokenized_datasets[split] = ds


    print(tokenized_datasets)

    if training_args.do_eval and data_args.max_eval_samples is not None:
        # During Feature creation dataset samples might increase, we will select required samples again
        max_eval_samples = min(len(tokenized_datasets['validation']), data_args.max_eval_samples)
        tokenized_datasets['validation'] = tokenized_datasets['validation'].select(range(max_eval_samples))

    if training_args.do_predict and data_args.max_predict_samples is not None:
        # During Feature creation dataset samples might increase, we will select required samples again
        max_predict_samples = min(len(tokenized_datasets['test']), data_args.max_predict_samples)
        tokenized_datasets['test'] = tokenized_datasets['test'].select(range(max_predict_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=data_args.padding_strategy)

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'] if training_args.do_train else None,
        eval_dataset=tokenized_datasets['validation'] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # train, eval, predict time!

    # Training (eval done within Trainer i.e. every eval_steps steps)
    if training_args.do_train:
        logger.info("*** Training ***")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)


    # Evaluation (also should use checkpoint if avaliable)
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(tokenized_datasets['validation'])

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(tokenized_datasets['validation'] )
        metrics["eval_samples"] = min(max_eval_samples, len(tokenized_datasets['validation']))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(tokenized_datasets['test'])
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(tokenized_datasets['test'])
        )
        max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(tokenized_datasets['test'])
        metrics["predict_samples"] = min(max_predict_samples, len(tokenized_datasets['test']))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)






# Below is unused
def decode_predictions(preds:List):
    for p in preds:
        print(tokenizer.decode(p, skip_special_tokens=True))
        print('-----------------------------------------')


def t5_model_init():
    print_gpu_utilization()
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to('cuda')
    print_gpu_utilization()
  
    return model


