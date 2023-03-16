# Piazza Question-Answering Bot (QABot)


## Introduction

Our goal is to develop a bot that can respond to CSC108: Introduction to Programming student questions on the Piazza discussion board.
-   Input: Question text written by the student on Piazza
-   Output: A collection of ~4-5 possible answers to the question that course staff can reference and modify.

The answers to the question need not be exactly accurate. Instead, the answer needs to be helpful to the course staff, and reduce the amount of time it takes for them to type up an answer.

### Notation
I use 🤗  in place of the term "HuggingFace" which is consistent to what is used in the 🤗 documentation.
## Folder Structure 
Below are the relevant files and folders for my implementation. Anything not listed here can be safely ignored. The below folders and files will be described in more detail in the rest of this document.

    .
    ├── deployment/              
    ├── model/                  
    ├── utils/ # Tasks such as textual data cleaning, creating train/val/test splits and logging into Piazza 
    ├── deploy_main.py        # Entry script for deploying model 
    ├── training_main.py      # Entry script for model training 
    ├── config.json           # Model training/eval/predict parameters
    ├── sample_inference.py   # Generate a model prediction for a sample question
    ├── qabot-env.yml         # Conda environment  

## Environment Setup 

1. Install Anaconda. I've installed Miniconda3, a lightweight version of Anaconda. The installation instructions can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

2. Run: ``` conda env create -n [ENVNAME] --file qabot-env.yml```
3. Run: export WANDB_DISABLED=true. This disables wandb ML experiment tracking which you don't have to worry about for now.


## Prerequisites 
Before you can begin model training or inference, ensure that you have:
 1. The Piazza Q&A dataset  
 2. The model checkpoint folder. 

Due to the large file sizes of the above folders and the need for the dataset to be anonymized, they are not stored in the GitHub repo. Instead, they are stored locally at: `/home/shared/train_large_2/` and `/home/shared/checkpoint-18210/`. By default, the files that use these folders point to the shared file path so **you don't have to change anything**. However, if you wish you can copy them to your local directory and update the appropriate file paths in `config.json` and `sample_inference.py` to point to them. 


## Dataset
The dataset consists of archived Piazza posts from various instances of the courses CSC108, CSC148, CSC209, and CSC263. The train/val/test split is: 9105/506/506 question/answer pairs. 

## Getting Started

First, start by checking whether you can use the model to make a single inference (prediction).

```sample_inference.py``` contains code for using the model to generate 4 answers to the question: "Where can we check our final grade of this course?" Running ```python3.9 sample_inference.py``` should give output like:


```
ANSWER 0 IS : https://mcs.utm.utoronto.ca/148/syllabus.pdf
ANSWER 1 IS : You can check your grades on MarkUs right after you submit it.
ANSWER 2 IS : https://mcs.utm.utoronto.ca/148/syllabus.pdf
ANSWER 3 IS : https://mcs.utm.utoronto.ca/108/syllabus.pdf
```

If this works, then you should not have many issues running the training and deployment scripts below.

Note that you will have to change the filepath in `load_model()` in `sample_inference.py` to point to the model checkpoint directory 

## Model Training, Evaluation and Inference
```training_main.py``` is the entry point for model training, evaluation and inference. Running ```python3.9 training_main.py train <path-to-config-file>``` will perform model training, evaluation and/or inference based on the parameters specified in the config file (i.e. in ```config.json```).  
 

Refer to ```config_explained.json ``` for an explanation of the parameters in ```config.json```. 

Note that you will have to change the train_file/validation_file/test_file and resume_from_checkpoint paths to point to the appropriate dataset and model checkpoint paths.

## Deployment
```deploy_main.py``` is the entry point for the deployment of the model on the Piazza discussion board. It uses the model checkpoint to generate inferences to questions posted on the discussion board. It then posts the generated answers as private followups that are visible only to instructors and TAs.


## Model Training Implementation Details
Our current model checkpoint is the T5 transformer fine-tuned on the above dataset. We use [🤗 Transformers](https://huggingface.co/docs/transformers/index) for our implementation of T5. [Here](https://huggingface.co/docs/transformers/model_doc/t5) are the 🤗 docs for T5. We use a 🤗    community T5 model called [t5_abs_qa](https://huggingface.co/tuner007/t5_abs_qa) that is a bit more tailored to our task. 

For model training/evaluation/inference, we use the [HuggingFace Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer). [Here](https://huggingface.co/course/chapter3/3?fw=pt) is a 🤗  tutorial on how to fine-tune a model using the API.

 [Here](https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887) is a Medium article that explains how to fine-tune T5 in particular. It explains what the parameters in `config.json` means. The author even uses the same ROGUE1 evaluation metric that I use. My code in ```model/model.py``` is similar to what is shown in the article so I highly suggest reading the article to get a better understanding of how ```model/model.py``` works!

### Subclassing 🤗 Trainer API 
 It seems like 🤗 only partially supports varying model generation parameters such as max_length, num_beams, temperature and top_k/top_p as mentioned [here](https://huggingface.co/blog/how-to-generate). The 🤗 generate() function in ```generation_utils.py``` in the 🤗 API supports the varying of generation parameters. However, it doesn't seem to be integrated in the  🤗  Trainer API. Thus, I have created a file: ```model/custom_trainer.py``` which inherits 🤗  Trainer and subclasses Trainer methods such as evaluate(), predict() and prediction_step(). I manually retrieve and pass the generation params (i.e. num_beams, temperature, ...) to the generate() method.

### Implementing the ROGUE1 Evaluation Metric
This is outlined in the Medium article I previously referenced above. In particular, it is done in `compute_metrics()` in `model.py`. Note that for a single question, 4 answers are generated. The ROGUE1 score is taken to be the best ROGUE1 score of the 4 answers.

### Useful Methods in 🤗 Transformers 
You may find it useful to examine the 🤗 Transformers Python package when debugging. For me, this is located at: `/home/<utorid>/miniconda3/envs/qabot/lib/python3.9/site-packages/transformers`. Below are some useful files and functions in the API:

- generation_utils.py
  -   generate()
  -   beam_search()
- trainer.py
  - train()
  - predict()
  - evaluation_loop()
  - prediction_step()
- modeling_t5.py : This is the actual🤗 implementation of T5 which uses Pytorch.
  - forward()
  - class T5Block
  - class T5PreTrainedModel
  - class T5Model
- trainer_seq2seq.py




## Future Directions


- Change evaluation metric
  - ROGUE1 only considers unigram overlap and does not consider semantic similarity between model generated answer and actual answer. Thus, a metric like [BERTScore](https://arxiv.org/abs/1904.09675)  which considers the cosine similarity between answer embeddings might be more meaningful as it accounts for answer semantics.
- Try a larger generative language model like gpt2/gpt3/codex
    - HuggingFace has gpt2 ([link](https://huggingface.co/gpt2), [link](https://huggingface.co/docs/transformers/model_doc/gpt2)) so swapping out T5 for gpt2 shouldn't be too much work. In addition, HuggingFace also has GPT Neo and GPT-J which are open source alternatives to GPT-3 made by [EleutherAI](https://www.eleuther.ai/).
      - Checkout [this](https://www.ankursnewsletter.com/p/openais-gpt-3-vs-open-source-alternatives) article comparing gpt-j to gpt3. 
    - For gpt3/OpenAI, you would need to use the [OpenAI API](https://openai.com/api/) which requires more work. 
    - Experiment with few-shot and zero-shot learning ([paper](https://arxiv.org/abs/2005.14165)) if using gpt2/gpt3/codex.

- Expand model to account for images and code-snippets
  



## Warning: Memory Usage

There is currently 82G available on the "lisa" node in which this model is run. This should be more than enough memory but be mindful of memory usage. 

I believe that doing `df -H .` gives the avaliable disk space and `du -h --max-depth=1 .` gives the space that the folders in the current directory uses up.






## Additional References 


