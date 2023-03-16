""" Generate a sample inference (prediction) using the model.

**Ensure that you change the filepath in load_model() to point to
the model checkpoint directory**

"""
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, pipeline)


t5_model_path ="/home/shared/qabot/checkpoint-18210/"
gpt2_model_path ="gpt2"

def load_model():

    config = AutoConfig.from_pretrained(
       t5_model_path 
    )
    tokenizer = AutoTokenizer.from_pretrained(
       t5_model_path 
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
       t5_model_path 
    )

    return config, tokenizer, model

"""
Ignore this for now. I was just playing around with the generation capabilities
of gpt2.
"""
def gpt2_inference():
    text = """
            Question: In what country is Normandy located?
            Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries. 
            Answer: France 

            Question: When were the Normans in Normandy?
            Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries. 
            Answer: 10th and 11th centuries

            Question: From which countries did the Norse originate?
            Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries. 
            Answer: Denmark, Iceland and Norway

            Question: Who was the Norse leader?
            Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries. 
            Answer: 

            """
    generator = pipeline('text-generation', model='gpt2')
    out = generator(text, max_length=800, num_return_sequences=3, top_p=0.9, do_sample=True)
    for a in out:
        print(f'\n\n {a["generated_text"]}')


def main():
    # gpt2_inference()
    config, tokenizer, model = load_model()
    question = "Where can we check our final grade of this course?"
    tokenized_input = tokenizer(question, return_tensors="pt")
    input_ids = tokenized_input['input_ids']
    attention_mask = tokenized_input['attention_mask']
    num_beams = 4
    answer_beams = model.generate(input_ids, attention_mask=attention_mask, do_sample=True,
                                                    num_beams=num_beams, num_return_sequences=num_beams, top_k=50, top_p=1.0,
                                                    early_stopping=False, no_repeat_ngram_size=6, max_length=400)
    answers = [tokenizer.decode(a, skip_special_tokens=True) for a in answer_beams]
    for i in range(num_beams):
        print(f'ANSWER {i} IS : {answers[i]}')


if __name__ == "__main__":
    main()
