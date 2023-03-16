"""
Deployment Steps:

1. log into Piazza
2. get feed containing all posts (no Piazza API throttling with getting feed) 
3. filter feed and retrieve full posts (post retrieval is throttled )
    - wait a delay of ~2.5s b/w fetching posts

4. make inferences on those new posts
5. post model generated answers as private followups to the retrieved posts

Considerations
    - bot needs to be TA/instructor to post instructor only private followups
    - ensure posts get responded to by bot (avoid starvation)
        - i.e. TAs and other students don't respond faster than bot
    - bot polling limit in between fetching posts?
        - can this be event driven? i.e. bot gets notified when a student posts a new question?
        - else, what should polling time be?
            - ? __ secs
        - 
    - race condition between post retrieval and responding to post
        - ensure no student/instructor answer after post has been retrieved 
"""

import html
import re
import sys
import time


from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM)


from utils.utils import login, strip_tags

from piazza_api.exceptions import RequestError
# using relative imports here
from .my_piazza_api import MyNetwork

# 1 hour poll period
# POLL_SECS = 3600
POLL_SECS = 3600 
NUM_GRAB_POSTS = 10

def load_model():

    config = AutoConfig.from_pretrained(
        "./model/checkpoint-18210/"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "./model/checkpoint-18210/"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "./model/checkpoint-18210/"
    )

    return config, tokenizer, model




def event_loop(user_profile, course: MyNetwork):
    # print(f'user profile is {user_profile}')
    # print(f'course is {course}')

    # get piazzabot id so don't need to call get_users() in PiazzaAPI which
    # throttles
    all_users = course.get_all_users()
    piazzabot_id = None
    for u in all_users:
        # ** change to appropriate Piazza user (i.e. Brandon Jaipersaud, Paul Zhang, ...)  **
        # this ensures that the bot only retrieves posts that it hasn't already responded to
        if u['name'] == 'PiazzaBot-QABot':
            piazzabot_id = u['id']
            print(f'Piazzabot id is {piazzabot_id}')
            break

    assert(piazzabot_id != None)

    config, tokenizer, model = load_model()

    num_poll_cycles = 0
    num_throttle_errors = 0


    while(1):
        print(f'\n\n\nNum cycles is : {num_poll_cycles}')
        print(f'Num throttle errors is : {num_throttle_errors}')
        # grab posts based on feed (no throttle)
        # still filtered in some sense 
        # set limit based on how much posts we want to respond to?
        posts = course.iter_all_posts(piazzabot_id, limit=NUM_GRAB_POSTS) 

        # total filtered feed
        total_feed = 0 
        # additional filtering based on question content. Contains questions actually responded to
        filtered_feed = 0
        num_posts = 0

        try: 
            for p in posts:
                print(f'posts is {num_posts}')
                num_posts += 1
                total_feed += 1
                # we check again in addition to checking in feed to avoid race condition where a
                # student/TA answers a question in b/w feed and post collection
                if p['type'] == 'question'  and p['no_answer']: 
                    # respond to question as a private followup
                    # gets the most recent edit. History doesn't inc followups
                    subject = p['history'][0]['subject']
                    content = p['history'][0]['content']
                    question:str = subject + '.' + content
                    question:str = html.unescape(question)

                    if re.search('@|<img|<pre|\.png|def', question):
                        # ignore post
                        continue

                    # filter html tags
                    question = strip_tags(question)

                    print("-" * 60)
                    print(f'Question is : {question}')
                    # print(f'Subject is : {subject}')
                    # print(f'Content is : {content}')
                    print("-" * 60)
                    # print(p)
                    filtered_feed += 1

                    # feed question to model and get predictions
                    # sequential inference helps prevent Piazza throttling 

                    tokenized_input = tokenizer(question, return_tensors="pt")
                    input_ids = tokenized_input['input_ids']
                    attention_mask = tokenized_input['attention_mask']

                    # generate params
                    num_beams = 4

                    answer_beams = model.generate(input_ids, attention_mask=attention_mask, do_sample=True,
                                                num_beams=num_beams, num_return_sequences=num_beams, top_k=50, top_p=1.0,
                                                early_stopping=False, no_repeat_ngram_size=6, max_length=400)
                    answers = [tokenizer.decode(a, skip_special_tokens=True) for a in answer_beams]
                    print(answers)

                    # post followups 
                    # piazzaapi modified to only post private (instr-only followups)
                    for i in range(num_beams):
                        course.create_followup(p, f"ANSWER {i} : {answers[i]}")

                # add 2.5s delay to avoid throttling
                time.sleep(2.5)

            num_poll_cycles += 1
            # time.sleep(3)
            time.sleep(POLL_SECS)

        # handle throttling error
        except RequestError: 
            print("REQUEST ERROR!")
            time.sleep(5)
            num_throttle_errors += 1
            continue 

    print(f'Num Total feed is {total_feed}')
    print(f'Num Filtered feed is {filtered_feed}')


def deploy():
    cred_file_path = sys.argv[1]
    user_profile, course = login(cred_file_path)
    course = MyNetwork(course._nid, course._rpc.session)
    event_loop(user_profile, course)



