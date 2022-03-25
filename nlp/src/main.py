import os
import pathlib
import torch
from fastapi import FastAPI, Depends
from starlette.exceptions import HTTPException
from dotenv import load_dotenv
from starlette.requests import Request
import requests


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

dir = pathlib.Path(__file__).parent.parent.absolute()
localenv = os.path.join(dir, "local.env")
if os.path.exists(localenv):
    load_dotenv(localenv, override=True)

#client_id = os.environ.get("CLIENT_ID")

# pre fill client id
#swagger_ui_init_oauth = {
#    "usePkceWithAuthorizationCodeGrant": "true",
#    "clientId": client_id,
#    "appName": "LIS",
#}

app = FastAPI()


#INITIALISING MODELS

#DialoGPT
DialoGPTtokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
DialoGPTmodel = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

#BERT
unmasker = pipeline('fill-mask', model='bert-base-uncased')

#Sentence Transformers
Sentence_Transformers = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')

#DEFINING NECESSARY FUNCTIONS


#DialoGPT
def dialoGPT(input):
  new_user_input_ids = DialoGPTtokenizer.encode(input +  DialoGPTtokenizer.eos_token, return_tensors='pt')
    # append the new user input tokens to the chat history
  bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
  chat_history_ids = DialoGPTmodel.generate(bot_input_ids, max_length=1000, pad_token_id=DialoGPTtokenizer.eos_token_id)

  #results = "DialoGPT: {}".format(DialoGPTtokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
  dialoGPT_answer = DialoGPTtokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
  return dialoGPT_answer



#Masks a randomn word in the sentence
def input_mask(results):
  orig_text_list = results.split()
  len_input = len(orig_text_list)
  #Random index where we want to replace the word 
  rand_idx = random.randint(1,len_input-1)
  new_text_list = orig_text_list.copy()
  new_text_list[rand_idx] = '[MASK]'
  new_mask_sent = ' '.join(new_text_list)
  return new_mask_sent

#Unmasks the sentence using BERT and returns 5 sentence suggestions
def unmasking_BERT(new_mask_sent):
    unmasked_sentence = unmasker(new_mask_sent)
    sentence1 = unmasked_sentence[0]['sequence']
    sentence2 = unmasked_sentence[1]['sequence']
    sentence3 = unmasked_sentence[2]['sequence']
    sentence4 = unmasked_sentence[3]['sequence']
    sentence5 = unmasked_sentence[4]['sequence']
    return sentence1, sentence2, sentence3, sentence4, sentence5




#POS + masking sentence
def POS_masking(input):
  tokenized_input = input.split()
  pos_input = nltk.pos_tag(tokenized_input)
  noun_to_mask = []
  adj_to_mask = []
  for i in pos_input:
    if 'NN' in i:
      noun_to_mask += [i[0]]
    if 'JJ' in i:
      adj_to_mask += [i[0]]
  return noun_to_mask, adj_to_mask




#Computes sentence similarity
def sentence_similarity(unmasked_sentences,dialoGPT_answer):
  embedding_dialoGPT_answer = Sentence_Transformers.encode(dialoGPT_answer)
  embedding_1 = Sentence_Transformers.encode(unmasked_sentences[0])
  embedding_2 = Sentence_Transformers.encode(unmasked_sentences[1])
  embedding_3 = Sentence_Transformers.encode(unmasked_sentences[2])
  embedding_4 = Sentence_Transformers.encode(unmasked_sentences[3])
  embedding_5 = Sentence_Transformers.encode(unmasked_sentences[4])
  similarity_score_1 = util.cos_sim(embedding_dialoGPT_answer, embedding_1)
  similarity_score_2 = util.cos_sim(embedding_dialoGPT_answer, embedding_2)
  similarity_score_3 = util.cos_sim(embedding_dialoGPT_answer, embedding_3)
  similarity_score_4 = util.cos_sim(embedding_dialoGPT_answer, embedding_4)
  similarity_score_5 = util.cos_sim(embedding_dialoGPT_answer, embedding_5)
  return similarity_score_1, similarity_score_2, similarity_score_3, similarity_score_4, similarity_score_5 


step = 0  
    

@app.get('/api/healthcheck', status_code=200, tags=["api"])
async def healthcheck():
    return 'Ready'

@app.get('/api/autosuggest', tags=["api"]) 
async def autosuggest(sentence: str, request: Request):
    dialoGPT_answer = dialoGPT(sentence)
    masked_sentence = input_mask(dialoGPT_answer)
    unmasked_sentences = unmasking_BERT(masked_sentence)
    return unmasked_sentences