import os
import pathlib
from fastapi import FastAPI, Depends
from starlette.exceptions import HTTPException
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from dotenv import load_dotenv
from starlette.requests import Request
import requests

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

# Add the bearer middleware
#app.add_middleware(AuthenticationMiddleware, backend=AadBearerMiddleware())

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
step= 0

#fill_mask = pipeline("fill-mask", model="camembert-base", tokenizer="camembert-base")
#fill_mask = pipeline("fill-mask", model= model, tokenizer= tokenizer)
# Let's chat for 5 lines

    
    

@app.get('/api/healthcheck', status_code=200, tags=["api"])
async def healthcheck():
    return 'Ready'

@app.get('/api/autosuggest', tags=["api"]) 
async def autosuggest(sentence:str, request: Request):
    #results = fill_mask(sentence)
    
     # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(">> User:" + sentence + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    results = "DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
    return results

