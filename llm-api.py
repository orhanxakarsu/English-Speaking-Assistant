import torch
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List,Optional

app = FastAPI(title="llm-api" ,version="0.1")

#from huggingface_hub import login
#login()


class Data(BaseModel):
    
    past_content:list[dict]
    question:str


class DataOfDict(BaseModel):
    content:list[dict] 





@app.post("/info/")
def info():
    return """
            Bu API get_question üzerinde çalışmaktadır.
            Bu API 2 parametre almaktadır -> past_content, question

            past_content listesinde geçmiş soru cevap verileri tutulur.
            
            Yapı olarak user ve assistant olarak ayrılmaktadır.
            user ve assistant sıralı olarak konuşmayı ilerletmesi gerekmektedir. Yoksa exception fırlatır.
            Gelen istek şekli -> [{"role":"user", "content":"daha önce gönderilmiş mesaj"},{"role":"assistant", "content":["daha önce gelen mesaj"] }]
            
            
            İkinci parametre olan question'a ise sadece gönderdiğimiz soru bilgisi yeterlidir. 
            
            Model direkt olarak past_content yapısında veri dönmektedir. 

            """


pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",  # replace with "mps" to run on a Mac device
)
#response = pipe(chat, max_new_tokens=512)



@app.post("/get_question_from_messages/")
def get_question_from_messages(data: DataOfDict):
    messages = data.content
    if len(messages) == 1:
        # Kullanıcı rolü ve ilk mesaj ayarı
        start_content ={"role": "user", "content": "You are my English teacher. My English level is between A2 and B1. I want to practice English. Correct me if I make mistakes while speaking English. Tell the truth. You shouldn't use emojis. Let's talk. You must write at my English level. Your name is Hanna."}

        
        user_message = messages[0]["content"] if messages else ""
        #combined_input = f"{QA_role}\n{user_message}"
        user_content = [{"role": "user", "content":[start_content["content"],user_message]}]
        #input = [start_content, user_content ]
        # Modelden yanıt alın
        outputs = pipe(user_content,max_new_tokens=200)#, **generation_args)
        #me
        # Mesaj listesini güncelleyin
    
    else:

        
        # Modelden yanıt alın
        outputs = pipe(messages,max_new_tokens=512)#, **generation_args)
        #assistant_response = outputs[0]["generated_text"][-1]['content']
        
        # Mesaj listesini güncelleyin

    return  outputs[0]["generated_text"]



    
    
    
    
    
    
    
    
    
    
    
