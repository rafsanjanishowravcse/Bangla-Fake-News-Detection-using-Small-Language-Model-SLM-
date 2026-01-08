import requests
import mimetypes
import os
import json

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from sarvamai import SarvamAI

grok_api_key = os.getenv('GROK_API_KEY')
serp_dev_api_key = os.getenv('SERP_DEV_API_KEY')
sarvam_api_key = os.getenv('SARVAM_API_KEY')

model_multi_query = os.getenv('model_multi_query')
model_summarizer = os.getenv('model_summarizer')
model_judge = os.getenv('model_judge')

#model_multi_query = 'llama3-8b-8192'
#model_summarizer = 'llama3-8b-8192'
#model_judge = 'llama3-8b-8192'

empty_search = None

class SerperRetrieverWrapper:
    #Class to use Serper as retriver agent for the RAG framework
    def __init__(self, api_key: str, num_results: int = 15):
        self.api_key = api_key
        self.num_results = num_results
    
    def get_relevant_documents(self, query: str):
        """
        Query Serper.dev and return up to `num_results` organic search hits.
        Each hit is a dict: { "title": str, "link": str, "snippet": str }.
        """
        try:
            _SERPER_SEARCH_URL = "https://google.serper.dev/search"
    
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": '(site:news18.com OR site:ptinews.com OR site:politifact.com) ' + query,
                "num": self.num_results,
            }
            
            resp = requests.post(_SERPER_SEARCH_URL, headers=headers, json=payload, timeout=5)
            
            if resp.status_code != 200:
                raise Exception(f"Serper API Error: {resp.text}")
            results = resp.json()
            
            documents = []
            for result in results.get("organic", [])[:self.num_results]:
                content = f"{result.get('title', '')}\n{result.get('snippet', '')}"
                documents.append(Document(page_content=content, metadata={"source": result.get("link", "")}))
            
            if not documents:
                empty_search = Exception("No result found")
                raise empty_search
            
            return documents
        except Exception as e:
            if e is empty_search:
                print('serper main exception1')
                raise empty_search
            else:
                print('serper main exception2')
                raise e
        
def verify_news(user_claim, input_lang = 'auto'):
    """
       Description: This function is used to verify the claim provided by the user and output as REAL or FAKE or UNSURE based on the context with a short explanation
       INPUT: user_claim --> The news user wish to verify
       OUTPUT: FAKE/REAL/UNSURE with explanation
    """
    try:
       print('here1: ', user_claim)
       claim1 = user_claim.replace("'","")
       claim1 = user_claim.replace("\n"," ")
       print('here2: ', claim1)
        
       serper_retriever = SerperRetrieverWrapper(api_key=serp_dev_api_key)
       context_retriever = RunnableLambda(serper_retriever.get_relevant_documents)
       
       """
       Implemeting RAG framework with Multi Query Translation
       Step 1 -- Take input from user for the news to verify
       Step 2 -- Generate 3 variants of the news for better seach results
       Step 3 -- Summarise the results from all the previous steps to be passed to main prompt
       Step 4 -- Deliver the final verdict with a short explanation
       """
       
       #Multi Query Generation
       
       multi_query_template = """You are an AI language model assistant. Your task is to generate three 
       different versions of the given user question to retrieve relevant documents from a vector 
       database. By generating multiple perspectives on the user question, your goal is to help
       the user overcome some of the limitations of the distance-based similarity search. 
       Provide these alternative questions separated by newlines. Original question: {question}"""
       perspectives_prompt = ChatPromptTemplate.from_template(multi_query_template)
       
       llm_multi_query = ChatGroq(api_key = grok_api_key, model_name = model_multi_query)
       
       generate_queries = (
           perspectives_prompt 
           | llm_multi_query
           | StrOutputParser() 
           | (lambda x: x.split("\n"))
       )
       
       #Summarization using multi query
       
       summarizer_template = '''
          You are an assistant summarizing factual evidence from multiple documents.
       
          Based on the following documents, extract the key facts relevant to the claim.
          
          Claim: {question}
          
          Documents:
          {context}
          
          Return a short neutral summary of the key facts only.
       '''
       summarizer_prompt = PromptTemplate.from_template(summarizer_template)
       
       llm_summarizer = ChatGroq(api_key = grok_api_key, model_name = model_summarizer)
       summarizer_chain = (
           {
               "context": context_retriever,
               "question": generate_queries
           }
           | summarizer_prompt
           | llm_summarizer
           | StrOutputParser()
       )
       
       #Final Judgement 
       
       fact_checker_template = '''
          You are a fact-checking assistant.
          
          Claim: {question}
          
          Evidence:
          {evidence}
          
          Decide whether the claim is REAL or FAKE or UNSURE based only on the evidence.
          
          Respond in this format:
          Classification: REAL or FAKE
          Explanation: <your reasoning>
       '''
       
       fact_checker_prompt = PromptTemplate.from_template(fact_checker_template)
       
       llm_fact_checker = ChatGroq(api_key = grok_api_key, model_name = model_judge)
       fact_checker_chain = (
           {
               "question": RunnablePassthrough(),
               "evidence": summarizer_chain 
           }
           | fact_checker_prompt
           | llm_fact_checker
           | StrOutputParser()
       )
   
       claim = claim1
       
       #Calling SARVAM API to translate Indic languages to English
       client = SarvamAI(api_subscription_key = sarvam_api_key)
       
       try:
           translation = client.text.translate(
           input=claim,
           source_language_code="auto",
           target_language_code="en-IN"
           )
       except Exception as e:
           print(f"Error during translation: {e}")
           error_msg = 'It appears you have provided input in an alien language. Please try again with some other language'
           return error_msg,error_msg,error_msg
       
       claim_final = translation.translated_text if translation else claim
       claim_orig_lang = translation.source_language_code
       print(f"Translated claim: {claim_final}")
       
       verdict_orig = fact_checker_chain.invoke(claim_final)
       verdict_class = verdict_orig.split('\n')[0]
       verdict_explan = verdict_orig.split('\n')[-1]
       
       if input_lang == 'auto':
           trans_lang = claim_orig_lang
       else:
           trans_lang = input_lang
   
       if claim_orig_lang != 'en-IN':
           try:
               translation_class = client.text.translate(
               input=verdict_class,
               source_language_code='en-IN',
               target_language_code=claim_orig_lang
               )
           except Exception as e:
               print(f"Error during verdict translation: {e}")  
               error_msg = 'Something went wrong while translating the verdict. Please try again'
               return error_msg,error_msg,error_msg
               
           try:
               translation_explan = client.text.translate(
               input=verdict_explan,
               source_language_code='en-IN',
               target_language_code=claim_orig_lang
               )
           except Exception as e:
               print(f"Error during verdict translation: {e}")  
               error_msg = 'Something went wrong while translating the verdict. Please try again'
               return error_msg,error_msg,error_msg
           
           verdict_trans_class = translation_class.translated_text
           verdict_trans_explan = translation_explan.translated_text
           verdict_trans = verdict_trans_class + '\n\n' + verdict_trans_explan
           
           verdict_orig = verdict_class + '\n\n' + verdict_explan
       else:
           verdict_orig = verdict_class + '\n\n' + verdict_explan
           verdict_trans = verdict_orig
       
       return(claim_final, verdict_orig, verdict_trans)
    except Exception as e:
        if str(e) == 'No result found':
            print('Error in main proc1. Error is ', e)
            error_msg = 'The search for this claim came back empty. Please rephrase the claim or try with a new one'
            return('UNSURE' ,error_msg,error_msg)
        else:
            print('Error in main proc2. Error is ', e)
            error_msg = 'Something went wrong. Please try after some time'
            return(error_msg,error_msg,error_msg)
    
def transcribe_audio(audio):
    """
       Description: This function trascibes audio using SarvamAI STT model 
    """
    try:
        client = SarvamAI(api_subscription_key = sarvam_api_key)
        mime_type, _ = mimetypes.guess_type(audio)
        
        with open(audio, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=("audio.mp3", f, mime_type or "audio/mpeg"),
                model="saarika:v2.5",
                language_code="unknown"
            )
        ret_var = response.transcript
        ret_lang = response.language_code
    except Exception as e:
        print(f"Error during translation: {e}")
        ret_var = ''

    return ret_var, ret_lang
    
def verify_news_audio(audio):
    """
       Description: This function verifies the news where input method is Audio
    """
    
    claim, orig_lang = transcribe_audio(audio)
    if claim == '':
        error_msg = 'I could not understand your message. Please try recording again'
        return(error_msg,error_msg,error_msg)
    
    final_claim, verdict, verdict_trans = verify_news(claim, orig_lang)
    return final_claim, verdict, verdict_trans
    
if __name__ == '__main__':
    print('helllo')