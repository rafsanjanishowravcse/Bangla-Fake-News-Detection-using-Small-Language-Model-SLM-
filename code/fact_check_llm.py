import requests
import mimetypes
import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from sarvamai import SarvamAI

# Load environment variables from .env file
load_dotenv()

grok_api_key = os.getenv('GROQ_API_KEY')
serp_dev_api_key = os.getenv('SERP_DEV_API_KEY')
sarvam_api_key = os.getenv('SARVAM_API_KEY')

model_multi_query = os.getenv('model_multi_query') or 'llama-3.1-8b-instant'
model_summarizer = os.getenv('model_summarizer') or 'llama-3.1-8b-instant'
model_judge = os.getenv('model_judge') or 'llama-3.1-8b-instant'

# Check if required API keys are present
if not grok_api_key or grok_api_key == 'your_grok_api_key_here':
    print("Warning: GROQ_API_KEY not set. Please update your .env file with a valid Groq API key.")
if not serp_dev_api_key or serp_dev_api_key == 'your_serper_dev_api_key_here':
    print("Warning: SERP_DEV_API_KEY not set. Please update your .env file with a valid Serper Dev API key.")
if not sarvam_api_key or sarvam_api_key == 'your_sarvam_api_key_here':
    print("Warning: SARVAM_API_KEY not set. Please update your .env file with a valid Sarvam AI API key.")

empty_search = None

class BengaliSemanticRetriever:
    """Bengali-specific semantic search for news articles"""
    
    def __init__(self, api_key: str, num_results: int = 5):
        self.api_key = api_key
        self.num_results = num_results
        self.bengali_sources = [
            'site:prothomalo.com',
            'site:bdnews24.com', 
            'site:jugantor.com',
            'site:ittefaq.com.bd',
            'site:samakal.com',
            'site:kalerkantho.com',
            'site:amadershomoy.com',
            'site:dailynayadiganta.com',
            'site:manabzamin.com',
            'site:banglanews24.com'
        ]
        
    def search_bengali_news(self, query: str) -> List[Document]:
        """Search for Bengali news articles using direct Serper API"""
        try:
            # Get raw search results from Serper API directly
            raw_results = self._get_bengali_search_results(query)
            if not raw_results:
                return []
            
            # Convert results to documents without semantic processing
            documents = []
            for result in raw_results[:self.num_results]:
                content = f"{result.get('title', '')}\n{result.get('snippet', '')}"
                documents.append(Document(
                    page_content=content, 
                    metadata={
                        "source": result.get("link", ""),
                        "title": result.get('title', ''),
                        "language": "bengali"
                    }
                ))
            
            print(f"Found {len(documents)} Bengali documents")
            return documents
            
        except Exception as e:
            print(f"Bengali search error: {e}")
            return []
    
    def _get_bengali_search_results(self, query: str) -> List[Dict]:
        """Get Bengali news search results"""
        try:
            _SERPER_SEARCH_URL = "https://google.serper.dev/search"
            
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Use the exact same format as your working JavaScript code
            payload = {
                "q": query,  # Use Bengali query directly like in your working example
                "gl": "bd",  # Bangladesh
                "hl": "bn"   # Bengali language
            }
            
            print(f"Searching with Bengali query: {query}")
            
            resp = requests.post(_SERPER_SEARCH_URL, headers=headers, json=payload, timeout=10)
            
            if resp.status_code != 200:
                raise Exception(f"Serper API Error: {resp.text}")
            
            results = resp.json()
            return results.get("organic", [])
            
        except Exception as e:
            print(f"Bengali search error: {e}")
            return []
    
    def _extract_english_keywords(self, bengali_text: str) -> str:
        """Extract English keywords from Bengali text for search"""
        # Common Bengali-English word mappings for news
        keyword_mappings = {
            'সেনাবাহিনী': 'army military',
            'বিরুদ্ধে': 'against',
            'পরিকল্পিত': 'planned',
            'ষড়যন্ত্র': 'conspiracy',
            'রুখতে': 'stop prevent',
            'হবে': 'will',
            'খবর': 'news',
            'রাজনীতি': 'politics',
            'অর্থনীতি': 'economy',
            'শিক্ষা': 'education',
            'স্বাস্থ্য': 'health',
            'ক্রীড়া': 'sports',
            'বিনোদন': 'entertainment',
            'প্রযুক্তি': 'technology',
            'পরিবেশ': 'environment',
            'আইন': 'law',
            'নিরাপত্তা': 'security',
            'দুর্নীতি': 'corruption',
            'চুরি': 'theft',
            'হত্যা': 'murder',
            'দুর্ঘটনা': 'accident',
            'বন্যা': 'flood',
            'ভূমিকম্প': 'earthquake',
            'আগুন': 'fire',
            'বোমা': 'bomb',
            'আত্মঘাতী': 'suicide',
            'সন্ত্রাস': 'terrorism'
        }
        
        keywords = []
        for bengali_word, english_words in keyword_mappings.items():
            if bengali_word in bengali_text:
                keywords.extend(english_words.split())
        
        return ' '.join(keywords) if keywords else ''
    
    def generate_bengali_search_queries(self, claim: str) -> List[str]:
        """Generate Bengali search query variations"""
        try:
            # Create simple Bengali query variations manually to avoid API issues
            queries = []
            
            # Original claim - keep it as is
            queries.append(claim)
            
            # Extract key words from the claim for better search
            # Split by spaces, not by characters
            words = claim.split()
            if len(words) > 1:
                # Use first few words
                if len(words) >= 3:
                    queries.append(' '.join(words[:3]))
                # Use last few words  
                if len(words) >= 3:
                    queries.append(' '.join(words[-3:]))
                # Use middle words
                if len(words) > 4:
                    queries.append(' '.join(words[1:-1]))
            
            # Remove duplicates and empty queries
            unique_queries = []
            for q in queries:
                q = q.strip()
                if q and q not in unique_queries and len(q) > 2:
                    unique_queries.append(q)
            
            print(f"Generated Bengali queries: {unique_queries}")
            return unique_queries[:3]  # Limit to 3 queries
            
        except Exception as e:
            print(f"Error generating Bengali queries: {e}")
            return [claim]  # Fallback to original claim

class SemanticNewsRetriever:
    """Semantic search for news articles using embeddings"""
    
    def __init__(self, api_key: str, num_results: int = 20):
        self.api_key = api_key
        self.num_results = num_results
        # Load a multilingual sentence transformer model
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.index = None
        self.documents = []
        
    def search_news(self, query: str) -> List[Document]:
        """Search for news articles using semantic similarity"""
        try:
            # First, get raw search results from Serper
            raw_results = self._get_raw_search_results(query)
            if not raw_results:
                return []
            
            # Create embeddings for the query and documents
            query_embedding = self.model.encode([query])
            doc_texts = [f"{result['title']} {result['snippet']}" for result in raw_results]
            doc_embeddings = self.model.encode(doc_texts)
            
            # Calculate similarities
            similarities = np.dot(query_embedding, doc_embeddings.T).flatten()
            
            # Sort by similarity and get top results
            top_indices = np.argsort(similarities)[::-1][:self.num_results]
            
            documents = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Threshold for relevance
                    result = raw_results[idx]
                    content = f"{result.get('title', '')}\n{result.get('snippet', '')}"
                    documents.append(Document(
                        page_content=content, 
                        metadata={
                            "source": result.get("link", ""),
                            "similarity": float(similarities[idx]),
                            "title": result.get('title', '')
                        }
                    ))
            
            print(f"Found {len(documents)} semantically relevant documents")
            return documents
            
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def _get_raw_search_results(self, query: str) -> List[Dict]:
        """Get raw search results from Serper API"""
        try:
            _SERPER_SEARCH_URL = "https://google.serper.dev/search"
            
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": f'"{query}" (news OR article OR report) -opinion -editorial',
                "num": 50,  # Get more results for better semantic filtering
            }
            
            resp = requests.post(_SERPER_SEARCH_URL, headers=headers, json=payload, timeout=10)
            
            if resp.status_code != 200:
                raise Exception(f"Serper API Error: {resp.text}")
            
            results = resp.json()
            return results.get("organic", [])
            
        except Exception as e:
            print(f"Raw search error: {e}")
            return []

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
                "q": f'"{query}" (news OR article OR report) -opinion -editorial',
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
        
       # Detect if input is in Bengali
       is_bengali = any('\u0980' <= char <= '\u09FF' for char in claim1)
       
       if is_bengali:
           print("Detected Bengali input - using Bengali-specific search")
           return verify_bengali_news(claim1)
       else:
           print("Non-Bengali input - using English search with translation")
           return verify_english_news(claim1, input_lang)
    except Exception as e:
        print('Error in main verification: ', e)
        error_msg = 'Something went wrong. Please try after some time'
        return(error_msg, error_msg, error_msg)

def verify_bengali_news(claim: str):
    """Verify Bengali news claims using Bengali-specific search"""
    try:
        # Use Bengali semantic retriever
        bengali_retriever = BengaliSemanticRetriever(api_key=serp_dev_api_key, num_results=5)
        
        # Generate Bengali search queries
        queries = bengali_retriever.generate_bengali_search_queries(claim)
        
        # Search directly with the generated queries instead of using RunnableLambda
        all_documents = []
        print(f"Generated Bengali search queries: {queries}")
        
        for query in queries:
            try:
                docs = bengali_retriever.search_bengali_news(query.strip())
                print(f"Found {len(docs)} Bengali results for query: {query}")
                all_documents.extend(docs)
            except Exception as e:
                print(f"Bengali search failed for query '{query}': {e}")
                continue
        
        # Remove duplicates based on source URL
        seen_sources = set()
        unique_docs = []
        for doc in all_documents:
            source = doc.metadata.get('source', '')
            if source not in seen_sources:
                seen_sources.add(source)
                unique_docs.append(doc)
        
        print(f"Total unique Bengali documents found: {len(unique_docs)}")
        
        # Limit to top 5 documents to stay within token limits
        limited_docs = unique_docs[:5]
        print(f"Using {len(limited_docs)} documents for processing")
        
        # Create a simple context retriever that returns the documents
        def simple_context_retriever(input_data):
            return limited_docs
        
        context_retriever = RunnableLambda(simple_context_retriever)
        
        # Bengali-specific summarizer
        bengali_summarizer_template = '''
           তুমি একজন বাংলা ভাষার বিশেষজ্ঞ। তোমার কাজ হলো খবরের নিবন্ধ থেকে মূল ঘটনা বের করা।
       
           দাবি: {question}
           
           নিবন্ধসমূহ:
           {context}
           
           শুধুমাত্র ঘটনামূলক তথ্য বের করো। সংক্ষিপ্ত সারসংক্ষেপ দাও।
        '''
        
        summarizer_prompt = PromptTemplate.from_template(bengali_summarizer_template)
        llm_summarizer = ChatGroq(api_key = grok_api_key, model_name = model_summarizer)
        
        summarizer_chain = (
            {
                "context": context_retriever,
                "question": RunnablePassthrough()
            }
            | summarizer_prompt
            | llm_summarizer
            | StrOutputParser()
        )
        
        # Bengali fact-checker
        bengali_fact_checker_template = '''
           তুমি একজন সত্যতা যাচাইকারী সহায়ক।
           
           দাবি: {question}
           
           প্রমাণ:
           {evidence}
           
           নির্দেশনা:
           - শক্ত প্রমাণ পেলে REAL
           - বিপরীত প্রমাণ পেলে FAKE  
           - অপর্যাপ্ত প্রমাণ পেলে UNSURE
           
           উত্তর:
           Classification: REAL or FAKE or UNSURE
           Explanation: <সংক্ষিপ্ত যুক্তি>
        '''
        
        fact_checker_prompt = PromptTemplate.from_template(bengali_fact_checker_template)
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
        
        verdict_orig = fact_checker_chain.invoke(claim)
        verdict_class = verdict_orig.split('\n')[0]
        verdict_explan = verdict_orig.split('\n')[-1]
        
        verdict_orig = verdict_class + '\n\n' + verdict_explan
        verdict_trans = verdict_orig  # Already in Bengali
        
        return(claim, verdict_orig, verdict_trans)
        
    except Exception as e:
        print('Error in Bengali verification: ', e)
        error_msg = 'কিছু সমস্যা হয়েছে। দয়া করে আবার চেষ্টা করুন'
        return(error_msg, error_msg, error_msg)

def verify_english_news(claim: str, input_lang = 'auto'):
    """Verify non-Bengali news claims using English search with translation"""
    try:
       print('here1: ', claim)
       claim1 = claim.replace("'","")
       claim1 = claim.replace("\n"," ")
       print('here2: ', claim1)
        
       # Use semantic search instead of keyword search
       semantic_retriever = SemanticNewsRetriever(api_key=serp_dev_api_key, num_results=20)
       
       def semantic_search_multiple_queries(queries):
           """Search for multiple query variations using semantic similarity"""
           all_documents = []
           print(f"Generated search queries: {queries}")
           
           # Fallback to old search method if semantic search fails
           fallback_retriever = SerperRetrieverWrapper(api_key=serp_dev_api_key)
           
           for query in queries:
               try:
                   docs = semantic_retriever.search_news(query.strip())
                   print(f"Found {len(docs)} semantically relevant results for query: {query}")
                   all_documents.extend(docs)
               except Exception as e:
                   print(f"Semantic search failed for query '{query}': {e}")
                   # Fallback to keyword search
                   try:
                       docs = fallback_retriever.get_relevant_documents(query.strip())
                       print(f"Fallback: Found {len(docs)} keyword-based results for query: {query}")
                       all_documents.extend(docs)
                   except Exception as fallback_e:
                       print(f"Both semantic and keyword search failed for query '{query}': {fallback_e}")
                       continue
           
           # Remove duplicates based on source URL
           seen_sources = set()
           unique_docs = []
           for doc in all_documents:
               source = doc.metadata.get('source', '')
               if source not in seen_sources:
                   seen_sources.add(source)
                   unique_docs.append(doc)
           
           print(f"Total unique documents found: {len(unique_docs)}")
           return unique_docs[:20]  # Limit to top 20 results
       
       context_retriever = RunnableLambda(semantic_search_multiple_queries)
       
       #Multi Query Generation
       multi_query_template = """You are an AI language model assistant. Your task is to generate three 
       different search queries for fact-checking the given news claim. Create variations that will help 
       find relevant news articles and reports about this topic. Focus on different aspects like:
       1. The main event/claim
       2. Key people or organizations mentioned
       3. Location or time period if relevant
       
       Provide these alternative search queries separated by newlines. Original claim: {question}"""
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
          You are an assistant summarizing factual evidence from multiple news articles.
       
          Based on the following documents, extract the key facts relevant to the claim.
          Focus on factual information that directly supports or contradicts the claim.
          
          Claim: {question}
          
          Documents:
          {context}
          
          Instructions:
          - Extract only factual information, not opinions
          - Note the source and date when available
          - Identify any contradictions between sources
          - Focus on verifiable facts related to the claim
          
          Return a concise summary of the key facts found.
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
          You are a fact-checking assistant. Your job is to determine if a news claim is accurate based on available evidence.
          
          Claim: {question}
          
          Evidence:
          {evidence}
          
          Instructions:
          - If you find strong evidence supporting the claim from reliable news sources, classify as REAL
          - If you find evidence contradicting the claim from reliable sources, classify as FAKE  
          - If the evidence is insufficient, unclear, or conflicting, classify as UNSURE
          - Consider the credibility and recency of sources
          - Be conservative - when in doubt, choose UNSURE rather than making assumptions
          
          Respond in this format:
          Classification: REAL or FAKE or UNSURE
          Explanation: <your detailed reasoning based on the evidence>
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