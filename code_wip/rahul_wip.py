#Code for translating text using Sarvam
from sarvamai import SarvamAI

translation = client.text.translate(
    input="यह एक नमूना पाठ है।",
    source_language_code="auto",
    target_language_code="en-HN"
)

#Seacrh API examples:
from serpapi import GoogleSearch

params = {
  "api_key": "da9d2e48d0be67fdb97893198682c3f46925d6948a4ca26dffeaf63dd5439f69",
  "engine": "google",
  "q": "site:news18.com donald trump is president of usa",
  "google_domain": "google.co.in",
  "gl": "in",
  "hl": "en",
  "num": "5"
}

search = GoogleSearch(params)
results = search.get_dict()


params = {
  "api_key": "da9d2e48d0be67fdb97893198682c3f46925d6948a4ca26dffeaf63dd5439f69",
  "engine": "google",
  "q": "site:ptinews.com donald trump is president of usa",
  "google_domain": "google.co.in",
  "gl": "in",
  "hl": "en",
  "num": "5"
}

search = GoogleSearch(params)
results = search.get_dict()


params = {
  "engine": "google",
  "q": "site:politifact.com vaccine causes autism",
  "google_domain": "google.com",
  "gl": "in",
  "hl": "en",
  "num": "5",
  "api_key": "secret_api_key"
}

search = GoogleSearch(params)
results = search.get_dict()