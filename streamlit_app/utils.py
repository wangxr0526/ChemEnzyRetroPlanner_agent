from langchain.llms import OpenAI, Ollama
import requests
import json

def oai_key_isvalid(api_key):
    """Check if a given OpenAI key is valid"""
    try:
        llm = OpenAI(openai_api_key = api_key)
        out = llm("This is a test")
        return True
    except:
        return False

def ollama_base_url_isvalid(ollama_base_url):
    """Check if a given Ollama base url is valid"""
    try:
        llm = Ollama(base_url=ollama_base_url, model='llama3.1:70b', cache=False)
        out = llm("This is a test")
        return True
    except:
        return False

def retroplanner_base_url_isvalid(retroplanner_base_url):
    """Check if a given RetroPlanner base url is valid"""
    try:
        headers = {"Content-Type": "application/json"}
        data = {"reaction": 'C>>CO'}
        response = requests.post(
            f'{retroplanner_base_url}/api/reaction_rater', headers=headers, data=json.dumps(data)
        )
        return True
    except:
        return False

