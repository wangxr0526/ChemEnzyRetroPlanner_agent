from paperqa import Settings, ask

local_llm_config = {
    "model_list": [
        {
            "model_name": "ollama/llama3.1:70b",
            "litellm_params": {
                "model": "ollama/llama3.1:70b",
                "api_base": "http://locahost:11434",
            },
        }
    ]
}

answer = ask(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=Settings(
        llm="ollama/llama3.1:70b",
        llm_config=local_llm_config,
        summary_llm="ollama/llama3.1:70b",
        summary_llm_config=local_llm_config,
        embedding="ollama/mxbai-embed-large",
        verbosity=True,
    ),
)

# import os
# from langchain.chat_models import ChatOllama
# from chemcrow.tools.search import Scholar2ResultLLM


# def test_litsearch(questions):
#     llm = ChatOllama(
#         temperature=0.1,
#         model="llama3.1:70b",
#         base_url='http://locahost:11434',
#     )

#     searchtool = Scholar2ResultLLM(llm=llm)
#     for q in questions:
#         ans = searchtool._run(q)
#         assert isinstance(ans, str)
#         assert len(ans) > 0
#     if os.path.exists("../query"):
#         os.rmdir("../query")
        
# if __name__ == '__main__':
    
#     test_litsearch( "What are the effects of norhalichondrin B in mammals?")