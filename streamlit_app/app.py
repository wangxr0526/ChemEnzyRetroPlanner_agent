import os

# Init with fake key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "none"

import openai
import pandas as pd
import streamlit as st
from IPython.core.display import HTML
from PIL import Image
from langchain.callbacks import wandb_tracing_enabled
from chemcrow.agents import ChemCrow, make_tools
from chemcrow.frontend.streamlit_callback_handler import StreamlitCallbackHandlerChem
from utils import oai_key_isvalid, ollama_base_url_isvalid, retroplanner_base_url_isvalid

from dotenv import load_dotenv

load_dotenv()
ss = st.session_state
ss.prompt = None

icon = Image.open("assets/logo-Logo-beta.png")
st.set_page_config(page_title="ChemEnzyRetroPlanner Agent", page_icon=icon)

# Set width of sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 450px;
        max-width: 450px;
    }
    """,
    unsafe_allow_html=True,
)

agent = ChemCrow(
    model="llama3.1:70b",
    tools_model="llama3.1:70b",
    temp=0.1,
    # ollama_base_url="http://localhost:11434",
    # retroplanner_base_url="http://localhost:8001/retroplanner",
    ollama_base_url=ss.get('ollama_base_url', 'http://localhost:11434'),
    retroplanner_base_url=ss.get('retroplanner_base_url', 'http://localhost:8001/retroplanner'),
    llm_cache=False,
).agent_executor

tools = agent.tools

tool_list = pd.Series({f"✅ {t.name}": t.description for t in tools}).reset_index()
tool_list.columns = ["Tool", "Description"]


def on_api_key_change():
    api_key = ss.get("api_key") or os.getenv("OPENAI_API_KEY")
    # Check if key is valid
    if not oai_key_isvalid(api_key):
        st.write("Please input a valid OpenAI API key.")

def on_ollama_base_url_change():
    ollama_base_url = ss.get("ollama_base_url")

    if not ollama_base_url_isvalid(ollama_base_url):
        st.write("Please input a valid Ollama Base URL.")

def on_retroplanner_base_url_change():
    retroplanner_base_url = ss.get("retroplanner_base_url")

    if not retroplanner_base_url_isvalid(retroplanner_base_url):
        st.write("Please input a valid RetroPlanner Base URL.")


# def run_prompt(prompt):
#     st.chat_message("user").write(prompt)
#     with st.chat_message("assistant"):
#         st_callback = StreamlitCallbackHandlerChem(
#             st.container(),
#             max_thought_containers = 3,
#             collapse_completed_thoughts = False,
#             output_placeholder=ss
#         )
#         try:
#             with wandb_tracing_enabled():
#                 response = agent.run(prompt, callbacks=[st_callback])
#                 st.write(response)
#         except openai.error.AuthenticationError:
#             st.write("Please input a valid OpenAI API key")
#         except openai.error.APIError:
#             # Handle specific API errors here
#             print("OpenAI API error, please try again!")


def run_prompt(prompt):
    st.chat_message("user").write(prompt.replace('[', '\[').replace(']', '\]'))
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandlerChem(
            st.container(),
            max_thought_containers=3,
            collapse_completed_thoughts=False,
            output_placeholder=ss,
            use_rdkit=True,
        )
        try:
            # with wandb_tracing_enabled():
            response = agent.run(prompt, callbacks=[st_callback])
            st.write(response.replace('[', '\[').replace(']', '\]'))
        except openai.error.AuthenticationError:
            st.write("Please input a valid OpenAI API key")
        except openai.error.APIError:
            # Handle specific API errors here
            print("OpenAI API error, please try again!")


pre_prompts = [
    "Help me plan the synthetic route for Ibuprofen",
    (
        "The enzyme that can catalyze the enzyme-catalyzed reaction "
        "(SMILES: O=C(O)C(=O)CO.O=C[C@H](O)CO>>O=C(CO)[C@@H](O)[C@H](O)CO) belongs "
        "to EC Number: 2.2.1.1. Please recommend the most suitable "
        "enzyme structure and indicate its active site."
    ),
    (
        "Determine whether the reaction SMILES: O=C(O)C(=O)CO.O=C[C@H](O)CO>>O=C(CO)[C@@H](O)[C@H](O)CO "
        "is feasible, and if so, assess whether it can be catalyzed by an enzyme."
    ),
    "Tell me how to synthesize O=C(O)c1ccccc1O, and the price of the precursors.",
]

# sidebar
with st.sidebar:
    # chemcrow_logo = Image.open('assets/chemcrow-logo-bold-new.png')
    chemcrow_logo = Image.open("assets/logo-Logo.png")
    st.image(chemcrow_logo)

    # # Input OpenAI api key
    # st.text_input(
    #     'Input your OpenAI API key.',
    #     placeholder = 'Input your OpenAI API key.',
    #     type='password',
    #     key='api_key',
    #     on_change=on_api_key_change,
    #     label_visibility="collapsed"
    # )

    st.text_input(
        "Enter Ollama Base URL",
        value="http://localhost:11434",
        help="URL for the Ollama service",
        key='ollama_base_url',
        on_change=on_ollama_base_url_change,
    )
    st.text_input(
        "Enter ChemEnzyRetroPlanner Base URL",
        value="http://localhost:8001/retroplanner",
        help="URL for the ChemEnzyRetroPlanner service",
        key='retroplanner_base_url',
        on_change=on_retroplanner_base_url_change,
    )

    # Display prompt examples
    st.markdown("# What can I ask?")
    cols = st.columns(2)
    with cols[0]:
        st.button(
            "Help me plan the synthetic route for Ibuprofen.",
            on_click=lambda: run_prompt(pre_prompts[0]),
        )
        st.button(
            "Recommend suitable enzymes for enzyme-catalyzed reactions and indicate the active sites of the enzymes",
            on_click=lambda: run_prompt(pre_prompts[1]),
        )
    with cols[1]:
        st.button(
            "Determine whether the reaction is enzyme-catalyzed and recommend suitable enzymes",
            on_click=lambda: run_prompt(pre_prompts[2]),
        )
        st.button(
            "Synthesize molecule with price of precursors",
            on_click=lambda: run_prompt(pre_prompts[3]),
        )

    st.markdown("---")
    # Display available tools
    st.markdown(f"# {len(tool_list)} available tools")
    st.dataframe(tool_list, use_container_width=True, hide_index=True, height=200)

# Execute agent on user input
if user_input := st.chat_input():
    run_prompt(user_input)
