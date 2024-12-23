import os

from langchain import agents
from langchain.base_language import BaseLanguageModel

from chemcrow.tools import *


def make_tools(
    llm: BaseLanguageModel,
    api_keys: dict = {},
    verbose=True,
    retroplanner_base_url="http://localhost:8001",
):
    serp_api_key = api_keys.get("SERP_API_KEY") or os.getenv("SERP_API_KEY")
    rxn4chem_api_key = api_keys.get("RXN4CHEM_API_KEY") or os.getenv("RXN4CHEM_API_KEY")
    openai_api_key = api_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    chemspace_api_key = api_keys.get("CHEMSPACE_API_KEY") or os.getenv(
        "CHEMSPACE_API_KEY"
    )
    molport_api_key = api_keys.get("MOLPORT_API_KEY") or os.getenv(
        "MOLPORT_API_KEY"
    )
    semantic_scholar_api_key = api_keys.get("SEMANTIC_SCHOLAR_API_KEY") or os.getenv(
        "SEMANTIC_SCHOLAR_API_KEY"
    )

    all_tools = agents.load_tools(
        [
            "python_repl",
            # "ddg-search",
            "wikipedia",
            # "human"
        ]
    )

    all_tools += [
        Query2SMILES(chemspace_api_key),
        Query2CAS(),
        SMILES2Name(),
        PatentCheck(),
        MolSimilarity(),
        SMILES2Weight(),
        FuncGroups(),
        ExplosiveCheck(),
        ControlChemCheck(),
        SimilarControlChemCheck(),
        SafetySummary(llm=llm),
        Scholar2ResultLLM(
            llm=llm,
            openai_api_key=openai_api_key,
            semantic_scholar_api_key=semantic_scholar_api_key,
        ),
        RetroPlanner(base_url=f"{retroplanner_base_url}/api/retroplanner", llm=llm),
        RetroPlannerSingleStep(base_url=f"{retroplanner_base_url}/api/single_step"),
        ReactionRater(base_url=f"{retroplanner_base_url}/api/reaction_rater"),
        EnzymaticRXNIdentifier(
            base_url=f"{retroplanner_base_url}/api/enzymatic_rxn_identifier"
        ),
        EnzymeRecommender(base_url=f"{retroplanner_base_url}/api/enzyme_recommender"),
        EasIFAAnotator(base_url=f"{retroplanner_base_url}/api/easifa", llm=llm),
        ConditionPredictor(base_url=f"{retroplanner_base_url}/api/condition_predictor"),
    ]
    if chemspace_api_key:
        all_tools += [GetMoleculePrice(chemspace_api_key=chemspace_api_key)]
    if molport_api_key:
        all_tools += [GetMoleculePrice(molport_api_key=molport_api_key)]
    if serp_api_key:
        all_tools += [WebSearch(serp_api_key)]
    if rxn4chem_api_key:
        all_tools += [
            RXNPredict(rxn4chem_api_key),
            RXNRetrosynthesis(rxn4chem_api_key, openai_api_key),
        ]

    return all_tools
