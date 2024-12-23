"""Wrapper for RXN4Chem functionalities."""

import json
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
import requests
import pandas as pd
from io import StringIO

from chemcrow.utils import (
    is_multiple_reaction,
    is_reaction,
    is_smiles,
    is_valid_ec_number,
)

__all__ = [
    "RetroPlanner",
    "RetroPlannerSingleStep",
    "ReactionRater",
    "ConditionPredictor",
    "EnzymaticRXNIdentifier",
    "EnzymeRecommender",
    "EasIFAAnotator",
]


class RetroPlanner(BaseTool):
    """Use RetroPlanner predict retrosynthesis."""

    name = "RetroPlannerRetrosynthesis"
    description = (
        "Obtain the synthetic route to a chemical compound. "
        "Takes as input the SMILES of the product, returns recipe."
    )

    base_url: str = None
    headers: dict = dict()
    llm: BaseChatModel = None

    def __init__(self, base_url=None, llm=None):
        super().__init__()
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "http://localhost:8001/retroplanner/api/retroplanner"
        self.headers = {"Content-Type": "application/json"}
        self.llm = llm

    def _summary_gpt(self, json: dict) -> str:
        """Describe synthesis."""
        if self.llm:
            llm = self.llm
        else:
            llm = ChatOllama(  # type: ignore
                temperature=0.05,
                model="llama3.1:70b",
                base_url="http://localhost:11434",
                cache=False,
            )
        prompt = (
            "Here is a chemical synthesis described as a json.\nYour task is "
            "to describe the synthesis, as if you were giving instructions for"
            "a recipe. Only use the substances, quantities, temperatures, and"
            "any general operations mentioned in the JSON file, and pay attention"
            "to whether it is an enzymatic reaction. This is your "
            "only source of information, do not make up anything else. Also, "
            "add 15mL of DCM as a solvent in the first step. If a reaction step is enzymatic, simply notify the user: 'The tool suggests that this step can be completed with an appropriate enzyme, but the provided steps are still for a standard organic reaction.' If you ever need "
            'to refer to the json file, refer to it as "(by) the tool". '
            "However avoid references to it. \nFor this task, give as many "
            f"details as possible.\n {str(json)}"
        )
        return llm([HumanMessage(content=prompt)]).content

    def _preproc_reactions(self, tree):
        steps = {}
        stack = [(tree, None)]  # 栈中保存 (当前节点, 父节点)，初始父节点为 None
        step_counter = 1  # 用来计数步骤的序号

        while stack:
            node, parent = stack.pop()  # 取出栈顶的节点及其父节点

            # 如果当前节点是分子节点，检查是否有子节点
            if node["type"] == "mol" and node.get("children"):
                # 将子节点与当前节点作为父节点一起加入栈中（逆序放入保持顺序）
                for child in reversed(node["children"]):
                    stack.append((child, node))  # 将分子节点作为父节点传递给其子节点

            # 如果当前节点是反应节点，处理反应步骤
            if node["type"] == "reaction" and parent:
                # 提取反应物
                reactants = [
                    child["smiles"]
                    for child in node["children"]
                    if child["type"] == "mol"
                ]
                buyable_reactants = {
                    child["smiles"]: child["in_stock"]
                    for child in node["children"]
                    if child["type"] == "mol"
                }

                rxn_attribute = node["rxn_attribute"]
                condition_df = pd.read_json(StringIO(rxn_attribute["condition"]))
                rxn_classification_df = pd.read_json(
                    StringIO(rxn_attribute["organic_enzyme_rxn_classification"])
                )

                # 将反应步骤信息添加到结果字典，使用 step-1, step-2... 作为键
                steps[f"Step_{step_counter}"] = {
                    "This step applied reaction template": ">>".join(
                        reversed(node["template"].split(">>"))
                    ),
                    "reactants": reactants,
                    "Whether the reactants are available for purchase": buyable_reactants,
                    "product": parent["smiles"],
                    "Temperature": int(condition_df["Temperature"].tolist()[0]),
                    "Solvent": condition_df["Solvent"].tolist()[0],
                    "Reagent": condition_df["Reagent"].tolist()[0],
                    "Catalyst": condition_df["Catalyst"].tolist()[0],
                    "Step reaction type": rxn_classification_df[
                        "Reaction Type"
                    ].tolist()[0],
                }
                for child in reversed(node["children"]):
                    stack.append((child, node))  # 将分子节点作为父节点传递给其子节点
                step_counter += 1  # 步骤计数器加1
        steps_list = sorted(
            list(steps.items()), key=lambda x: int(x[0].split("_")[-1]), reverse=True
        )
        steps = {}
        for idx, (_, step_info) in enumerate(steps_list):
            steps[f"Step_{idx+1}"] = step_info
        return steps

    def _run(self, smiles: str) -> str:
        if not is_smiles(smiles):
            return "Smiles is Not Valid."

        data = {"smiles": smiles, "savedOptions": {}}
        response = requests.post(
            self.base_url, headers=self.headers, data=json.dumps(data)
        )

        # curl -X POST http://localhost:8001/retroplanner/api/retroplanner -H "Content-Type: application/json" -d '{"smiles": "CCCCOCCCCC", "savedOptions":{}}'

        if response.status_code == 200:
            data_dict = response.json()

            if "routes" in data_dict:
                reaction_trees = data_dict["routes"]
                results_id = data_dict["results_id"]
                num_routes = len(reaction_trees)

                top_routes_summary = self._preproc_reactions(reaction_trees[0])

                return (
                    f"RetroPlanner found a total of {num_routes} synthetic routes,"
                    f"and the overview of the optimal synthetic route is as follows:\n{self._summary_gpt(json.dumps(top_routes_summary))}\n"
                    f"Results ID: {results_id}"
                )
            else:
                raise KeyError
        elif response.status_code == 400:
            data_dict = response.json()
            if "error" in data_dict:
                return data_dict["error"]
            else:
                KeyError
        else:
            raise KeyError

    async def _arun(self, smiles):
        raise NotImplementedError("Async not implemented.")


class RetroPlannerSingleStep(BaseTool):
    """Use RetroPlannerSingleStep for retrosynthesis."""

    name = "RetroPlannerSingleStep"
    description = (
        "Obtain the reactants that can synthesize the target compound. "
        "Takes as input the SMILES of the product, returns the one most likely reaction SMILES to obtain this target compound."
        "This tool only provides single-step prediction."
        "This tool concatenates the predicted reaction SMILES and the target product SMILES into a reaction SMILES format as follows: 'PredictedReactantsSMILES>>TargetCompoundSMILES'."
    )

    base_url: str = None
    headers: dict = dict()

    def __init__(self, base_url=None):
        super().__init__()
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "http://localhost:8001/retroplanner/api/single_step"
        self.headers = {"Content-Type": "application/json"}

    def _concatenate_to_reactions(self, target_smiles, reactants_list):

        rxn_smiles_list = []
        for reactants in reactants_list:
            rxn_smiles_list.append(f"{reactants}>>{target_smiles}")

        return "\n" + "\n".join(rxn_smiles_list) + "\n"

    def _run(self, smiles: str) -> str:
        if not is_smiles(smiles):
            return "Smiles is Not Valid."

        data = {
            "smiles": smiles,
            "savedOptions": {"topk": 3, "oneStepModel": ["Reaxys"]},
        }
        response = requests.post(
            self.base_url, headers=self.headers, data=json.dumps(data)
        )

        # curl -X POST http://localhost:8001/retroplanner/api/single_step -H "Content-Type: application/json" -d '{"smiles": "CCCCOCCCCC", "savedOptions":{"topk":10, "oneStepModel":["Reaxys"]}}'

        if response.status_code == 200:
            data_dict = response.json()

            if "one_step_results" in data_dict:
                one_step_results = data_dict["one_step_results"]
                reactants_list = one_step_results["reactants"]
                num_results = len(reactants_list)

                return f"The RetroPlannerSingleStepRetrosynthesis tool has identified the {num_results} most likely reactions to produce the given target compound, with the reaction equations as follows:\n{self._concatenate_to_reactions(smiles, reactants_list[:1])}"
            else:
                raise KeyError
        elif response.status_code == 400:
            data_dict = response.json()
            if "error" in data_dict:
                return data_dict["error"]
            else:
                KeyError
        else:
            raise KeyError

    async def _arun(self, smiles):
        raise NotImplementedError("Async not implemented.")


class ReactionRater(BaseTool):
    """Use ReactionRater to determine if a reaction is feasible and can occur."""

    name = "ReactionRater"
    description = (
        "Input the SMILES of a reaction and evaluate whether the reaction is feasible."
        "The appropriate input for this tool is 'ReactantsSMILES>>ProductsSMILES'."
    )

    base_url: str = None
    headers: dict = dict()
    threshold: float = 0.75

    def __init__(self, base_url=None):
        super().__init__()
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "http://localhost:8001/retroplanner/api/reaction_rater"
        self.headers = {"Content-Type": "application/json"}
        # self.threshold = 0.85

    def _run(self, rxn_smiles: str) -> str:
        if "\n" in rxn_smiles:
            rxn_smiles = rxn_smiles.split("\n")[0]
        if not is_reaction(rxn_smiles):
            if is_multiple_reaction(rxn_smiles):
                return "It seems that multiple reaction SMILES have been inputted; this tool only supports the evaluation of a single reaction."
            return "Reaction Smiles is Not Valid."

        data = {"reaction": rxn_smiles}
        response = requests.post(
            self.base_url, headers=self.headers, data=json.dumps(data)
        )

        # curl -X POST http://localhost:8001/retroplanner/api/reaction_rater -H "Content-Type: application/json" -d '{"reaction": "CCO.CC(=O)O>>CN"}'

        if response.status_code == 200:
            data_dict = response.json()

            if "results" in data_dict:
                results = data_dict["results"]

                _, confidence = results["reaction_is_feasible"], results["confidence"]
                reaction_is_feasible = True if confidence > self.threshold else False
                feasible_str = "feasible" if reaction_is_feasible else "not feasible"
                return f"ReactionRater has determined the feasibility of the entered reaction, using {self.threshold} as the threshold. The reaction is deemed {feasible_str} with a confidence score of {confidence:.4f}."
            else:
                raise KeyError
        elif response.status_code == 400:
            data_dict = response.json()
            if "error" in data_dict:
                return data_dict["error"]
            else:
                KeyError
        else:
            raise KeyError

    async def _arun(self, rxn_smiles):
        raise NotImplementedError("Async not implemented.")


class ConditionPredictor(BaseTool):
    """Use ConditionPredictor to recommand the suitable reaction conditions."""

    name = "ConditionPredictor"
    description = (
        "Input a reaction's SMILES, and return the three most likely reaction conditions."
        " Each set should include temperature, catalyst, reagent, and solvent, along with"
        " a confidence score for each set of reaction conditions."
        "The appropriate input for this tool is 'ReactantsSMILES>>ProductsSMILES'."
    )

    base_url: str = None
    headers: dict = dict()
    threshold: float = 0.75

    def __init__(self, base_url=None):
        super().__init__()
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "http://localhost:8001/retroplanner/api/condition_predictor"
        self.headers = {"Content-Type": "application/json"}
        # self.threshold = 0.85

    def _run(self, rxn_smiles: str) -> str:
        if "\n" in rxn_smiles:
            rxn_smiles = rxn_smiles.split("\n")[0]
        if not is_reaction(rxn_smiles):
            if is_multiple_reaction(rxn_smiles):
                return "It seems that multiple reaction SMILES have been inputted; this tool only supports the evaluation of a single reaction."
            return "Reaction Smiles is Not Valid."

        data = {"reaction": rxn_smiles, "return_dataframe": True}
        response = requests.post(
            self.base_url, headers=self.headers, data=json.dumps(data)
        )

        # curl -X POST http://localhost:8001/retroplanner/api/reaction_rater -H "Content-Type: application/json" -d '{"reaction": "CCO.CC(=O)O>>CN"}'

        if response.status_code == 200:
            data_dict = response.json()

            if "results" in data_dict:
                results = data_dict["results"]
                condition_df: pd.DataFrame = pd.read_json(
                    StringIO(results["condition_df"])
                )
                return "\n" + condition_df.to_markdown(index=False)

            else:
                raise KeyError
        elif response.status_code == 400:
            data_dict = response.json()
            if "error" in data_dict:
                return data_dict["error"]
            else:
                KeyError
        else:
            raise KeyError

    async def _arun(self, rxn_smiles):
        raise NotImplementedError("Async not implemented.")


class EnzymaticRXNIdentifier(BaseTool):
    """Use EnzymaticRXNIdentifier to determine whether a reaction can be catalyzed by an enzyme."""

    name = "EnzymaticRXNIdentifier"
    description = (
        "Input the SMILES of the reaction to return the type of reaction and the confidence level of that type. The returned types may be: 'Organic Reaction' or 'Enzymatic Reaction'"
        "The appropriate input for this tool is 'ReactantsSMILES>>ProductsSMILES'."
    )

    base_url: str = None
    headers: dict = dict()

    def __init__(self, base_url=None):
        super().__init__()
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "http://localhost:8001/retroplanner/api/enzymatic_rxn_identifier"
        self.headers = {"Content-Type": "application/json"}

    def _summary_results(self, rxn_type, confidence):

        summary_str = "EnzymaticRXNIdentifier determines the reaction as an '{}' with a confidence of {:.4f}, "

        if rxn_type == "Enzymatic Reaction" and confidence > 0.8:
            return (
                summary_str.format(rxn_type, confidence)
                + "suggesting that it is very likely that an enzyme exists that can catalyze this reaction."
            )
        elif (
            rxn_type == "Enzymatic Reaction"
            and (0.5 <= confidence)
            and (confidence <= 0.8)
        ):
            return (
                summary_str.format(rxn_type, confidence)
                + "suggesting that it is possible an enzyme exists that can catalyze this reaction."
            )

        elif rxn_type == "Organic Reaction" and confidence > 0.8:
            return (
                summary_str.format(rxn_type, confidence)
                + "indicating it is difficult to find an enzyme that can catalyze this reaction, or this reaction has already been well-developed using conventional organic synthesis methods, thus not recommending enzymatic catalysis."
            )

        elif (
            rxn_type == "Organic Reaction"
            and (0.5 <= confidence)
            and (confidence <= 0.8)
        ):
            return (
                summary_str.format(rxn_type, confidence)
                + "indicating it is somewhat difficult to find an enzyme that can catalyze this reaction, or this reaction has been relatively well-developed using conventional organic synthesis methods, thus not highly recommending enzymatic catalysis."
            )
        else:
            ValueError

    def _run(self, rxn_smiles: str) -> str:
        if "\n" in rxn_smiles:
            rxn_smiles = rxn_smiles.split("\n")[0]
        if not is_reaction(rxn_smiles):
            return "Reaction Smiles is Not Valid."

        data = {"reaction": rxn_smiles}
        response = requests.post(
            self.base_url, headers=self.headers, data=json.dumps(data)
        )

        # curl -X POST http://localhost:8001/retroplanner/api/enzymatic_rxn_identifier -H "Content-Type: application/json" -d '{"reaction": "CCO.CC(=O)O>>CC(=O)OCC"}'
        if response.status_code == 200:
            data_dict = response.json()

            if "results" in data_dict:
                results = data_dict["results"]
                rxn_type, confidence = results["reaction type"], results["confidence"]
                return self._summary_results(rxn_type, confidence)
            else:
                raise KeyError
        elif response.status_code == 400:
            data_dict = response.json()
            if "error" in data_dict:
                return data_dict["error"]
            else:
                KeyError
        else:
            raise KeyError

    async def _arun(self, rxn_smiles):
        raise NotImplementedError("Async not implemented.")


class EnzymeRecommender(BaseTool):
    """Use EnzymeRecommender to determine whether a reaction can be catalyzed by an enzyme."""

    name = "EnzymeRecommender"
    description = (
        "Input the SMILES for an 'Enzymatic Reaction' and return the possible categories of enzymes (EC Number) that could catalyze this reaction."
        "The appropriate input for this tool is 'ReactantsSMILES>>ProductsSMILES'."
    )

    base_url: str = None
    headers: dict = dict()

    def __init__(self, base_url=None):
        super().__init__()
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "http://localhost:8001/retroplanner/api/enzyme_recommender"
        self.headers = {"Content-Type": "application/json"}

    def _summary_results(self, recommended_enzyme_type_list, confidences_list):

        summary_str = "EnzymeRecommender has recommended five types of enzymes most likely to catalyze this enzymatic reaction. The enzyme with the highest confidence has an EC Number of {}, with a confidence score of {:.4f}. The five predicted results are as follows:\n\n".format(
            recommended_enzyme_type_list[0], confidences_list[0]
        )

        summary_df = pd.DataFrame(
            {
                "EC Number": recommended_enzyme_type_list,
                "Confidence": confidences_list,
            }
        )
        summary_str += summary_df.to_markdown(index=False)

        # summary_str += 'EC Number\tconfidence\n'

        # for ec, confidence in zip(recommended_enzyme_type_list, confidences_list):
        #     summary_str += f'{ec}\t{confidence:.4f}\n'
        return summary_str

    def _run(self, rxn_smiles: str) -> str:
        if "\n" in rxn_smiles:
            rxn_smiles = rxn_smiles.split("\n")[0]
        if not is_reaction(rxn_smiles):
            return "Reaction Smiles is Not Valid."

        data = {"reaction": rxn_smiles}
        response = requests.post(
            self.base_url, headers=self.headers, data=json.dumps(data)
        )

        # curl -X POST http://localhost:8001/retroplanner/api/enzyme_recommender -H "Content-Type: application/json" -d '{"reaction": "CCO.CC(=O)O>>CC(=O)OCC"}'

        if response.status_code == 200:
            data_dict = response.json()

            if "results" in data_dict:
                results = data_dict["results"]
                recommended_enzyme_type_list, confidences_list = (
                    results["recommended enzyme type"][0],
                    results["confidence"][0],
                )
                return self._summary_results(
                    recommended_enzyme_type_list, confidences_list
                )
            else:
                raise KeyError
        elif response.status_code == 400:
            data_dict = response.json()
            if "error" in data_dict:
                return data_dict["error"]
            else:
                KeyError
        else:
            raise KeyError

    async def _arun(self, rxn_smiles):
        raise NotImplementedError("Async not implemented.")


class EasIFAAnotator(BaseTool):
    """Use EasIFAAnotator to download the enzyme structure corresponding to an enzymatic reaction, and annotate the catalytic active site of this potential active enzyme."""

    name = "EasIFAAnotator"
    description = (
        "Enter the SMILES of the enzymatic reaction and the EC number of the enzyme that may catalyze the reaction. The tool will automatically download the structure of the enzyme and return the UniProt ID of the enzyme, possible active sites, and types of active sites."
        "The appropriate input for reaction SMILES is 'ReactantsSMILES>>ProductsSMILES??ECNumber' (example: 'C1CCCCCC1=O>>C1CCCCCC1??1.13.99.1')."
    )

    base_url: str = None
    headers: dict = dict()
    llm: BaseChatModel = None

    def __init__(self, base_url=None, llm=None):
        super().__init__()
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "http://localhost:8001/retroplanner/api/easifa"
        self.headers = {"Content-Type": "application/json"}
        if llm:
            self.llm = llm

    def _summary_enzyme_data(self, enzyme_data, results_id):
        active_data_df = pd.read_json(StringIO(enzyme_data["active_data"]))
        uniprot_id = enzyme_data["alphafolddb_id"]

        summary_str = (
            "Based on the given reaction and the corresponding enzyme's EC number,"
            " EasIFA has identified an enzyme that may have the capability to catalyze"
            f" this reaction, with the UniProt ID of {uniprot_id}. The information about its active"
            " sites is as follows:\n\n"
        )

        summary_str += self._summary_gpt(
            uniprot_id, active_data_df.to_string(index=False)
        )

        summary_str += f"\n\n\nresults_id:{results_id}\n\n\n"

        summary_str += (
            "Enzyme Active Site Informations:\n\n"
            + active_data_df.to_markdown(index=False)
        )

        return summary_str

    def _summary_gpt(self, uniprot_id: str, info: str) -> str:
        """Describe enzyme active sites."""

        if self.llm:
            llm = self.llm
        else:
            llm = ChatOllama(  # type: ignore
                temperature=0.05,
                model="llama3.1:70b",
                base_url="http://localhost:11434",
            )
        prompt = (
            f"This is a description of an enzyme (UniProt ID is {uniprot_id})'s active sites. "
            "The input is a table of the enzyme's active sites. "
            "Please do not omit any active sites. Summarize and "
            "describe the information about these active sites. "
            "Below is a table string of active sites, consisting"
            " of [Residue Index, Residue Name, Active Type], with"
            f" each line representing an active site:\n{info}\n"
            # "Please note that the output should start with 'Final Answer:'"
        )
        return llm([HumanMessage(content=prompt)]).content

    def _run(self, rxn_smiles_with_ec: str) -> str:
        if "\n" in rxn_smiles_with_ec:
            rxn_smiles_with_ec = rxn_smiles_with_ec.split("\n")[0]

        if "??" not in rxn_smiles_with_ec:
            return "Incorrect input. Please connect the reaction SMILES and EC Number using '??'. An example would be: ReactantsSMILES>>ProductsSMILES??X.X.X.X"

        rxn_smiles, ec_number = rxn_smiles_with_ec.split("??")

        if not is_reaction(rxn_smiles):
            return "Reaction Smiles is Not Valid."

        elif not is_valid_ec_number(ec_number):
            return "EC number is Not Valid."

        data = {
            "reaction": rxn_smiles,
            "EC number": ec_number,
            "return_dataframe": True,
        }
        response = requests.post(
            self.base_url, headers=self.headers, data=json.dumps(data)
        )

        # curl -X POST http://localhost:8001/retroplanner/api/easifa -H "Content-Type: application/json" -d '{"reaction": "C1CCCCCC1=O>>C1CCCCCC1", "EC number":"1.13.99.-"}'

        if response.status_code == 200:
            data_dict = response.json()

            if "enzyme_data" in data_dict:
                enzyme_data = data_dict["enzyme_data"][0]

                results_id = data_dict["results_id"]

                return self._summary_enzyme_data(enzyme_data, results_id)

            else:
                raise KeyError
        elif response.status_code == 400:
            data_dict = response.json()
            if "error" in data_dict:
                return data_dict["error"]
            elif "message" in data_dict:
                return data_dict["message"]
            else:
                KeyError
        else:
            raise KeyError

    async def _arun(self, rxn_smiles_with_ec):
        raise NotImplementedError("Async not implemented.")
