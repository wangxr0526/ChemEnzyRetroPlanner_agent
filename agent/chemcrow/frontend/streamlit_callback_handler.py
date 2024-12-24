import re
from typing import Any, Dict, List, Optional
import streamlit as st
from streamlit_modal import Modal
from langchain.callbacks.streamlit.streamlit_callback_handler import (
    CHECKMARK_EMOJI,
    EXCEPTION_EMOJI,
    THINKING_EMOJI,
    LLMThought,
    LLMThoughtLabeler,
    LLMThoughtState,
    StreamlitCallbackHandler,
    ToolRecord,
)
from langchain_core.schema import AgentAction, AgentFinish, LLMResult
from streamlit.delta_generator import DeltaGenerator

from chemcrow.utils import is_reaction, is_smiles

from .utils import (
    cdk,
    parse_easifa_results,
    reaction_smiles_to_png,
    molecule_smiles_to_png,
)


class LLMThoughtChem(LLMThought):
    def __init__(
        self,
        parent_container: DeltaGenerator,
        labeler: LLMThoughtLabeler,
        expanded: bool,
        collapse_on_complete: bool,
        use_rdkit: bool,
        retroplanner_base_url:str = 'http://cadd.zju.edu.cn/retroplanner'
    ):
        super().__init__(
            parent_container,
            labeler,
            expanded,
            collapse_on_complete,
        )
        self.use_rdkit = use_rdkit
        self.retroplanner_base_url = retroplanner_base_url

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        output_ph: dict = {},
        input_tool: str = "",
        serialized: dict = {},
        **kwargs: Any,
    ) -> None:
        # Depending on the tool name, decide what to display.
        if serialized["name"] == "Name2SMILES":
            safe_smiles = output.replace("[", "\[").replace("]", "\]")
            if self.use_rdkit:
                if is_smiles(output):
                    self._container.markdown(
                        f"**{safe_smiles}**<br><img src='{molecule_smiles_to_png(output)}' style='border-radius: 25px;'/>",
                        unsafe_allow_html=True,
                    )
            else:
                if is_smiles(output):
                    self._container.markdown(
                        f"**{safe_smiles}**{cdk(output)}", unsafe_allow_html=True
                    )

        if serialized["name"] == "ReactionPredict":
            rxn = f"{input_tool}>>{output}"
            safe_smiles = rxn.replace("[", "\[").replace("]", "\]")
            if self.use_rdkit:
                self._container.markdown(
                    f"**{safe_smiles}**<br><img src='{reaction_smiles_to_png(rxn)}' style='border-radius: 25px;'/>",
                    unsafe_allow_html=True,
                )
            else:
                self._container.markdown(
                    f"**{safe_smiles}**{cdk(rxn)}", unsafe_allow_html=True
                )

        if serialized["name"] == "ReactionRetrosynthesis":
            output = output.replace("[", "\[").replace("]", "\]")

        if serialized["name"] == "EasIFAAnotator":

            results_id, uniprot_id = parse_easifa_results(output)

            if results_id is not None and uniprot_id is not None:
                # self._container.markdown(st.components.v1.iframe(f"http://localhost:8001/api/enzyme_show/{results_id}&&&&{uniprot_id}", height=600))
                self._container.markdown(
                    f"""
                    <div style="display: flex; border-radius: 25px; overflow: hidden;">
                        <iframe src="{self.retroplanner_base_url}/api/enzyme_show/{results_id}&&&&{uniprot_id}" height="350" width="420" style="border:none; border-radius: 25px;"></iframe>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            split_output = output.split("Enzyme Active Site Informations:\n\n")
            if len(split_output) == 2:
                self._container.markdown(f"### UniProt ID {uniprot_id}")
                self._container.markdown("### Enzyme Active Site Information")
                self._container.markdown(split_output[-1])

        if serialized["name"] == "EnzymeRecommender":

            split_output = output.split(
                "The five predicted results are as follows:\n\n"
            )
            if len(split_output) == 2:
                self._container.markdown("### Enzyme Recommender Results")
                self._container.markdown(split_output[-1])

        if serialized["name"] == "RetroPlannerSingleStep":

            split_output = output.split("with the reaction equations as follows:\n\n")
            if len(split_output) == 2:
                reactions_str = split_output[-1][:-1]
                try:
                    reaction_list = reactions_str.split("\n")
                    self._container.markdown("### Predicted Reactions")
                    for reaction in reaction_list:
                        safe_smiles = reaction.replace("[", "\[").replace("]", "\]")
                        if self.use_rdkit:
                            self._container.markdown(
                                f"**{safe_smiles}**<br><img src='{reaction_smiles_to_png(reaction)}' style='border-radius: 25px;'/>",
                                unsafe_allow_html=True,
                            )
                        else:
                            self._container.markdown(
                                f"**{safe_smiles}**{cdk(reaction)}",
                                unsafe_allow_html=True,
                            )
                except:
                    pass
        if serialized["name"] == "ReactionRater":
            pattern = r"The reaction is deemed (.+?) with a confidence score of ([0-9]+\.[0-9]+)"
            match = re.search(pattern, output)
            if match:
                feasible_str = match.group(1)
                confidence = float(match.group(2))
                self._container.markdown(
                    f"#### {feasible_str.capitalize()}\n Confidence: {confidence}"
                )
        if serialized["name"] == "ConditionPredictor":

            condition_table = output[1:]
            self._container.markdown(f"#### Predicted Reaction Conditions")
            self._container.markdown(condition_table)

        if serialized["name"] == "EnzymaticRXNIdentifier":
            pattern = (
                r"the reaction as an '(.+?)' with a confidence of ([0-9]+\.[0-9]+)"
            )
            match = re.search(pattern, output)
            if match:
                rxn_type = match.group(1)
                confidence = float(match.group(2))
                self._container.markdown(
                    f"#### {rxn_type.capitalize()}\n Confidence: {confidence}"
                )
        if serialized["name"] == "RetroPlannerRetrosynthesis":
            pattern = r"Results ID: (\w+-\w+-\w+-\w+-\w+)"
            match = re.search(pattern, output)
            if match:
                self._container.markdown(output.split('Results ID:')[0], unsafe_allow_html=True)
                results_id = match.group(1)
                button_style = f"""
                <a href="{self.retroplanner_base_url}/results/{results_id}&resultsLimit-20" target="_blank">
                    <button style="color: white; background-color: #3c3c3c; border: none; border-radius: 20px; padding: 8px 16px; box-shadow: 0px 0px 30px rgba(0, 0, 0, 0.15); margin: 10px auto; font-size:16px">
                        Show All Routes
                    </button>
                </a>
                """
                self._container.markdown(button_style, unsafe_allow_html=True)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        # Called with the name of the tool we're about to run (in `serialized[name]`),
        # and its input. We change our container's label to be the tool name.
        self._state = LLMThoughtState.RUNNING_TOOL
        tool_name = serialized["name"]
        self._last_tool = ToolRecord(name=tool_name, input_str=input_str)
        for idx in range(len(self._container._child_records)):
            self._container._child_records[idx].kwargs['body'] = self._container._child_records[idx].kwargs['body'].replace('[', '\[').replace(']', '\]')

        self._container.update(
            new_label=(
                self._labeler.get_tool_label(self._last_tool, is_complete=False)
                .replace("[", "\[")
                .replace("]", "\]")
            )
        )

        # Display note of potential long time
        if (
            serialized["name"] == "ReactionRetrosynthesis"
            or serialized["name"] == "LiteratureSearch"
            or serialized["name"] == "RetroPlannerRetrosynthesis"
            or serialized["name"] == "EasIFAAnotator"
        ):
            self._container.markdown(
                f"‼️ Note: This tool can take some time to complete execution ‼️",
                unsafe_allow_html=True,
            )
        if (
            serialized["name"] == "ReactionRater"
            or serialized["name"] == "EnzymaticRXNIdentifier"
            or serialized["name"] == "EnzymeRecommender"
            or serialized["name"] == "ConditionPredictor"
        ):
            safe_smiles = input_str.replace("[", "\[").replace("]", "\]")
            if self.use_rdkit:
                if is_reaction(input_str):
                    self._container.markdown(
                        f"**{safe_smiles}**<br><img src='{reaction_smiles_to_png(input_str)}' style='border-radius: 25px;'/>",
                        unsafe_allow_html=True,
                    )
            else:
                if is_reaction(input_str):
                    self._container.markdown(
                        f"**{safe_smiles}**{cdk(input_str)}", unsafe_allow_html=True
                    )

        if serialized["name"] == "EasIFAAnotator":
            smiles = input_str.split("??")[0]
            safe_smiles = smiles.replace("[", "\[").replace("]", "\]")
            if self.use_rdkit:
                if is_reaction(input_str):
                    self._container.markdown(
                        f"**{safe_smiles}**<br><img src='{reaction_smiles_to_png(input_str)}' style='border-radius: 25px;'/>",
                        unsafe_allow_html=True,
                    )
            else:
                if is_reaction(smiles):
                    self._container.markdown(
                        f"**{safe_smiles}**{cdk(smiles)}", unsafe_allow_html=True
                    )
        if serialized["name"] == "RetroPlannerSingleStep":
            safe_smiles = input_str.replace("[", "\[").replace("]", "\]")
            if self.use_rdkit:
                if is_smiles(input_str):
                    self._container.markdown(
                        f"**Input: {safe_smiles}**<br><img src='{molecule_smiles_to_png(input_str)}' style='border-radius: 25px;'/>",
                        unsafe_allow_html=True,
                    )
            else:
                if is_smiles(input_str):
                    self._container.markdown(
                        f"**Input: {safe_smiles}**{cdk(input_str)}",
                        unsafe_allow_html=True,
                    )

    def complete(self, final_label: Optional[str] = None) -> None:
        """Finish the thought."""
        if final_label is None and self._state == LLMThoughtState.RUNNING_TOOL:
            assert (
                self._last_tool is not None
            ), "_last_tool should never be null when _state == RUNNING_TOOL"
            final_label = self._labeler.get_tool_label(
                self._last_tool, is_complete=True
            )
        self._state = LLMThoughtState.COMPLETE

        final_label = final_label.replace("[", "\[").replace("]", "\]")
        if self._collapse_on_complete:
            self._container.update(new_label=final_label, new_expanded=False)
        else:
            self._container.update(new_label=final_label)


class StreamlitCallbackHandlerChem(StreamlitCallbackHandler):
    def __init__(
        self,
        parent_container: DeltaGenerator,
        *,
        max_thought_containers: int = 4,
        expand_new_thoughts: bool = True,
        collapse_completed_thoughts: bool = True,
        thought_labeler: Optional[LLMThoughtLabeler] = None,
        output_placeholder: dict = {},
        use_rdkit: bool = False,
        retroplanner_base_url:str = 'http://cadd.zju.edu.cn/retroplanner'
    ):
        super(StreamlitCallbackHandlerChem, self).__init__(
            parent_container,
            max_thought_containers=max_thought_containers,
            expand_new_thoughts=expand_new_thoughts,
            collapse_completed_thoughts=collapse_completed_thoughts,
            thought_labeler=thought_labeler,
        )

        self._output_placeholder = output_placeholder
        self.last_input = ""
        self.use_rdkit = use_rdkit
        self.retroplanner_base_url = retroplanner_base_url

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        if self._current_thought is None:
            self._current_thought = LLMThoughtChem(
                parent_container=self._parent_container,
                expanded=self._expand_new_thoughts,
                collapse_on_complete=self._collapse_completed_thoughts,
                labeler=self._thought_labeler,
                use_rdkit=self.use_rdkit,
                retroplanner_base_url=self.retroplanner_base_url,
            )

        self._current_thought.on_llm_start(serialized, prompts)

        # We don't prune_old_thought_containers here, because our container won't
        # be visible until it has a child.

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        self._require_current_thought().on_tool_start(serialized, input_str, **kwargs)
        self._prune_old_thought_containers()
        self._last_input = input_str
        self._serialized = serialized

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._require_current_thought().on_tool_end(
            output,
            color,
            observation_prefix,
            llm_prefix,
            output_ph=self._output_placeholder,
            input_tool=self._last_input,
            serialized=self._serialized,
            **kwargs,
        )
        self._complete_current_thought()

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        if self._current_thought is not None:

            for idx in range(len(self._current_thought._container._child_records)):
                self._current_thought._container._child_records[idx].kwargs['body'] = self._current_thought._container._child_records[idx].kwargs['body'].replace('[', '\[').replace(']', '\]')
            self._current_thought.complete(
                self._thought_labeler.get_final_agent_thought_label()
                .replace("[", "\[")
                .replace("]", "\]")
            )
            self._current_thought = None
