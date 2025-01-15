import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from langchain.chat_models import ChatOpenAI, ChatOllama
from rmrkl.agent import ChatZeroShotOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from .prompts import (FINAL_ANSWER_ACTION, FINAL_ANSWER_ACTION_END, FINAL_ANSWER_ACTION_SUPPLEMENT1, FINAL_ANSWER_ACTION_SUPPLEMENT2, FORMAT_INSTRUCTIONS,
                      QUESTION_PROMPT, SUFFIX)


class RetroPlannerChatZeroShotOutputParser(ChatZeroShotOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:

        # Remove 'Thought' SUFFIX
        if text.startswith('Thought:'):
            text = text[8:]
        if "Observation" in text:
            text = text.split('Observation')[0]

        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        
        elif FINAL_ANSWER_ACTION_SUPPLEMENT1 in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION_SUPPLEMENT1)[-1].strip()}, text
            )

        elif FINAL_ANSWER_ACTION_SUPPLEMENT2 in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION_SUPPLEMENT2)[-1].strip()}, text
            )
        elif FINAL_ANSWER_ACTION_END in text:
            return AgentFinish(
                {"output": text}, text
            )


        # \s matches against tab/newline/whitespace
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(action, action_input.strip(" ").strip('"'), text.strip())
        # return AgentAction(action, action_input.split('\n')[0].strip(" ").strip('"'), text.strip())



