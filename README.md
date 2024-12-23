# ChemEnzyRetroPlanner Agent
This is an Agent paired with the ChemEnzyRetroPlanner API, adapted for Llama 3.1 based on ChemCrow, with partial localization of certain synthesis planning tools.  
## Introduction
![ChemEnzyRetroPlanner](./streamlit_app/assets/agent_intro_b.png)
## Installation Guide
```
git clone https://github.com/wangxr0526/ChemEnzyRetroPlanner_agent.git
cd ChemEnzyRetroPlanner_agent
conda env create -f envs.yml
conda activate retro_planner_agent_env
pip install -r requirements.txt
pip install chemprice
pip install -e ./agent
cd streamlit_app
streamlit run app.py
```

## Interface
![interface](streamlit_app/assets/interface.png)



## Related repository
[ChemEnzyRetroPlanner](https://github.com/wangxr0526/ChemEnzyRetroPlanner.git)