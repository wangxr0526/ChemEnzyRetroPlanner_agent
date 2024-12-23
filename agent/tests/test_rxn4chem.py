




from chemcrow.tools.rxn4chem import RXNRetrosynthesisOllama


retroplanner_api = RXNRetrosynthesisOllama(rxn4chem_api_key='apk-e7b74cceebd1d84d774c00377b2a5feafd4a184586a7741bcc2a87bbdd5f4886')
results = retroplanner_api._run('C1CCCOCCCC(C=O)C1')
pass