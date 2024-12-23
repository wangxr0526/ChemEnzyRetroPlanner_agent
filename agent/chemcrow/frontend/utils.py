import re
import requests
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from rdkit import Chem
from rdkit.Chem import rdChemReactions, Draw
import io
from PIL import Image
import base64
from io import BytesIO


def cdk(smiles):
    """
    Get a depiction of some smiles.
    """

    url = "https://www.simolecule.com/cdkdepict/depict/wob/svg"
    headers = {"Content-Type": "application/json"}
    response = requests.get(
        url,
        headers=headers,
        params={
            "smi": smiles,
            "annotate": "colmap",
            "zoom": 2,
            "w": 150,
            "h": 80,
            "abbr": "off",
        },
    )
    return response.text


def reaction_smiles_to_png(reaction_smiles):
    try:
        # Parse the reaction SMILES to a reaction object
        rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        if not rxn:
            raise ValueError("Invalid reaction SMILES string")

        # Set up the drawing object for PNG output
        drawer = Draw.MolDraw2DCairo(800, 400)  # Width: 400, Height: 200
        drawer.DrawReaction(rxn, highlightByReactant=True)
        drawer.FinishDrawing()

        # Get the PNG data
        byte_io = BytesIO(drawer.GetDrawingText())
        byte_io.seek(0)
        b64_data = base64.b64encode(byte_io.read()).decode()
        return f"data:image/png;base64,{b64_data}"

    except Exception as e:
        return f"Error processing reaction SMILES: {str(e)}"


def molecule_smiles_to_png(molecule_smiles):
    try:
        # Parse the molecule SMILES to a molecule object
        mol = Chem.MolFromSmiles(molecule_smiles)
        if not mol:
            raise ValueError("Invalid molecule SMILES string")

        # Set up the drawing object for PNG output
        drawer = Draw.MolDraw2DCairo(400, 400)  # Width: 400, Height: 400
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        # Get the PNG data
        byte_io = BytesIO(drawer.GetDrawingText())
        byte_io.seek(0)
        b64_data = base64.b64encode(byte_io.read()).decode()
        return f"data:image/png;base64,{b64_data}"

    except Exception as e:
        return f"Error processing molecule SMILES: {str(e)}"


def parse_easifa_results(text):
    results_id, uniprot_id = None, None
    pattern1 = r"\n\n\nresults_id:*(.*?)\n\n\n"
    match1 = re.search(pattern1, text)
    if match1:
        results_id = match1.group(1)

    pattern2 = r"this reaction, with the UniProt ID of (\w+)\."
    match2 = re.search(pattern2, text)
    if match2:
        uniprot_id = match2.group(1)

    return results_id, uniprot_id
