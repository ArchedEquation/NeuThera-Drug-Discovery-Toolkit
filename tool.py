import os
import sys
import requests
import ast
import json
import hashlib
import tempfile
import scipy
from datetime import datetime
from glob import glob
from io import StringIO

import pandas as pd
import numpy as np
from db import db
from dotenv import load_dotenv
from arango import ArangoClient

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.llms.bedrock import Bedrock
from langchain_community.graphs import ArangoGraph
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain.tools import Tool

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw, AllChem

import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModel
import streamlit as st
import networkx as nx
from pyvis.network import Network
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_community.graphs import ArangoGraph
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_core.tools import tool
from langchain.tools import Tool

from DeepPurpose import utils
from DeepPurpose import DTI as models
from TamGen_custom import TamGenCustom

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw, AllChem

from Bio.PDB import MMCIFParser

import faiss

import boto3

def text_to_aql(query: str):
    """Execute a Natural Language Query in ArangoDB, and return the result as text."""
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=arango_graph,  # Assuming arango_graph is already initialized
        verbose=True,
        allow_dangerous_requests=True
    )
    
    result = chain.invoke(query)

    return str(result["result"])

def predict_binding_affinity(X_drug, X_target, y=[7.635]):
    """
    Predicts the binding affinity for given drug and target sequences.

    Parameters:
    X_drug (list): List containing the SMILES representation of the drug.
    X_target (list): List containing the amino acid sequence of the protein target.

    Returns:
    float: Predicted binding affinity (log(Kd) or log(Ki)).
    """

    print("Predicting binding affinity: ", X_drug, X_target)
    
    model = models.model_pretrained(path_dir='DTI_model')

    X_pred = utils.data_process(X_drug, X_target, y,
                                drug_encoding='CNN', 
                                target_encoding='CNN', 
                                split_method='no_split')
   
    predictions = model.predict(X_pred)

    return predictions[0]

def get_amino_acid_sequence_from_pdb(pdb_id):    
    """
    Extracts amino acid sequences from a given PDB structure file in CIF format.

    Args:
        pdb_id (str): pdb id of the protein.

    Returns:
        dict: A dictionary where keys are chain IDs and values are amino acid sequences.
    """

    print("Getting Amino Acid sequence for ", pdb_id)

    cif_file_path = f"./database/PDBlib/{pdb_id.lower()}.cif"

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_file_path)
    
    sequences = {}
    for model in structure:
        for chain in model:
            seq = "".join(residue.resname for residue in chain if residue.id[0] == " ")
            sequences[chain.id] = seq 
            
    return sequences
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

def get_chemberta_embedding(smiles):
    """
    Generate a ChemBERTa vector embedding for a given molecule represented as a SMILES string.

    Args:
        smiles (str): A valid SMILES representation of a molecule.

    Returns:
        List[float] or None: A 768-dimensional vector as a list of floats if successful, 
                             otherwise None if the input is invalid.
    """
    
    print("Getting vector embedding")

    if not isinstance(smiles, str) or not smiles.strip():
        return None 

    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).tolist()[0]

def generate_key(smiles):
    """Generate a unique _key for the compound using SMILES hash."""
    hash_value = hashlib.sha256(smiles.encode()).hexdigest()[:8]
    return f"GEN:{hash_value}"



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

worker = TamGenCustom(
    data="./TamGen_Demo_Data",
    ckpt="checkpoints/crossdock_pdb_A10/checkpoint_best.pt",
    use_conditional=True
)


def prepare_pdb_data(pdb_id):
    """
    Checks if the PDB data for the given PDB ID is available.  
    If not, downloads and processes the data.

    ALWAYS RUN THIS FUNCTION BEFORE WORKING WITH PDB

    Args:
        pdb_id (str): PDB ID of the target structure.

    """

    DemoDataFolder="TamGen_Demo_Data"
    ligand_inchi=None
    thr=10

    out_split = pdb_id.lower()
    FF = glob(f"{DemoDataFolder}/*")
    for ff in FF:
        if f"gen_{out_split}" in ff:
            print(f"{pdb_id} is downloaded")
            return
    
    os.makedirs(DemoDataFolder, exist_ok=True)
    
    with open("tmp_pdb.csv", "w") as fw:
        if ligand_inchi is None:
            print("pdb_id", file=fw)
            print(f"{pdb_id}", file=fw)
        else:
            print("pdb_id,ligand_inchi", file=fw)
            print(f"{pdb_id},{ligand_inchi}", file=fw)

    script_path = os.path.abspath("TamGen/scripts/build_data/prepare_pdb_ids.py")
    os.system(f"python {script_path} tmp_pdb.csv gen_{out_split} -o {DemoDataFolder} -t {thr}")
    os.remove("tmp_pdb.csv")

@tool
def generate_compounds(pdb_id, num_samples=10, max_seed=30):
    """
    Generates and sorts compounds based on similarity to a reference molecule, 
    all generated compounds are added back to the database for futher inference.

    Parameters:
    - pdb_id (str): The PDB ID of the target protein.
    - num_samples (int): Number of compounds to generate. (DEFAULT=500)
    - max_seed (int): Maximum seed variations. (DEFAULT=30)

    Returns:
    - dict: {
        'generated': [list of rdkit Mol objects],
        'reference': rdkit Mol object,
        'reference_smile': SMILE string of the reference compound
        'generated_smiles': [list of SMILES strings, sorted by similarity to reference]
      }
    """

    print("Generating Compounds for PDB ", pdb_id)
    try:
        # Ensure the required PDB data is prepared
        # prepare_pdb_data(pdb_id)

        worker.reload_data(subset=f"gen_{pdb_id.lower()}")

        print(f"Generating {num_samples} compounds...")
        generated_mols, reference_mol = worker.sample(
            m_sample=num_samples, 
            maxseed=max_seed
        )

        if reference_mol:
            # Ensure reference_mol is an RDKit Mol object
            if isinstance(reference_mol, str):
                reference_mol = Chem.MolFromSmiles(reference_mol)

            fp_ref = MACCSkeys.GenMACCSKeys(reference_mol)

            gens = []
            for mol in generated_mols:
                if isinstance(mol, str):  # Convert string SMILES to Mol
                    mol = Chem.MolFromSmiles(mol)
                if mol:  # Ensure conversion was successful
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    similarity = DataStructs.FingerprintSimilarity(fp_ref, fp, metric=DataStructs.TanimotoSimilarity)
                    gens.append((mol, similarity))

            sorted_mols = [mol for mol, _ in sorted(gens, key=lambda e: e[1], reverse=True)]
        
        else:
            sorted_mols = generated_mols

        generated_smiles = [Chem.MolToSmiles(mol) for mol in sorted_mols if mol]

        reference_smile = Chem.MolToSmiles(reference_mol)
        
        print("Inserting to ArangoDB...")
        for smiles in generated_smiles:
            _key = generate_key(smiles) 
            drug_id = f"drug/{_key}"
            protein_id = f"protein/{pdb_id}"

            if drug_collection.has(_key):
                continue

            embedding = get_chemberta_embedding(smiles)
            doc = {
                "_key": _key,
                "_id": drug_id, 
                "accession": "NaN",
                "drug_name": "NaN",
                "cas": "NaN",
                "unii": "NaN",
                "synonym": "NaN",
                "key": "NaN",
                "chembl": "NaN",
                "smiles": smiles,
                "inchi": "NaN",
                "generated": True,
                "embedding": embedding
            }
            drug_collection.insert(doc)

            existing_links = list(db.aql.execute(f'''
                FOR link IN `drug-protein` 
                FILTER link._from == "{drug_id}" AND link._to == "{protein_id}" 
                RETURN link
            '''))

            if not existing_links:
                link_doc = {
                    "_from": drug_id,
                    "_to": protein_id,
                    "generated": True
                }
                link_collection.insert(link_doc)

        return {
            "generated": sorted_mols,
            "reference": reference_mol,
            "reference_smile": reference_smile,
            "generated_smiles": generated_smiles
        }

    except Exception as e:
        print(f"Error in compound generation: {str(e)}")
        return {"error": str(e)}

def generate_report(columns, rows):
    """
    Generate a report in CSV format with a timestamped filename. This function uses pandas to create a CSV.
    
    Parameters:
    columns (list): List of column names.
    rows (list of lists): Data rows corresponding to the columns.
    
    Returns:
    str: Path of the generated CSV report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.csv"
    
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(filename, index=False)
    
    return filename

def find_similar_drugs(smile, top_k=5):
    """
    Finds the top K most similar drugs based on given smile of a query molecule. Automatically gets vector embeddings.

    Args:
        smile (string): Smile of the query molecule.
        top_k (int, optional): Number of most similar drugs to retrieve. Default is 5.

    Returns:
        List[Dict{str, [float]}]: A list of (drug_name, similarity_score) sorted by similarity.
    """
    
    print("Finding similar drugs...")

    embedding = get_chemberta_embedding(smile)
    
    aql_query = f"""
    LET query_vector = @query_vector
    FOR doc IN drug
        LET score = COSINE_SIMILARITY(doc.embedding, query_vector)
        SORT score DESC
        LIMIT @top_k
        RETURN {{ drug: doc._key, similarity_score: score }}
    """
    
    cursor = db.aql.execute(aql_query, bind_vars={"query_vector": embedding, "top_k": top_k})
    
    return list(cursor)
# TOOOOOOOOOOOOOLSSSSSSSSSSS


text_to_aql_tool=Tool(
    name="TextToAql",
    func=text_to_aql,
    description=text_to_aql.__doc__
)
binding_affinity_tool=Tool(
    name="PredictionBindityAffinity",
    func=predict_binding_affinity,
    description=predict_binding_affinity.__doc__
)
amino_acid_tool=Tool(
    name="GetAminoSequenceFromPdb",
    func=get_amino_acid_sequence_from_pdb,
    description=get_amino_acid_sequence_from_pdb.__doc__
)
chemberta_embedding_tool=Tool(
    name="GetChembertaEmbeddings",
    func=get_chemberta_embedding,
    description=get_chemberta_embedding.__doc__
)
pdb_data_tool=Tool(
    name="PreparePDBData",
    func=prepare_pdb_data,
    description=prepare_pdb_data.__doc__
)
generate_compounds_tool=Tool(
    name="GenerateCompunds",
    func=generate_compounds,
    description=generate_compounds.__doc__
)
report_tool=Tool(
    name="DrugReport",
    func=generate_report,
    description=generate_report.__doc__
)
similar_drugs_tool=Tool(
    name="FindSimilarDrugs",
    func=find_similar_drugs,
    description=find_similar_drugs.__doc__
)



tools = [text_to_aql_tool,binding_affinity_tool,amino_acid_tool,chemberta_embedding_tool,pdb_data_tool,generate_compounds_tool,report_tool,similar_drugs_tool]
