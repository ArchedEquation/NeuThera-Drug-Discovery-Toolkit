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

import streamlit as st
import networkx as nx
from pyvis.network import Network

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
