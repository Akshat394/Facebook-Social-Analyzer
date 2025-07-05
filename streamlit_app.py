# Main Streamlit app for deployment
# This file serves as the entry point for Streamlit Cloud deployment

import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the advanced dashboard
from app import main

# Run the main function
main() 