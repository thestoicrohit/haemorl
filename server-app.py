# Entry point for multi-mode deployment
# Imports the main app from root
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app
