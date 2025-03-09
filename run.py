#!/usr/bin/env python
"""
Financial Assistant Runner

This script is the entry point for running the Financial Assistant application.
"""

import streamlit.web.cli as stcli
import sys
import os


def main():
    """Run the Financial Assistant application."""
    # Get the directory of this script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Path to the app.py file
    app_path = os.path.join(dir_path, "financial_assistant", "app.py")
    
    # Check if the file exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main() 