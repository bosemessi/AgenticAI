#!/usr/bin/env python
import os
os.environ["OPENAI_API_KEY"] = "na"

import sys
import warnings

from datetime import datetime

import yaml

from streamlit_stats_app.crew import StreamlitStatsApp

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")



# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    print("Here is the app")
    with open("src/streamlit_stats_app/config/streamlit_stats_app.yaml", "r", encoding="utf-8") as file:
        examples = yaml.safe_load(file)

    inputs={
        "streamlit_stats_app": examples["example1_scatterplotter"]
    }
    
    try:
        streamlitStatsApp = StreamlitStatsApp().crew().kickoff(inputs=inputs)
        print("\nFinal code for the app")
        print("\n-------------------------")
        print(streamlitStatsApp)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        StreamlitStatsApp().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        StreamlitStatsApp().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        StreamlitStatsApp().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
