import os
from plant_tips import PlantTipsGenerator
import pandas as pd
from IPython.display import display, Markdown

# Ensure API key is set
assert os.getenv('GEMINI_API_KEY'), 'Set the GEMINI_API_KEY environment variable.'


# Instantiate both generators
tips_rag = PlantTipsGenerator(use_rag=True)
# tips_no_rag = PlantTipsGenerator(use_rag=False)