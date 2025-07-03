import os
import logging
from typing import Optional
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantTipsGenerator:
    """
    A class to generate plant care tips using the Gemini API via google-generativeai library.
    """
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate_gemini_tips(self, plant_species: str, prompt: Optional[str] = None) -> str:
        """
        Generate plant care tips using the Gemini API.
        Args:
            plant_species (str): The name of the plant species
            prompt (str, optional): Custom prompt. If None, use a default prompt.
        Returns:
            str: Gemini-generated care tips
        """
        if prompt is None:
            prompt = (
                f"You are an expert horticulturist and gardening assistant. "
                f"Provide detailed, practical care tips for a plant of the species: {plant_species}. "
                "Include information about watering, sunlight, soil, and special care instructions. "
                "Format your response in a clear, structured manner with the following sections: "
                "- Watering: How often and how much to water\n"
                "- Sunlight: Light requirements and placement\n"
                "- Soil: Soil type, pH, and potting mix recommendations\n"
                "- Special Care: Any specific care needs, pruning, fertilizing, etc.\n"
                "Make your advice practical and actionable for home gardeners."
            )
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise

if __name__ == "__main__":
    # Example usage for direct execution
    try:
        tips_generator = PlantTipsGenerator()
        plant_species = "Monstera deliciosa"
        print(f"Requesting Gemini care tips for: {plant_species}")
        tips = tips_generator.generate_gemini_tips(plant_species)
        print("\nGemini Plant Care Tips:\n" + tips)
    except Exception as e:
        print(f"Error: {e}\nMake sure the GEMINI_API_KEY environment variable is set and valid. Also ensure google-generativeai is installed.") 