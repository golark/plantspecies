import os
import logging
from typing import Optional
import google.generativeai as genai
from plant_rag import PlantRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantTipsGenerator:
    """
    A class to generate plant care tips using RAG-enhanced Gemini API.
    """
    def __init__(self, use_rag: bool = True):
        api_key = os.getenv("GEMINI_API_KEY")
        print(f'api_key = {api_key}')
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Initialize RAG system if requested
        self.use_rag = use_rag
        self.rag_system = None
        if use_rag:
            try:
                self.rag_system = PlantRAGSystem()
                # Load plant data if collection is empty
                if self.rag_system.get_collection_stats()["total_documents"] == 0:
                    self.rag_system.load_plant_data()
                logger.info("RAG system initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG system: {e}. Falling back to direct Gemini API.")
                self.use_rag = False

    def generate_gemini_tips(self, plant_species: str, prompt: Optional[str] = None, user_question: Optional[str] = None) -> str:
        """
        Generate plant care tips using RAG-enhanced Gemini API or direct Gemini API.
        Args:
            plant_species (str): The name of the plant species
            prompt (str, optional): Custom prompt. If None, use a default prompt.
            user_question (str, optional): Specific user question for RAG system
        Returns:
            str: Generated care tips
        """
        # Try RAG system first if available
        if self.use_rag and self.rag_system:
            try:
                logger.info(f"Using RAG system for {plant_species}")
                return self.rag_system.generate_rag_response(plant_species, user_question if user_question else None)
            except Exception as e:
                logger.warning(f"RAG system failed for {plant_species}: {e}. Falling back to direct Gemini API.")
        
        # Fallback to direct Gemini API
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
        
        # Add user question to prompt if provided
        if user_question:
            prompt += f"\n\nUser's specific question: {user_question}\nPlease address this question in your response."
        
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