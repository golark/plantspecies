#!/usr/bin/env python3
"""
Test script for the Plant RAG System
"""

import os
import sys
from plant_rag import PlantRAGSystem
from plant_tips import PlantTipsGenerator

def test_rag_system():
    """Test the RAG system functionality."""
    print("🧪 Testing Plant RAG System...")
    
    try:
        # Initialize RAG system
        print("1. Initializing RAG system...")
        rag_system = PlantRAGSystem()
        
        # Check if data is loaded
        stats = rag_system.get_collection_stats()
        print(f"   Collection stats: {stats}")
        
        if stats["total_documents"] == 0:
            print("2. Loading plant care data...")
            rag_system.load_plant_data()
            stats = rag_system.get_collection_stats()
            print(f"   Loaded {stats['total_documents']} documents")
        
        # Test search functionality
        print("3. Testing search functionality...")
        test_plants = ["Monstera deliciosa", "Ficus lyrata", "Aloe vera"]
        
        for plant in test_plants:
            print(f"\n   Testing: {plant}")
            
            # Test exact search
            plant_info = rag_system.get_plant_care_info(plant)
            if plant_info:
                print(f"   ✅ Found exact match: {plant_info['metadata']['species']}")
            else:
                print(f"   ⚠️  No exact match found, testing fuzzy search...")
                similar = rag_system.search_similar_plants(plant, n_results=1)
                if similar:
                    print(f"   ✅ Found similar: {similar[0]['metadata']['species']}")
                else:
                    print(f"   ❌ No similar plants found")
        
        # Test RAG response generation
        print("\n4. Testing RAG response generation...")
        test_plant = "Monstera deliciosa"
        response = rag_system.generate_rag_response(test_plant)
        print(f"   ✅ Generated response for {test_plant}")
        print(f"   Response length: {len(response)} characters")
        
        # Test with specific question
        question_response = rag_system.generate_rag_response(
            test_plant, 
            "How often should I water this plant in winter?"
        )
        print(f"   ✅ Generated response for specific question")
        print(f"   Question response length: {len(question_response)} characters")
        
        print("\n🎉 All RAG system tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def test_tips_generator():
    """Test the enhanced PlantTipsGenerator."""
    print("\n🧪 Testing Enhanced PlantTipsGenerator...")
    
    try:
        # Initialize tips generator with RAG
        print("1. Initializing PlantTipsGenerator with RAG...")
        tips_generator = PlantTipsGenerator(use_rag=True)
        
        # Test basic functionality
        print("2. Testing basic tips generation...")
        test_plant = "Monstera deliciosa"
        tips = tips_generator.generate_gemini_tips(test_plant)
        print(f"   ✅ Generated tips for {test_plant}")
        print(f"   Tips length: {len(tips)} characters")
        
        # Test with specific question
        print("3. Testing tips generation with specific question...")
        question_tips = tips_generator.generate_gemini_tips(
            test_plant, 
            user_question="How often should I water this plant in winter?"
        )
        print(f"   ✅ Generated tips for specific question")
        print(f"   Question tips length: {len(question_tips)} characters")
        
        print("\n🎉 All PlantTipsGenerator tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ PlantTipsGenerator test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Plant RAG System Tests")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("Please set your Gemini API key and try again.")
        return False
    
    # Run tests
    rag_success = test_rag_system()
    tips_success = test_tips_generator()
    
    print("\n" + "=" * 50)
    if rag_success and tips_success:
        print("🎉 All tests passed! RAG system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 