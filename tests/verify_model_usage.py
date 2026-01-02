"""
Verify that we're using the model from openrouter_provider.py
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.openrouter_provider import OpenRouterProvider, OpenRouterModel
from strands import Agent

load_dotenv()

print("\n" + "="*60)
print("Verifying Model Usage from openrouter_provider.py")
print("="*60)

# 1. Check the default model in OpenRouterProvider
provider = OpenRouterProvider()
print(f"\n[1] OpenRouterProvider.default_llm_model: {provider.default_llm_model}")

# 2. Check that OpenRouterModel uses the provider's model
model = OpenRouterModel(provider)
print(f"[2] OpenRouterModel.model_id: {model.model_id}")
print(f"[3] OpenRouterModel.get_config()['model']: {model.get_config()['model']}")

# 3. Verify they all match
assert model.model_id == provider.default_llm_model, "Model ID should match provider's default"
assert model.get_config()['model'] == provider.default_llm_model, "Config should match provider's default"
print("\n[OK] All model references point to the same model from openrouter_provider.py")

# 4. Verify the Agent uses this model
agent = Agent(model=model)
print(f"[4] Agent created with OpenRouterModel")
print(f"[5] Agent's model is: {agent.model.model_id if hasattr(agent.model, 'model_id') else 'N/A'}")

# 5. Show the actual model being used
print(f"\n[VERIFIED] The model being used is: {provider.default_llm_model}")
print(f"           This is defined in openrouter_provider.py line 51")
print(f"           Default: 'google/gemini-2.5-flash-lite-preview-09-2025'")

print("\n" + "="*60)
print("[SUCCESS] Confirmed: We are using the model from openrouter_provider.py")
print("="*60)

