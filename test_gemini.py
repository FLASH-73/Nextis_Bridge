import google.generativeai as genai
import os
import json

key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-flash-latest')

# Updated System Prompt to match the real one
prompt_context = """
You are the brain of a highly intelligent robotic arm. Your job is to decompose complex user commands into a tailored sequence of atomic actions.
INSTRUCTIONS:
1. Break down the user's request into a logical step-by-step plan.
2. You are NOT limited to a fixed set of tasks. You should generate descriptive "task" names that best fit the action.
3. The "task" field should be human-readable and concise.

OUTPUT FORMAT:
Output ONLY a valid JSON list of objects, each with "task" and "params".
"""

user_request = "Pick up the screw and place it into the top left hole and after that pick up the nut and screw it onto the screw."

final_prompt = f"{prompt_context}\n\nUser: {user_request}\nAssistant:"

print("Testing Open-Vocabulary Planner...")
try:
    response = model.generate_content(final_prompt)
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
