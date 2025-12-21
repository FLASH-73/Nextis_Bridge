import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import os
import google.generativeai as genai

class GeminiPlanner:
    def __init__(self, api_key):
        print("Loading Gemini Planner...")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-flash-latest') # Fallback to standard 1.5 Flash

    def plan(self, input_data):
        system_prompt = """
        You are the brain of a highly intelligent robotic arm. Your job is to decompose complex user commands into a tailored sequence of atomic actions.
        
        INSTRUCTIONS:
        1. Break down the user's request into a logical step-by-step plan.
        2. You are NOT limited to a fixed set of tasks. You should generate descriptive "task" names that best fit the action (e.g., "Pick Apple", "Screw Nut", "Place on Shelf", "Open Drawer").
        3. The "task" field should be human-readable and concise.
        4. Use the "params" field for any specific details (coordinates, object IDs, target locations).
        
        OUTPUT FORMAT:
        Output ONLY a valid JSON list of objects, each with "task" and "params".

        EXAMPLES:

        User: "Pick up the screw and place it into the top left hole, then pick up the nut and screw it onto the screw."
        Assistant:
        [
            {"task": "Pick Screw", "params": {"object": "screw"}},
            {"task": "Place in Hole", "params": {"location": "top left hole"}},
            {"task": "Pick Nut", "params": {"object": "nut"}},
            {"task": "Screw Nut", "params": {"target": "screw", "action": "screw onto"}}
        ]

        User: "Pick object xy up and put it on a shelf"
        Assistant:
        [
            {"task": "Pick Object XY", "params": {"object": "xy"}},
            {"task": "Place on Shelf", "params": {"target": "shelf"}}
        ]

        User: "Grab the red ball and put it in box B"
        Assistant:
        [
            {"task": "Pick Red Ball", "params": {"object": "red ball"}},
            {"task": "Place in Box B", "params": {"box_id": "B"}}
        ]

        User: "Actually pick up the phone instead of the ball"
        Assistant:
        [
            {"task": "Pick Phone", "params": {"object": "phone"}},
            {"task": "Place in Box B", "params": {"box_id": "B"}}
        ]
        """
        
        # Construct Chat History for Gemini
        history = []
        user_message = ""
        
        if isinstance(input_data, list):
            print(f"DEBUG: Planning with history length: {len(input_data)}")
            
            # Extract last known plan
            last_plan = None
            for msg in reversed(input_data):
                if msg.get("role") == "assistant":
                    try:
                        content = msg.get("content", "")
                        # Robust JSON extraction handling markdown
                        clean_content = re.sub(r'```json\n|```', '', content)
                        match = re.search(r'\[.*\]', clean_content, re.DOTALL)
                        if match:
                            last_plan = match.group(0)
                            print("DEBUG: Found previous plan in history.")
                            break
                    except Exception as e:
                        print(f"DEBUG: Error extracting plan: {e}")
            
            # Construct simple history string
            context_str = f"SYSTEM INSTRUCTION:\n{system_prompt}\n\nCONVERSATION HISTORY:\n"
            
            for msg in input_data[:-1]: # All but last
                role = "User" if msg.get("role") == "user" else "Assistant"
                context_str += f"{role}: {msg.get('content')}\n"
            
            last_msg = input_data[-1]
            last_content = last_msg.get('content')
            
            if last_plan:
                print("DEBUG: Injecting CURRENT PLAN into prompt.")
                injection = f"""
CURRENT PLAN:
{last_plan}

USER REQUEST:
{last_content}

INSTRUCTION:
Refine the CURRENT PLAN based on the USER REQUEST.
- If the user says "swap X for Y", REPLACE X with Y but KEEP other tasks (like Z) unchanged.
- You MUST output the FULL updated plan, including unchanged items.
"""
                user_message = f"{context_str}\nUser: {injection}\nAssistant:"
            else:
                print("DEBUG: No previous plan found. independent generation.")
                user_message = f"{context_str}\nUser: {last_content}\nAssistant:"

        else:
            user_message = f"SYSTEM INSTRUCTION:\n{system_prompt}\n\nUser: {str(input_data)}\nAssistant:"

        try:
            # print(f"DEBUG: Sending to Gemini:\n{user_message}") # Uncomment if really stuck
            response = self.model.generate_content(user_message)
            text = response.text
             # Clean up response to find JSON
            clean_text = re.sub(r'```json\n|```', '', text)
            match = re.search(r'\[.*\]', clean_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                 print(f"No JSON found in response: {text}")
                 return []
        except Exception as e:
            print(f"Gemini Error: {e}")
            return [{"task": "error", "params": {"msg": f"Gemini API Error: {str(e)}"}}]


class LocalPlanner:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
        print(f"Loading Local Planner Model: {model_id}...")
        self.device = device
        
        # Proactive CUDA Check
        use_cuda = False
        if device != "cpu":
            try:
                if torch.cuda.is_available():
                    # Try a small operation
                    t = torch.tensor([1.0]).cuda()
                    use_cuda = True
            except Exception as e:
                print(f"⚠️ CUDA detected but broken: {e}")
                use_cuda = False

        # Initialize Tokenizer (CPU safe)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            self.model = None
            return

        try:
            if use_cuda:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                print("✅ Planner Model Loaded Successfully (GPU)!")
            else:
                self.model = None
                print("CUDA not available. Local Planner Disabled.")

        except Exception as e:
            print(f"❌ Failed to load model on GPU: {e}")
            self.model = None

    def plan(self, input_data):
        if not self.model:
            return [{"task": "error", "params": {"msg": "Local Model not loaded"}}]

        system_prompt = """
        You are the brain of a robotic arm. Your job is to decompose user commands into a JSON list of sequential atomic tasks.
        
        AVAILABLE TASKS:
        - move_to_bin(bin_id: str)  -> Move to a specific bin (A, B, C, etc.).
        - pick_object(object_name: str) -> Pick up an object.
        - place_in_box(box_id: str) -> Place the held object into a box.
        - home() -> Return to home position.

        RULES:
        1. Use 'move_to_bin' ONLY if the user explicitly specifies a source location.
        2. The robot can only hold ONE object at a time. Sequence: pick -> place.
        3. If the user asks to move multiple items, handle them ONE BY ONE.
        4. Always end with 'home()'.
        5. Output ONLY a valid JSON list.

        CONVERSATIONAL UPDATES:
        - If the user modifies the request (e.g., "replace X with Y"), YOU MUST PRESERVE all other parts of the previous plan that weren't changed.
        - Do not drop items from the plan unless explicitly told to remove them.
        - Merge the new intent with the existing context to form a complete, updated plan.
        
        ID HANDLING:
        - If the user does NOT specify a bin/box ID (e.g., just "the box"), use "default" as the ID. Do NOT hallucinate "A", "B", or "C".

        EXAMPLES:

        User: "Pick up the red ball and put it in box B"
        Assistant:
        [
            {"task": "pick_object", "params": {"object_name": "red ball"}},
            {"task": "move_to_bin", "params": {"bin_id": "B"}},
            {"task": "place_in_box", "params": {"box_id": "B"}},
            {"task": "home", "params": {}}
        ]

        User: "Actually, pick up the blue cube instead of the ball"
        Assistant:
        [
            {"task": "pick_object", "params": {"object_name": "blue cube"}},
            {"task": "move_to_bin", "params": {"bin_id": "B"}},
            {"task": "place_in_box", "params": {"box_id": "B"}},
            {"task": "home", "params": {}}
        ]

        User: "Grab the apple and the book and put them in the box"
        Assistant:
        [
            {"task": "pick_object", "params": {"object_name": "apple"}},
            {"task": "place_in_box", "params": {"box_id": "default"}},
            {"task": "pick_object", "params": {"object_name": "book"}},
            {"task": "place_in_box", "params": {"box_id": "default"}},
            {"task": "home", "params": {}}
        ]

        User: "Actually pick up the phone instead of the apple"
        Assistant:
        [
            {"task": "pick_object", "params": {"object_name": "phone"}},
            {"task": "place_in_box", "params": {"box_id": "default"}},
            {"task": "pick_object", "params": {"object_name": "book"}},
            {"task": "place_in_box", "params": {"box_id": "default"}},
            {"task": "home", "params": {}}
        ]
        """
        
        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]
        
        if isinstance(input_data, list):
            # It's a history list
            
            # --- CONTEXT INJECTION STRATEGY ---
            # Extract the last valid plan from the assistant's history to force the model to see it.
            last_plan = None
            for msg in reversed(input_data):
                if msg.get("role") == "assistant":
                    # Try to find JSON in this message
                    try:
                        content = msg.get("content", "")
                        match = re.search(r'\[.*\]', content, re.DOTALL)
                        if match:
                            last_plan = match.group(0)
                            break
                    except:
                        continue
            
            for i, msg in enumerate(input_data):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # If this is the LAST message (User's new request) and we found a previous plan, INJECT IT.
                if i == len(input_data) - 1 and role == "user" and last_plan:
                    content = f"CURRENT PLAN:\n{last_plan}\n\nUSER REQUEST: {content}\n\nINSTRUCTION: Update the CURRENT PLAN based on the USER REQUEST. Keep all items from the CURRENT PLAN that are not explicitly removed or replaced."
                
                messages.append({"role": role, "content": content})
        else:
            # It's a single string
            messages.append({"role": "user", "content": str(input_data)})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=False  # Deterministic for planning
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up response to find JSON
        try:
            # Find the first '[' and last ']'
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                 print(f"No JSON found in response: {response}")
                 return []
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            return []
