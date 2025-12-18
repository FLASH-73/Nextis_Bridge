import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

class LocalPlanner:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
        print(f"Loading Planner Model: {model_id}...")
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
                raise RuntimeError("CUDA not available. CPU fallback disabled by user request.")

        except Exception as e:
            print(f"❌ Failed to load model on GPU: {e}")
            with open("planner_error.log", "w") as f:
                f.write(str(e))
            self.model = None
            # Do NOT fallback to CPU

    def plan(self, user_instruction: str):
        if not self.model:
            return [{"task": "error", "params": {"msg": "Model not loaded"}}]

        system_prompt = """
        You are the brain of a robotic arm. Your job is to decompose high-level user commands into a JSON list of sequential atomic tasks.
        
        AVAILABLE TASKS:
        - move_to_bin(bin_id: str)  -> Move to a specific bin (A, B, C, etc.).
        - pick_object(object_name: str) -> Pick up an object.
        - place_in_box(box_id: str) -> Place the held object into a box.
        - home() -> Return to home position.

        RULES:
        1. Use 'move_to_bin' ONLY if the user explicitly specifies a source location (e.g. "from bin A"). If not specified, assume the object is accessible directly via 'pick_object'.
        2. The robot can only hold ONE object at a time. You MUST 'place_in_box' before 'pick_object' for the next item.
        3. If the user asks to move multiple items, handle them ONE BY ONE.
        4. Always end with 'home()'.
        5. Output ONLY a valid JSON list. No markdown, no explanations.

        EXAMPLES:

        User: "Pick up the red ball and put it in box B"
        Assistant:
        [
            {"task": "pick_object", "params": {"object_name": "red ball"}},
            {"task": "move_to_bin", "params": {"bin_id": "B"}},
            {"task": "place_in_box", "params": {"box_id": "B"}},
            {"task": "home", "params": {}}
        ]

        User: "Grab the apple from bin A and the banana from bin C and put them in the basket"
        Assistant:
        [
            {"task": "move_to_bin", "params": {"bin_id": "A"}},
            {"task": "pick_object", "params": {"object_name": "apple"}},
            {"task": "move_to_bin", "params": {"bin_id": "Basket"}},
            {"task": "place_in_box", "params": {"box_id": "Basket"}},
            {"task": "move_to_bin", "params": {"bin_id": "C"}},
            {"task": "pick_object", "params": {"object_name": "banana"}},
            {"task": "move_to_bin", "params": {"bin_id": "Basket"}},
            {"task": "place_in_box", "params": {"box_id": "Basket"}},
            {"task": "home", "params": {}}
        ]
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_instruction}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature=0.1
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract JSON from response
        try:
            # Find the first '[' and last ']'
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                print(f"Failed to parse JSON from: {response}")
                return []
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            return []

if __name__ == "__main__":
    # Test
    planner = LocalPlanner()
    plan = planner.plan("Pack the red apple into box B")
    print(plan)
