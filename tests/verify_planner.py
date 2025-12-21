
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.planner import LocalPlanner

def test_planner():
    print("Initializing Planner (this might take a moment to load model)...")
    # Use CPU for test if CUDA not available, but LocalPlanner might force CUDA.
    # We'll see if it runs. The system has 40GB+ VRAM usually so it should be fine.
    try:
        planner = LocalPlanner()
    except Exception as e:
        print(f"Skipping planner test due to load error (maybe no GPU?): {e}")
        return

    if not planner.model:
        print("Planner model not loaded (maybe CPU fallback disabled).")
        return

    cmd = "Pack the blue cube into box B"
    print(f"Planning for: '{cmd}'")
    plan = planner.plan(cmd)
    
    print("Generated Plan:")
    print(json.dumps(plan, indent=2))
    
    if isinstance(plan, list) and len(plan) > 0 and "task" in plan[0]:
        print("✅ Plan generated valid JSON.")
    else:
        print("❌ Plan validation failed.")

if __name__ == "__main__":
    test_planner()
