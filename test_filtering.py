
# Simulation of TeleopService filtering logic

def map_logic(active_arms, leader_obs):
    action = {}
    
    def is_active(side, group):
        if active_arms is None: return True
        id_str = f"{side}_{group}"
        if id_str not in active_arms:
             return False
        return True

    for k, v in leader_obs.items():
        if not k.endswith(".pos"):
            continue
            
        # Parse Key: e.g. "left_link1.pos"
        side = "default"
        if k.startswith("left_"): side = "left"
        elif k.startswith("right_"): side = "right"
        
        # Check if this KEY should be processed
        leader_active = is_active(side, "leader")
        follower_active = is_active(side, "follower")
        
        print(f"Key: {k} -> Side: {side} -> LeaderActive: {leader_active} -> FollowerActive: {follower_active}") # DEBUG
        
        if leader_active and follower_active:
            action[k] = v
            
    return action

# Test Cases
print("--- TEST 1: All Active (None) ---")
obs = {"left_link1.pos": 100, "right_link1.pos": 200}
res = map_logic(None, obs)
print(f"Result: {res}")

print("\n--- TEST 2: Left Only (['left_leader', 'left_follower']) ---")
active = ["left_leader", "left_follower"]
res = map_logic(active, obs)
print(f"Result: {res}")
assert "left_link1.pos" in res
assert "right_link1.pos" not in res

print("\n--- TEST 3: Mismatch (['left_leader', 'right_follower']) ---")
active = ["left_leader", "right_follower"]
res = map_logic(active, obs)
print(f"Result: {res}")
# Expect LEFT to fail (follower inactive) and RIGHT to fail (leader inactive) -> Empty
assert len(res) == 0

print("\n--- TEST 4: Single Arm Format (['default_leader', 'default_follower']) with 'link1.pos' ---")
obs_single = {"link1.pos": 500}
active = ["default_leader", "default_follower"]
res = map_logic(active, obs_single)
print(f"Result: {res}")
# Expect match (side=default)
assert "link1.pos" in res

print("\n--- TEST 5: Single Arm Format with Left Keys? ---")
# If I have single arm but keys are 'left_link1.pos'?
active = ["default_leader", "default_follower"]
res = map_logic(active, obs) # left_link1.pos
print(f"Result: {res}")
# side="left". is_active("left", "leader") -> checks "left_leader". Not in active. -> False.
# Dropped. Correct behavior for single arm vs bi-arm keys.

