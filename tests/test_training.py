import sys
from unittest.mock import MagicMock, patch
from app.training.trainer import TrainingJob

def test_training_job():
    print("\n--- Testing Training Job Wrapper ---")
    
    # Mock subprocess
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.poll.return_value = None # Running
        mock_popen.return_value = mock_process
        
        job = TrainingJob("roberto/test_dataset", "test_job_1")
        job.start()
        
        # Verify command construction
        args, _ = mock_popen.call_args
        cmd_list = args[0]
        print(f"Generated Command: {cmd_list}")
        
        assert "lerobot/scripts/train.py" in str(cmd_list[1])
        assert "dataset_repo_id=roberto/test_dataset" in cmd_list
        assert "policy=diffusion" in cmd_list
        
        status = job.get_status()
        print(f"Job Status: {status}")
        assert status == "RUNNING"
        
        print("Training Job Test Passed!")

if __name__ == "__main__":
    test_training_job()
