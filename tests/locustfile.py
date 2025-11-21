"""
Locust Load Testing File

This file defines load testing scenarios for the Audio Classification API.

Run with:
    locust -f tests/locustfile.py --host=http://localhost:5000

Then open browser at http://localhost:8089
"""

import os
from locust import HttpUser, task, between
import random


class AudioClassificationUser(HttpUser):
    """
    Simulated user for load testing the Audio Classification API
    """

    # Wait time between tasks (simulating real user behavior)
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """
        Called when a simulated user starts
        You can use this to perform login or setup
        """
        # Check if API is healthy
        response = self.client.get("/health")
        if response.status_code == 200:
            print("API is healthy, starting load test")
        else:
            print(f"Warning: API health check failed with status {response.status_code}")

    @task(3)  # Weight of 3 - more frequent
    def health_check(self):
        """
        Test health endpoint
        Most frequent task
        """
        self.client.get("/health")

    @task(2)  # Weight of 2
    def get_model_info(self):
        """
        Test model info endpoint
        """
        with self.client.get("/model_info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                response.failure("Model not loaded")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)  # Weight of 1 - less frequent
    def predict_single_file(self):
        """
        Test single file prediction endpoint
        Requires a test audio file
        """
        # Path to test audio file (you'll need to provide this)
        test_file_path = "data/test/sample.wav"

        if os.path.exists(test_file_path):
            with open(test_file_path, 'rb') as f:
                files = {'file': ('sample.wav', f, 'audio/wav')}

                with self.client.post("/predict", files=files, catch_response=True) as response:
                    if response.status_code == 200:
                        result = response.json()
                        if 'predicted_class' in result:
                            response.success()
                        else:
                            response.failure("Missing predicted_class in response")
                    elif response.status_code == 503:
                        response.failure("Model not loaded")
                    else:
                        response.failure(f"Prediction failed: {response.status_code}")
        else:
            # If no test file exists, just skip this task
            pass

    @task(1)
    def get_metrics(self):
        """
        Test metrics endpoint
        """
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code in [200, 404]:
                # 404 is acceptable if no metrics exist yet
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")


class StressTestUser(HttpUser):
    """
    More aggressive user for stress testing
    """

    wait_time = between(0.1, 0.5)  # Very short wait time

    @task
    def rapid_health_checks(self):
        """Rapid fire health checks"""
        self.client.get("/health")


# Custom load shape (optional)
# from locust import LoadTestShape

# class CustomLoadShape(LoadTestShape):
#     """
#     Custom load pattern: ramp up, plateau, ramp down
#     """
#     def tick(self):
#         run_time = self.get_run_time()
#
#         if run_time < 60:
#             # Ramp up to 100 users over 60 seconds
#             user_count = int(run_time * 100 / 60)
#             spawn_rate = 2
#             return (user_count, spawn_rate)
#
#         elif run_time < 180:
#             # Stay at 100 users for 120 seconds
#             return (100, 2)
#
#         elif run_time < 240:
#             # Ramp down over 60 seconds
#             user_count = 100 - int((run_time - 180) * 100 / 60)
#             spawn_rate = 2
#             return (user_count, spawn_rate)
#
#         else:
#             # Stop test after 240 seconds
#             return None


"""
USAGE INSTRUCTIONS:

1. Basic load test:
   locust -f tests/locustfile.py --host=http://localhost:5000

2. Headless mode (no web UI):
   locust -f tests/locustfile.py --host=http://localhost:5000 \
          --users 100 --spawn-rate 10 --run-time 5m --headless

3. Test with specific user class:
   locust -f tests/locustfile.py --host=http://localhost:5000 \
          StressTestUser

4. Test with multiple Docker containers:
   # Terminal 1: 1 container
   docker-compose up

   # Terminal 2: Run locust
   locust -f tests/locustfile.py --host=http://localhost:5000 \
          --users 50 --spawn-rate 5 --run-time 2m --headless

   # Then scale up and test again
   docker-compose up --scale audio-classifier=3

   # Run locust again with more users
   locust -f tests/locustfile.py --host=http://localhost:5000 \
          --users 200 --spawn-rate 10 --run-time 2m --headless

METRICS TO RECORD:
- Response time (median, 95th percentile, 99th percentile)
- Requests per second (RPS)
- Failure rate
- Number of users
- Number of containers

Create a table comparing these metrics across:
- 1 container vs 2 containers vs 3 containers
- Different user loads (50, 100, 200 users)
"""
