import os
import time
import random
import logging
import json
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your Llama API token
LLAMA_API_KEY = "LA-9fb41d643e5d4a24a6f4337d4f91af6666fca18ab1c84ccc9a50c681d5fb0fc0"

def generate_random_prompt():
    """Generate a random prompt for testing."""
    prompts = [
        "Explain the concept of machine learning",
        "What is the difference between AI and ML?",
        "How do neural networks work?",
        "Explain transfer learning",
        "What is deep learning?",
        "Describe reinforcement learning",
        "What are transformers in AI?",
        "Explain the concept of backpropagation",
    ]
    return random.choice(prompts)

def send_traffic(num_requests=10, delay_range=(1, 3)):
    """
    Send traffic to Llama API
    
    Args:
        num_requests (int): Number of requests to send
        delay_range (tuple): Range of delay between requests in seconds
    """
    headers = {
        "Authorization": f"Bearer {LLAMA_API_KEY}",
        "Content-Type": "application/json"
    }

    for i in range(num_requests):
        try:
            prompt = generate_random_prompt()
            
            # Log the request
            logger.info(f"Request {i+1}/{num_requests}: {prompt}")
            
            # Prepare the payload for Llama API
            payload = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            
            # Send request to Llama API
            response = requests.post(
                "https://api.llama-api.com/chat/completions",  # Updated URL
                headers=headers,
                json=payload,
                timeout=30  # Increased timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                logger.info(f"Response received: {len(response_text)} chars")
            else:
                logger.error(f"Error: {response.status_code} - {response.text}")
            
            # Random delay between requests
            delay = random.uniform(*delay_range)
            logger.info(f"Waiting {delay:.2f} seconds before next request...")
            time.sleep(delay)
            
        except Exception as e:
            logger.error(f"Error in request {i+1}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    logger.info("Starting traffic generation...")
    send_traffic(num_requests=5, delay_range=(2, 5))
    logger.info("Traffic generation completed.")