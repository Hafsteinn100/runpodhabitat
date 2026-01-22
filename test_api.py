import numpy as np
import base64
import httpx
import json

def test_api():
    print("1. Generating a fake satellite patch...")
    # Create a random array of shape (15, 35, 35) like a real image
    fake_patch = np.random.rand(15, 35, 35).astype(np.float32)
    
    # Encode it to Base64 (just like the judges will do)
    patch_bytes = fake_patch.tobytes()
    encoded_patch = base64.b64encode(patch_bytes).decode("utf-8")
    
    # Define the payload
    payload = {"patch": encoded_patch}
    
    print("2. Sending request to http://localhost:4321/predict ...")
    try:
        # Send the POST request
        response = httpx.post("http://localhost:4321/predict", json=payload, timeout=10.0)
        
        # Check the result
        if response.status_code == 200:
            result = response.json()
            print("\nSUCCESS! ✅")
            print(f"Your model predicted class index: {result['prediction']}")
            print("The API is working perfectly.")
        else:
            print("\nFAILURE ❌")
            print(f"Status code: {response.status_code}")
            print(f"Error details: {response.text}")
            
    except httpx.ConnectError:
        print("\nCONNECTION ERROR ❌")
        print("Could not connect to localhost:4321.")
        print("Is your 'python api.py' running in a separate terminal?")

if __name__ == "__main__":
    test_api()