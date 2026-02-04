import json
import sys
import os

# Add webapp to path to import app directly for testing client
sys.path.append(os.path.join(os.path.dirname(__file__), 'webapp'))

# Use requests to mock, but we need the app running.
# Instead, let's use app.test_client()

from webapp.app import app

def test_inference():
    print("Testing inference...")
    client = app.test_client()
    
    # Sample text from a known variant (simplified)
    # Class 4: Likely Pathogenic
    sample_text = """
    The BRCA1 gene is a tumor suppressor gene. Mutations in this gene are associated with an increased risk of breast and ovarian cancer. 
    The p.Val1688del variant results in the deletion of a valine residue. This variant interferes with the BRCT domain function.
    Functional assays show loss of transcriptional activation activity.
    """
    
    # Test POST to /api/predict
    response = client.post('/api/predict', 
                          data=json.dumps({'text': sample_text}),
                          content_type='application/json')
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.get_json()
        print("Response Details:")
        print(f"  Prediction Class: {data.get('prediction')}")
        print(f"  Class Name: {data.get('class_name')}")
        print(f"  Description: {data.get('description')}")
        print(f"  Probabilities: {json.dumps(data.get('probabilities'), indent=2)}")
        return True
    else:
        print(f"Error: {response.get_data(as_text=True)}")
        return False

if __name__ == "__main__":
    success = test_inference()
    if success:
        print("\nINFERENCE VERIFIED SUCCESSFUL")
    else:
        print("\nINFERENCE FAILED")
        sys.exit(1)
