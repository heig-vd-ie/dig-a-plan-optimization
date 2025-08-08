import json
import os
import sys
import requests
from pathlib import Path

SERVER_HOST = "localhost"

if len(sys.argv) >= 2:
    SERVER_PORT = int(sys.argv[1])
elif "SERVER_PORT" in os.environ:
    SERVER_PORT = int(os.environ["SERVER_PORT"])
else:
    SERVER_PORT = 8080

SERVER_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

print(f"Testing API server at: {SERVER_BASE_URL}")


def test_stochastic_planning_with_default_data():
    """Test stochastic planning endpoint using data from default.json"""
    print("\n--- Testing Stochastic Planning with Default Data ---")

    # Load data from default.json file
    default_json_path = Path(__file__).parent / "../../../data/default.json"

    try:
        with open(default_json_path, "r") as f:
            request_data = json.load(f)

        print(f"Loaded data from: {default_json_path}")
        print(f"Request data keys: {list(request_data.keys())}")

        # Send POST request to stochastic planning endpoint
        response = requests.post(
            f"{SERVER_BASE_URL}/stochastic_planning",
            headers={"Content-Type": "application/json"},
            json=request_data,
        )

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✓ Request successful!")
            print(f"Response keys: {list(result.keys())}")

            # Validate response structure
            assert "objectives" in result, "Response missing 'objectives' key"
            assert "simulations" in result, "Response missing 'simulations' key"
            assert len(result["objectives"]) > 0, "No objectives in response"
            assert len(result["simulations"]) > 0, "No simulations in response"
            assert len(result["objectives"]) == len(
                result["simulations"]
            ), "Objectives and simulations length mismatch"

            print(
                f"✓ Found {len(result['objectives'])} objectives and {len(result['simulations'])} simulations"
            )
            print("✓ All validations passed!")

            return result
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except FileNotFoundError:
        print(f"✗ Could not find default.json at {default_json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing JSON: {e}")
        return None
    except requests.RequestException as e:
        print(f"✗ Request error: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None


def main():
    """Run the test"""
    result = test_stochastic_planning_with_default_data()

    if result:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
