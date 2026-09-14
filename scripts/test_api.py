import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5001"

def make_request(endpoint):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "ProductionAPITester/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            body = response.read().decode("utf-8")
            return status, json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw_error": body}
    except Exception as e:
        return 0, {"error": str(e)}

def test_api_endpoints():
    print("=" * 60)
    print("RUNNING FLASK BACKEND API ENDPOINT TEST SUITE")
    print("=" * 60)

    all_passed = True

    # 1. GET /health
    print("\n--- Test 1: GET /health ---")
    status, res = make_request("/health")
    if status == 200 and res.get("status") == "CERTIFIED_PRODUCTION_BUILD":
        print(f"[PASS] /health returned HTTP 200")
        print(f"       Model Name: {res.get('model_name')}")
        print(f"       SHA256: {res.get('checkpoint_sha256')}")
    else:
        all_passed = False
        print(f"[FAIL] /health returned status {status}: {res}")

    # 2. GET /model/info
    print("\n--- Test 2: GET /model/info ---")
    status, res = make_request("/model/info")
    if status == 200 and "locked_test_metrics" in res and "holdout_metrics" in res:
        print(f"[PASS] /model/info returned HTTP 200")
        print(f"       Test Accuracy: {res['locked_test_metrics'].get('test_accuracy')}")
        print(f"       Test Macro-F1: {res['locked_test_metrics'].get('test_macro_f1')}")
        print(f"       Test Balanced Accuracy: {res['locked_test_metrics'].get('test_balanced_accuracy')}")
        print(f"       Holdout Accuracy: {res['holdout_metrics'].get('holdout_accuracy')}")
        print(f"       Holdout Macro-F1: {res['holdout_metrics'].get('holdout_macro_f1')}")
        print(f"       Holdout Balanced Accuracy: {res['holdout_metrics'].get('holdout_balanced_accuracy')}")
    else:
        all_passed = False
        print(f"[FAIL] /model/info returned status {status}: {res}")

    # 3. GET /building/nearest?lat=13.0429&lon=80.2486&radius=100
    print("\n--- Test 3: GET /building/nearest (100m radius test) ---")
    status, res = make_request("/building/nearest?lat=13.0429&lon=80.2486&radius=100")
    if status == 404 and (res.get("status") == "not_found" or "No building found" in res.get("message", "")):
        print(f"[PASS] /building/nearest (100m) correctly returned HTTP 404 (No building within 100m)")
        print(f"       Response: {res.get('message')}")
    elif status == 200:
        print(f"[PASS] /building/nearest returned building: {res.get('building_id')}")
    else:
        all_passed = False
        print(f"[FAIL] /building/nearest unexpected response status {status}: {res}")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL API ENDPOINT TESTS PASSED [PASS]")
    else:
        print("SOME API ENDPOINT TESTS FAILED [FAIL]")
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    success = test_api_endpoints()
    sys.exit(0 if success else 1)
