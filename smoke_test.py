"""Quick check against a running service.

Usage:
    python smoke_test.py [base_url]
"""
import json
import sys
from urllib.request import Request, urlopen

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696"

SAMPLE = {
    "OverallQual": 7,
    "GrLivArea": 1710,
    "GarageCars": 2,
    "TotalBsmtSF": 856,
    "YearBuilt": 2003,
    "Neighborhood": "CollgCr",
    "MSZoning": "RL",
    "LotArea": 8450,
    "FullBath": 2,
    "BedroomAbvGr": 3,
}


def get(path):
    with urlopen(f"{BASE_URL}{path}", timeout=10) as response:
        return json.load(response)


def post(path, payload):
    request = Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def main():
    health = get("/health")
    assert health["status"] == "ok", health

    result = post("/predict", SAMPLE)
    assert 0 < result["predicted_price"] < 1_000_000, result
    assert result["currency"] == "USD", result

    print("health :", health)
    print("predict:", result)


if __name__ == "__main__":
    main()
