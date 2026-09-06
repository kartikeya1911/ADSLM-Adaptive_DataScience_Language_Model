"""
test_deployment.py
==================
Comprehensive deployment verification suite for ADSLM FastAPI & Streamlit integration.
"""

import os
import io
import sys
import toml
from pathlib import Path
from fastapi.testclient import TestClient

from main import app
from app.core.config import REPORTS_DIR

def run_tests():
    print("=" * 70)
    print("🚀 RUNNING ADSLM DEPLOYMENT VERIFICATION SUITE")
    print("=" * 70)

    client = TestClient(app)

    # 1. Health Check Test
    print("\n[1/7] Testing GET /health ...")
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.status_code} - {res.text}"
    data = res.json()
    assert data.get("status") == "healthy", f"Unexpected health status: {data}"
    print("  ✅ /health returned 200 OK and status='healthy'")

    # 2. Root Test
    print("\n[2/7] Testing GET / (Welcome / Root) ...")
    res = client.get("/")
    assert res.status_code == 200
    print("  ✅ / returned 200 OK with endpoint index")

    # 3. Streamlit Secrets Test
    print("\n[3/7] Testing .streamlit/secrets.toml ...")
    secrets_path = Path(".streamlit") / "secrets.toml"
    assert secrets_path.exists(), ".streamlit/secrets.toml does not exist!"
    secrets = toml.load(secrets_path)
    assert "API_URL" in secrets, "API_URL missing in secrets.toml"
    print(f"  ✅ .streamlit/secrets.toml configured with API_URL = '{secrets['API_URL']}'")

    # 4. Orchestrate Test (End-to-End ML Pipeline)
    print("\n[4/7] Testing POST /orchestrate with predictive_maintenance.csv ...")
    csv_path = Path("datasets") / "predictive_maintenance.csv"
    with open(csv_path, "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/orchestrate",
        files={"file": ("predictive_maintenance.csv", file_bytes, "text/csv")},
        data={"target_column": "Fault", "expertise_level": "intermediate"}
    )
    assert res.status_code == 200, f"/orchestrate failed: {res.status_code} - {res.text}"
    resp_json = res.json()
    
    assert "report_filename" in resp_json, "report_filename missing in /orchestrate response!"
    assert "report_text" in resp_json, "report_text missing in /orchestrate response!"
    assert "metadata" in resp_json, "metadata missing in /orchestrate response!"
    assert resp_json["metadata"]["task_type"] == "Classification"
    assert resp_json["metadata"]["best_model"] is not None

    report_filename = resp_json["report_filename"]
    print(f"  ✅ /orchestrate completed successfully! Best model: {resp_json['metadata']['best_model']}")
    print(f"  ✅ Generated report filename: '{report_filename}'")
    print(f"  ✅ Report text length: {len(resp_json['report_text'])} characters")

    # 5. Report Download Test
    print(f"\n[5/7] Testing GET /report/{report_filename} ...")
    res = client.get(f"/report/{report_filename}")
    assert res.status_code == 200, f"Report download failed: {res.status_code}"
    assert len(res.content) > 0, "Report response content is empty!"
    print(f"  ✅ GET /report/{report_filename} served {len(res.content)} bytes (media-type: {res.headers.get('content-type')})")

    txt_filename = report_filename.replace(".pdf", ".txt")
    res_txt = client.get(f"/report/{txt_filename}")
    if res_txt.status_code == 200:
        print(f"  ✅ GET /report/{txt_filename} served {len(res_txt.content)} bytes (TXT report)")

    # 6. Security: Path Traversal Tests
    print("\n[6/7] Testing Security & Path Traversal Prevention ...")
    traversal_payloads = [
        "../../etc/passwd",
        "..\\..\\Windows\\win.ini",
        "....//....//config.py",
        "non_existent_report_12345.pdf",
    ]
    for payload in traversal_payloads:
        res = client.get(f"/report/{payload}")
        assert res.status_code in (400, 403, 404), f"Security flaw: payload '{payload}' returned {res.status_code}"
        print(f"  🔒 Payload '{payload}' safely blocked or returned {res.status_code}")

    # 7. Invalid Upload Validation Tests
    print("\n[7/7] Testing File Upload Validation ...")
    res_empty = client.post(
        "/orchestrate",
        files={"file": ("empty.csv", b"", "text/csv")},
        data={"target_column": "Fault"}
    )
    assert res_empty.status_code == 400, f"Empty CSV returned {res_empty.status_code} instead of 400"
    print("  🔒 Empty CSV upload safely rejected with HTTP 400")

    res_txt = client.post(
        "/orchestrate",
        files={"file": ("test.txt", b"some text", "text/plain")},
        data={"target_column": "Fault"}
    )
    assert res_txt.status_code == 400, f"Non-CSV upload returned {res_txt.status_code} instead of 400"
    print("  🔒 Non-CSV upload safely rejected with HTTP 400")

    print("\n" + "=" * 70)
    print("🎉 ALL 7 DEPLOYMENT VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
