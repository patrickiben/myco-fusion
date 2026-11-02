import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from myco_fusion.server import make_server


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, route, data, **headers):
        request = Request(self.url + route, data=json.dumps(data).encode(), headers={"Content-Type": "application/json", **headers})
        with urlopen(request) as response:
            return json.load(response)

    def test_numerical_api_serializes_finite_report(self):
        report = self.post("/api/bench", {"seed": 7, "noise": 0, "source": "idealized"})
        self.assertEqual(report["source"], "idealized")
        self.assertEqual([row["independent"] for row in report["scenarios"]], [3, 3, 3, 4])
        report = self.post("/api/bench", {"seed": 7})
        self.assertEqual(report["source"], "measured")
        self.assertEqual([row["independent"] for row in report["scenarios"]][:3], [3, 3, 3])
        self.assertGreater(report["scenarios"][3]["independent"], 3)

    def test_bad_inputs_are_actionable_errors(self):
        with self.assertRaises(HTTPError) as failure:
            self.post("/api/validate", {"csv": "bad,header\n1,2"})
        self.assertEqual(failure.exception.code, 400)
        self.assertIn("columns", json.load(failure.exception)["error"])
        failure.exception.close()

    def test_cross_origin_rejected(self):
        with self.assertRaises(HTTPError) as failure:
            self.post("/api/manifest", {}, Origin="https://unrelated.example")
        self.assertEqual(failure.exception.code, 403)
        failure.exception.close()

    def test_does_not_serve_arbitrary_local_files(self):
        with self.assertRaises(HTTPError) as failure:
            urlopen(self.url + "/../pyproject.toml")
        self.assertEqual(failure.exception.code, 404)
        failure.exception.close()

    def test_manifest_roundtrip(self):
        result = self.post("/api/manifest", dict(specimen_id="demo", session_id="s1", device="phone", regime="R3", reference_id="tile1"))
        self.assertEqual(result["manifest"]["status"], "draft_capture_record")

    def test_calibration_and_prediction_roundtrip(self):
        with urlopen(self.url + "/api/calibration-example") as response:
            dataset = json.load(response)
        dataset["uncertainty"] = "session_cluster"
        report = self.post("/api/calibrate", {"dataset": dataset})
        self.assertEqual(report["kind"], "calibration_evaluation")
        self.assertIsNotNone(report["evaluation"]["regimes"][0]["mae_interval"])
        capture = {k: dataset[k] for k in ("device", "pipeline_id", "rgb_encoding", "data_kind")}
        capture["observations"] = [{"observation_id": "synthetic-new", "rgb": [.3, .4, .5]}]
        result = self.post("/api/predict", {"model": report["model"], "capture": capture})
        self.assertEqual(len(result["predictions"][0]["estimated_reflectance"]), 27)
        self.assertEqual(result["evidence_type"], "inferred")
        dataset["observations"][-1]["session_id"] = dataset["observations"][0]["session_id"]
        with self.assertRaises(HTTPError) as failure:
            self.post("/api/calibrate", {"dataset": dataset})
        self.assertEqual(failure.exception.code, 400)
        self.assertIn("Session leakage", json.load(failure.exception)["error"])
        failure.exception.close()

    def test_validation_does_not_ignore_requested_uncertainty(self):
        with urlopen(self.url + "/api/example") as response:
            csv = json.load(response)["csv"]
        no_sessions = "specimen_id,regime,repeat,wavelength_nm,reference,estimate\n" + "".join(
            f"A,R1,1,{wavelength},0.5,0.5\n" for wavelength in range(420, 681, 10))
        with self.assertRaises(HTTPError) as failure:
            self.post("/api/validate", {"csv": no_sessions, "uncertainty": "session_cluster"})
        self.assertEqual(failure.exception.code, 400)
        self.assertIn("session_id", json.load(failure.exception)["error"])
        failure.exception.close()
        self.assertEqual(self.post("/api/validate", {"csv": csv, "uncertainty": "specimen"})["uncertainty_method"], "specimen")

    def test_repeatability_route_and_examples(self):
        with urlopen(self.url + "/api/repeatability-example") as response:
            csv = json.load(response)["csv"]
        report = self.post("/api/repeatability", {"csv": csv})
        self.assertEqual(report["kind"], "repeatability")
        self.assertEqual(report["data_kind"], "synthetic")

    def test_interface_and_findings_are_served_from_the_package(self):
        for route in ("/", "/app.js", "/styles.css", "/api/findings", "/api/status"):
            with urlopen(self.url + route) as response:
                self.assertEqual(response.status, 200)
