"""Quick smoke test of the web app (no browser needed):  python test_app.py"""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
print("App loaded OK, tabs:", len(at.tabs))

next(b for b in at.button if b.label == "Classify mushroom").click().run()
assert not at.exception, at.exception
msg = [e.value for e in list(at.error) + list(at.success) if "Predicted" in e.value]
print("Prediction:", msg[0].replace("\n", " "))
