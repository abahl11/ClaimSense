"""Headless test of the Streamlit app with Streamlit's AppTest (skipped until models exist)."""

import pytest

from src.config import PROJECT_ROOT, load_config

CFG = load_config()
pytestmark = pytest.mark.skipif(
    not (CFG.path("regression_model").is_file() and CFG.path("classifier_model").is_file()),
    reason="models not trained yet (run the pipeline)",
)


def test_app_scores_the_default_claim():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    app.run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Expected ultimate cost" in labels
    assert len(app.table) == 1
