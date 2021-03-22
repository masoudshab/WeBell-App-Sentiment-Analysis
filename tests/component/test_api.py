import pytest


@pytest.fixture
def app():
    # TODO : Use fixture to create the application which will be use in latest
    #  tests. I recommend using Flask 'test_client()' for testing.
    app = ...

    return app


def test_health_check(app):
    resp = app.get("/health")

    assert resp.status_code == 200, "Health check failed."
