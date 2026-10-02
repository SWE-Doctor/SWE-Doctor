"""ReproTestConfig must carry a `language` field (default python) so the rest
of the Stage-1 pipeline can pick the right langpack. Go adds 'go'."""
from reproduction_test_agent.config import ReproTestConfig


def test_config_default_language_is_python():
    assert ReproTestConfig().language == "python"


def test_config_accepts_go_language():
    assert ReproTestConfig(language="go").language == "go"
