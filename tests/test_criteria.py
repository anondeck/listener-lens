from earshift_bakeoff.config import criteria_sha256, load_config, verify_criteria_hash


def test_locked_criteria_hash_matches_config() -> None:
    config = load_config()
    assert criteria_sha256() == config["criteria_sha256"]
    assert verify_criteria_hash(config) == config["criteria_sha256"]
