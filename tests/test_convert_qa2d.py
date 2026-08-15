from scripts.convert_qa2d import claim_key, qa2d_prompt


def test_qa2d_prompt_matches_model_contract() -> None:
    assert qa2d_prompt("Where is Carmen?", "Abruzzo.") == "where is carmen. abruzzo"
    assert claim_key("Who arrived?", "Rhea") != claim_key("Who arrived?", "Mina")
