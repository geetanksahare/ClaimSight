from claimsight.reporting.model_card import generate_model_card


def test_generate_model_card(tmp_path):
    out_card = tmp_path / "cost_test_card.md"
    card_path = generate_model_card("cost", output_path=out_card)
    assert card_path.exists()
    content = card_path.read_text(encoding="utf-8")
    assert "Cost Intelligence Model" in content
    assert "Built on public/synthetic data" in content
