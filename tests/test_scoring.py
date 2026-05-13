from agent_pathologies.tasks.scoring import (
    contains_value,
    extract_last_integer,
    extract_letter_choice,
    score_integer,
    score_letter,
    score_value,
)


def test_extract_last_integer_handles_commas():
    assert extract_last_integer("The answer is 1,024.") == 1024


def test_extract_last_integer_signed():
    assert extract_last_integer("Result: -47") == -47


def test_extract_last_integer_none_when_no_int():
    assert extract_last_integer("I don't know.") is None


def test_extract_last_integer_picks_last():
    # Avoids matching an operand from the prompt copy-back.
    assert extract_last_integer("Compute 23 + 47, result: 70.") == 70


def test_extract_letter_explicit():
    assert extract_letter_choice("The answer is B.") == "B"
    assert extract_letter_choice("option (C) is best") == "C"


def test_extract_letter_fallback_first_isolated():
    assert extract_letter_choice("I would go with D.") == "D"


def test_extract_letter_none_when_absent():
    assert extract_letter_choice("Twelve.") is None


def test_contains_value_word_boundary():
    assert contains_value("project codename is PHOENIX.", "PHOENIX")
    # Should not match substring inside other words.
    assert not contains_value("phoenixianism", "PHOENIX")
    assert contains_value("Budget cap = 2.4M", "2.4M")


def test_score_integer_strict():
    assert score_integer("The answer is 202.", "202") is True
    assert score_integer("The answer is 203.", "202") is False
    assert score_integer("I cannot compute that.", "202") is None  # excludable


def test_score_letter_strict():
    assert score_letter("The answer is C.", "C") is True
    assert score_letter("The answer is B.", "C") is False
    assert score_letter("idk", "C") is None


def test_score_value_word_boundary_avoids_false_match():
    assert score_value("It is PHOENIX.", "PHOENIX") is True
    assert score_value("phoenixianism", "PHOENIX") is False
    assert score_value("", "PHOENIX") is None
