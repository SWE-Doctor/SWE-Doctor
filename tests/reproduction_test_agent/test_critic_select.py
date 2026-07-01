"""Tests for select_per_aspect helper in critic.py."""
import pytest
from reproduction_test_agent.critic import select_per_aspect


def test_best_per_aspect_picks_highest_score():
    """Keep highest-score critic_ok=True entry per aspect."""
    evaluated = [
        {"aspect_id": "a0", "critic_ok": True, "score": 0.6, "candidate_id": "a0c0"},
        {"aspect_id": "a0", "critic_ok": True, "score": 0.9, "candidate_id": "a0c1"},
        {"aspect_id": "a1", "critic_ok": True, "score": 0.4, "candidate_id": "a1c0"},
        {"aspect_id": "a1", "critic_ok": True, "score": 0.7, "candidate_id": "a1c1"},
    ]
    result = select_per_aspect(evaluated)

    assert len(result) == 2
    # a0: highest score is 0.9 (a0c1)
    a0 = next(r for r in result if r["aspect_id"] == "a0")
    assert a0["candidate_id"] == "a0c1"
    assert a0["score"] == 0.9
    # a1: highest score is 0.7 (a1c1)
    a1 = next(r for r in result if r["aspect_id"] == "a1")
    assert a1["candidate_id"] == "a1c1"
    assert a1["score"] == 0.7


def test_all_rejected_aspect_dropped():
    """Aspects where all candidates have critic_ok=False are excluded."""
    evaluated = [
        {"aspect_id": "a0", "critic_ok": True, "score": 0.8, "candidate_id": "a0c0"},
        {"aspect_id": "a1", "critic_ok": False, "score": 0.5, "candidate_id": "a1c0"},
        {"aspect_id": "a1", "critic_ok": False, "score": 0.3, "candidate_id": "a1c1"},
    ]
    result = select_per_aspect(evaluated)

    assert len(result) == 1
    assert result[0]["aspect_id"] == "a0"
    assert result[0]["candidate_id"] == "a0c0"


def test_stable_order_matches_first_seen():
    """Output order follows first occurrence of each aspect_id in input."""
    evaluated = [
        {"aspect_id": "z", "critic_ok": True, "score": 0.5, "candidate_id": "z0"},
        {"aspect_id": "a", "critic_ok": True, "score": 0.5, "candidate_id": "a0"},
        {"aspect_id": "m", "critic_ok": True, "score": 0.5, "candidate_id": "m0"},
    ]
    result = select_per_aspect(evaluated)

    assert [r["aspect_id"] for r in result] == ["z", "a", "m"]
