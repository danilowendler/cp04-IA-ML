from src.preprocessing import clean_review


def test_clean_review_removes_html_and_normalizes_spaces():
    assert clean_review("Great<br /> movie!  <b>Really</b>") == "Great movie! Really"
