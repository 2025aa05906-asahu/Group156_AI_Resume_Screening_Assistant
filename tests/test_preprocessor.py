from preprocessing.text_preprocessor import TextPreprocessor


def test_clean_text():
    text = "Python Developer! " "Contact: test@example.com"

    result = TextPreprocessor.clean_text(text)

    assert "python" in result
    assert "test@example.com" not in result


def test_preprocess_returns_string():
    text = "Python developers work with SQL."

    result = TextPreprocessor.preprocess(text)

    assert isinstance(result, str)
    assert len(result) > 0


def test_empty_text():
    result = TextPreprocessor.preprocess("")

    assert result == ""
