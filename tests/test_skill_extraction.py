from preprocessing.skill_extraction import (
    SkillExtractor,
)


def test_extract_python():
    text = "Experienced Python developer."

    skills = SkillExtractor.extract_skills(text)

    assert "python" in skills


def test_extract_multiple_skills():
    text = (
        "Python, SQL, Docker and Machine Learning"
    )

    skills = SkillExtractor.extract_skills(text)

    assert "python" in skills
    assert "sql" in skills
    assert "docker" in skills
    assert "machine learning" in skills


def test_empty_input():
    assert SkillExtractor.extract_skills("") == []