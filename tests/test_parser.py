"""Tests for parser module."""

from parser.parser import detect_language, get_parser, SUPPORTED_LANGUAGES
from parser.extract import CodeEntity, extract_entities


def test_detect_language_python():
    assert detect_language("foo.py") == "python"


def test_detect_language_java():
    assert detect_language("Foo.java") == "java"


def test_detect_language_javascript():
    assert detect_language("app.js") == "javascript"
    assert detect_language("app.jsx") == "javascript"
    assert detect_language("app.mjs") == "javascript"


def test_detect_language_go():
    assert detect_language("main.go") == "go"


def test_detect_language_ruby():
    assert detect_language("app.rb") == "ruby"


def test_detect_language_php():
    assert detect_language("index.php") == "php"


def test_detect_language_unknown():
    assert detect_language("foo.txt") is None


def test_supported_languages_count():
    assert len(SUPPORTED_LANGUAGES) == 6


def test_get_parser_python():
    parser = get_parser("python")
    assert parser is not None


def test_get_parser_unsupported():
    try:
        get_parser("brainfuck")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported language" in str(e)


def test_extract_python_function():
    code = 'def hello():\n    """Say hello."""\n    return "world"'
    tree = get_parser("python").parse(code.encode("utf-8"))
    entities = extract_entities(code, tree, "python", "test-repo", "test.py")
    assert len(entities) >= 1
    func = entities[0]
    assert func.entity_type == "function"
    assert func.function_name == "hello"
    assert "Say hello" in (func.docstring or "")
    assert func.language == "python"
    assert func.repository == "test-repo"
    assert func.file_path == "test.py"


def test_extract_python_class():
    code = 'class MyClass:\n    def method(self):\n        pass'
    tree = get_parser("python").parse(code.encode("utf-8"))
    entities = extract_entities(code, tree, "python", "repo", "f.py")
    class_entities = [e for e in entities if e.entity_type == "class"]
    method_entities = [e for e in entities if e.entity_type == "method"]
    assert len(class_entities) >= 1
    assert class_entities[0].class_name == "MyClass"
    assert len(method_entities) >= 1


def test_extract_java_function():
    code = 'public class Foo {\n    public void bar() {\n    }\n}'
    tree = get_parser("java").parse(code.encode("utf-8"))
    entities = extract_entities(code, tree, "java", "repo", "Foo.java")
    assert len(entities) >= 1


def test_code_entity_identifier():
    entity = CodeEntity(
        repository="repo",
        file_path="f.py",
        language="python",
        entity_type="method",
        function_name="bar",
        class_name="Foo",
        signature="def bar(self):",
        docstring=None,
        source_code="def bar(self): pass",
        start_line=1,
        end_line=1,
    )
    assert entity.identifier == "Foo.bar"


def test_code_entity_to_structured_text():
    entity = CodeEntity(
        repository="repo",
        file_path="f.py",
        language="python",
        entity_type="function",
        function_name="hello",
        class_name=None,
        signature="def hello():",
        docstring="Say hello.",
        source_code="def hello():\n    pass",
        start_line=1,
        end_line=2,
    )
    text = entity.to_structured_text()
    assert "Language: Python" in text
    assert "Function: hello" in text
    assert "Signature: def hello():" in text
    assert "Documentation: Say hello." in text
    assert "def hello():" in text
