from unittest.mock import Mock
import pytest
from services import preprocessing as pp
from exceptions import UnsupportedLanguageError


def test_blank_text_bypasses_detector(monkeypatch):
    detector = Mock(side_effect=AssertionError('must not call'))
    monkeypatch.setattr(pp, 'detect', detector)
    assert pp.PreprocessingService().process('  ') == ('  ', False)


def test_non_english_rejected(monkeypatch):
    monkeypatch.setattr(pp, 'detect', lambda **kw: {'lang': 'ru'})
    with pytest.raises(UnsupportedLanguageError):
        pp.PreprocessingService().process('Привет, где заказ?')


def test_length_boundary_and_language_before_truncation(monkeypatch):
    observed = []
    def detect(**kw):
        observed.append(kw['text'])
        return {'lang': 'en'}
    monkeypatch.setattr(pp, 'detect', detect)
    monkeypatch.setattr(pp, 'correct_text', lambda text, max_dist: (text, []))
    service = pp.PreprocessingService()
    assert service.process('a' * 300)[1] is False
    text, truncated = service.process('a' * 301)
    assert truncated is True and len(text) == 300
    assert len(observed[-1]) == 301


def test_detector_failure_rejects_fail_closed(monkeypatch):
    monkeypatch.setattr(pp, 'detect', Mock(side_effect=RuntimeError('offline')))
    with pytest.raises(UnsupportedLanguageError):
        pp.PreprocessingService().process('Hello, I need support')


def test_punctuation_preserved_when_correction_does_not_change_length(monkeypatch):
    monkeypatch.setattr(pp, 'correct_text', lambda text, max_dist: (text, []))
    result, positions = pp.PreprocessingService()._correct_preserving_punctuation('Hello, world!')
    assert result == 'Hello, world!'
    assert [char for _, char in positions] == [',', '!']
