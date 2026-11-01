import io
import json
from unittest.mock import Mock
from urllib.error import URLError
import pytest
from services import search_assistant as sa

class Response:
    def __init__(self, payload):
        self.body = io.BytesIO(json.dumps(payload).encode())
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return self.body.read()


def test_ollama_payload_and_answer(monkeypatch):
    seen = {}
    def fake_urlopen(request, timeout):
        seen.update(json.loads(request.data.decode()))
        return Response({'message': {'content': '  Please contact support.  '}})
    monkeypatch.setattr(sa, 'urlopen', fake_urlopen)
    article = sa.KnowledgeArticle('CONTACT', 'customer_service', 'Contact support.')
    assert sa.generate_grounded_answer('Help', article, timeout=1) == 'Please contact support.'
    assert seen['stream'] is False and seen['think'] is False
    assert seen['messages'][0]['role'] == 'system'
    assert seen['options']['num_predict'] == 256


@pytest.mark.parametrize('payload', [{}, {'message': {}}, {'message': {'content': '  '}}])
def test_empty_ollama_answer_is_error(monkeypatch, payload):
    monkeypatch.setattr(sa, 'urlopen', lambda *a, **kw: Response(payload))
    with pytest.raises(RuntimeError, match='empty answer'):
        sa.generate_grounded_answer('Help', sa.KnowledgeArticle('CONTACT', 'customer_service', 'Contact support.'))


def test_ollama_unavailable(monkeypatch):
    monkeypatch.setattr(sa, 'urlopen', Mock(side_effect=URLError('connection refused')))
    with pytest.raises(RuntimeError, match='Cannot reach Ollama'):
        sa.generate_grounded_answer('Help', sa.KnowledgeArticle('CONTACT', 'customer_service', 'Contact support.'))
    assert sa.is_ollama_available('http://localhost:11434') is False
