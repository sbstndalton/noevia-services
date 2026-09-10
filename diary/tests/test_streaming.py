import asyncio
import json
import threading
import httpx
import pytest
from agent.llm import LLMClient, LLMError
from agent.streaming import exchange_stream


def test_provider_reasoning_is_emitted_before_answer_and_marker_is_not_shown():
    chunks = [
        {'reasoning_content': 'Synthetic live thinking'},
        {'content': 'A synthetic reply. ' * 20},
        {'content': '\n[LOG: ok]'},
    ]
    def provider(request):
        assert json.loads(request.content)['stream'] is True
        content = ''.join('data: '+json.dumps({'choices':[{'delta':d}]})+'\n\n' for d in chunks)+'data: [DONE]\n\n'
        return httpx.Response(200, text=content, headers={'Content-Type':'text/event-stream'})
    client=LLMClient('http://synthetic/v1')
    client._client.close()
    client._client=httpx.Client(base_url='http://synthetic/v1',transport=httpx.MockTransport(provider))
    events=[]
    result=client.chat_stream([],events.append)
    assert events[0] == {'type':'reasoning','text':'Synthetic live thinking'}
    assert any(e['type']=='delta' for e in events)
    assert events[-1]['type']=='answer'
    assert all('[LOG:' not in e.get('text','') for e in events)
    assert result.endswith('[LOG: ok]')
    assert result.reasoning=='Synthetic live thinking'
    client.close()


def test_truncated_provider_output_is_not_accepted_as_a_complete_answer():
    client=LLMClient('http://synthetic/v1')
    client._client.close()
    client._client=httpx.Client(base_url='http://synthetic/v1',transport=httpx.MockTransport(lambda r:httpx.Response(200,text='data: {"choices":[{"delta":{"content":"Partial answer"}}]}\n\n',headers={'Content-Type':'text/event-stream'})))
    events=[]
    with pytest.raises(LLMError,match='before completion'):
        client.chat_stream([],events.append)
    assert not any(e['type']=='answer' for e in events)
    client.close()


def test_exchange_progress_is_available_while_work_is_blocked():
    release=threading.Event()
    calls=[]
    def work(emit):
        calls.append('once')
        emit({'type':'reasoning','text':'Synthetic live thought'})
        assert release.wait(5)
        emit({'type':'answer','text':'Synthetic answer'})
        return {'reply':'Synthetic answer','decision':'skipped'}
    async def check():
        response=exchange_stream(work)
        stream=response.body_iterator
        assert 'Opening diary' in await anext(stream)
        assert 'Synthetic live thought' in await anext(stream)
        assert not release.is_set()
        release.set()
        tail=''.join([event async for event in stream])
        assert tail.index('answer') < tail.index('diary') < tail.index('done')
    asyncio.run(check())
    assert calls==['once']


def test_v1_stream_keeps_tenant_and_one_exchange(monkeypatch):
    import agent.app as appmod
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    tenant='11111111-1111-4111-8111-111111111111'
    st=SimpleNamespace()
    calls=[]
    monkeypatch.setattr(appmod,'check_auth',lambda r:True)
    def state(request):
        assert request.headers['X-Cowork-User-ID']==tenant
        return st
    monkeypatch.setattr(appmod,'_tenant_state',state)
    def exchange(actual,message,sid,owner,now,day,background,extra,emit):
        assert actual is st and owner==tenant
        calls.append(message)
        emit({'type':'status','text':'Reading synthetic context'})
        emit({'type':'reasoning','text':'Synthetic thought'})
        emit({'type':'answer','text':'Synthetic answer'})
        return {'reply':'Synthetic answer','decision':'skipped','xid':None}
    monkeypatch.setattr(appmod,'_run_exchange',exchange)
    r=TestClient(appmod.app).post('/v1/chat/completions',headers={'X-Cowork-User-ID':tenant},json={'stream':True,'diary_events':True,'session_id':'stream-test','messages':[{'role':'user','content':'Synthetic message'}]})
    assert r.status_code==200
    assert 'text/event-stream' in r.headers['content-type']
    assert calls==['Synthetic message']
    events=[json.loads(line[6:]) for line in r.text.splitlines() if line.startswith('data: ')]
    assert [e['type'] for e in events]==['status','status','reasoning','answer','diary','done']
    appmod.SESSIONS.pop(f'{tenant}:stream-test',None)
