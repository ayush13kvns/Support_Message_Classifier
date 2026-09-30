import csv
import json
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.train import train
from app.text import normalize

@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    train('data/demo.csv',tmp_path,demo=True)
    monkeypatch.setenv('MODEL_DIR',str(tmp_path))
    monkeypatch.setenv('APP_ENV','development')
    monkeypatch.setenv('API_KEY','test-key')
    monkeypatch.setenv('CONFIDENCE_THRESHOLD','0.65')
    return tmp_path

@pytest.fixture
def client(model_dir):
    with TestClient(create_app()) as client:
        yield client

AUTH={'X-API-Key':'test-key'}

def test_health_and_auth(client):
    assert client.get('/health/ready').status_code==200
    assert client.post('/v1/classify',json={'text':'payment refund'}).status_code==401
    assert client.post('/v1/classify',headers={'X-API-Key':'wrong'},json={'text':'payment refund'}).status_code==401
    assert client.get('/health/live').headers['X-Request-ID']

@pytest.mark.parametrize('text,category',[
    ('Please refund the double charge on my invoice','billing'),
    ('The app crashes with a server connection error','technical_issue'),
    ('Where can I find a brochure and product demonstration','general_inquiry')])
def test_classification(client,text,category):
    response=client.post('/v1/classify',headers=AUTH,json={'text':text})
    assert response.status_code==200
    data=response.json()
    assert data['category']==category
    assert sum(data['probabilities'].values())==pytest.approx(1)
    assert data['requires_review'] and 'demo_model' in data['review_reasons']

def test_batch_and_unknown(client):
    response=client.post('/v1/classify/batch',headers=AUTH,json={'messages':[{'text':'xyzzyqv'}, {'text':'refund invoice'}]})
    assert response.status_code==200 and len(response.json())==2
    assert 'no_known_vocabulary' in response.json()[0]['review_reasons']
    assert 'low_confidence' in response.json()[0]['review_reasons']

@pytest.mark.parametrize('body',[{'text':'  '},{'text':'x'*10001},{'text':123},{}, {'text':''}])
def test_invalid(client,body):
    response=client.post('/v1/classify',headers=AUTH,json=body)
    assert response.status_code==422
    assert response.json()=={'detail':'Invalid request. Check text length, nonblank input, and batch size.'}

def test_batch_limit(client):
    assert client.post('/v1/classify/batch',headers=AUTH,json={'messages':[]}).status_code==422
    assert client.post('/v1/classify/batch',headers=AUTH,json={'messages':[{'text':'hi'}]*101}).status_code==422

def test_preprocessing():
    assert normalize(' Test@EXAMPLE.com https://example.com 1234567890 ')=='emailtoken urltoken numbertoken'
    assert normalize('I do NOT want a refund')=='i do not want a refund'

def test_checksum(model_dir):
    (model_dir/'model.joblib').write_bytes(b'invalid')
    with pytest.raises(RuntimeError,match='checksum'):
        with TestClient(create_app()): pass

def test_production_key(model_dir,monkeypatch):
    monkeypatch.setenv('APP_ENV','production')
    with pytest.raises(RuntimeError,match='API_KEY'):
        with TestClient(create_app()): pass

def test_production_demo(model_dir,monkeypatch):
    monkeypatch.setenv('APP_ENV','production');monkeypatch.setenv('API_KEY','a'*32)
    with pytest.raises(RuntimeError,match='real-data'):
        with TestClient(create_app()): pass

def test_production_real_metadata(model_dir,monkeypatch):
    # Exercise production configuration; this fixture is not evidence of real-data accuracy.
    meta=json.loads((model_dir/'metadata.json').read_text());meta['demo_data']=False
    (model_dir/'metadata.json').write_text(json.dumps(meta))
    monkeypatch.setenv('APP_ENV','production');monkeypatch.setenv('API_KEY','a'*32)
    with TestClient(create_app()) as c:
        assert c.post('/v1/classify',headers={'X-API-Key':'a'*32},json={'text':'refund'}).status_code==200

def test_conflicting_duplicates(tmp_path):
    path=tmp_path/'bad.csv'
    path.write_text('text,label\nsame,billing\nsame,technical_issue\n')
    with pytest.raises(ValueError,match='Conflicting'):
        train(path,tmp_path/'out')

def test_small_data(tmp_path):
    path=tmp_path/'small.csv';path.write_text('text,label\nhello,billing\n')
    with pytest.raises(ValueError,match='10 distinct'):
        train(path,tmp_path/'out')
