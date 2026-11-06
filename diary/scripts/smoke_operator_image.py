"""Run only offline installed operator commands against synthetic files. No HTTP/model calls."""
import hashlib,json,os,pathlib,subprocess,tempfile
P=pathlib.Path
canary='synthetic-operator-credential-canary-814'
with tempfile.TemporaryDirectory(prefix='operator-',dir=os.environ['DIARY_OPERATOR_SMOKE_ROOT']) as temp:
 root=P(temp);tenant='00000000-0000-4000-8000-000000000123';key=f'noevia-backups/{tenant}/objects/synthetic/note';payload=b'synthetic diary\x00\xff\n';source=root/key;source.parent.mkdir(parents=True);source.write_bytes(payload)
 body=json.dumps({'format':'noevia-diary-backup-v1','tenant':tenant,'files':[{'path':'notes/note.bin','object':key,'sha256':hashlib.sha256(payload).hexdigest()}],'directories':['empty'],'settings':{'synthetic':True}},ensure_ascii=False).encode();manifest=root/(hashlib.sha256(body).hexdigest()+'.json');manifest.write_bytes(body)
 commands=[['diary-backup-restore'],['diary-entrypoint','backup_restore'],['diary-entrypoint','python','-m','agent.backup_restore'],['python','-m','agent.backup_restore']]
 for i,command in enumerate(commands):
  dest=root/f'restored-{i}';p=subprocess.run(command+[str(root),str(manifest),str(dest)],capture_output=True,timeout=15)
  assert p.returncode==0,'installed restore command failed'
  assert json.loads(p.stdout)=={'files':1,'bytes':len(payload),'tenant':tenant,'settings':{'synthetic':True}},'restore output mismatch'
  assert (dest/'notes/note.bin').read_bytes()==payload and (dest/'empty').is_dir(),'restore bytes mismatch'
 config=root/'config.yaml';config.write_text(f'llm:\n  api_key: {canary}\n  chat_model: {canary}\ncorpus:\n  webdav:\n    password: {canary}\n  s3:\n    secret_key: {canary}\n    session_token: {canary}\n')
 env={**os.environ,'LLM_API_KEY':canary,'WEBDAV_PASSWORD':canary,'S3_ACCESS_KEY_ID':canary,'S3_SECRET_ACCESS_KEY':canary,'S3_SESSION_TOKEN':canary,'DIARY_AUTH_TOKEN':canary}
 for command in [['diary-config'],['diary-entrypoint','config']]:
  p=subprocess.run(command+[str(config)],env=env,capture_output=True,timeout=15)
  assert canary.encode() not in p.stdout+p.stderr,'credential escaped diagnostic'
  assert p.returncode==0 and json.loads(p.stdout)=={'configured':True} and not p.stderr,'config status mismatch'
 config.write_text(f'x: !invalid {canary}\n')
 p=subprocess.run(['diary-entrypoint','config',str(config)],env=env,capture_output=True,timeout=15)
 assert p.returncode==1 and canary.encode() not in p.stdout+p.stderr,'credential escaped config error'
print('7/7 installed operator image checks passed')
