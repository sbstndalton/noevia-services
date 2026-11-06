"""Offline image-entrypoint and Python compatibility-spelling checks.
Run with Python 3.12 and an evidence directory outside the source checkout.
The shell's exec is intercepted to inspect argv; no server/binary is started.
"""
import ast,json,os,pathlib,shutil,subprocess,sys
P=pathlib.Path
root=P(__file__).resolve().parents[1];evidence=P(sys.argv[1]).resolve();evidence.mkdir(parents=True,exist_ok=True)
entry=root/'entrypoint.sh'
subprocess.run(['sh','-n',str(entry)],check=True)
cases=[(['backup_restore','root with spaces','manifest','new'],['/usr/local/bin/diary-backup-restore','root with spaces','manifest','new']),(['backup-restore','--help'],['/usr/local/bin/diary-backup-restore','--help']),(['diary-backup-restore','--','-name'],['/usr/local/bin/diary-backup-restore','--','-name']),(['config','path with spaces'],['/usr/local/bin/diary-config','path with spaces']),(['diary-config'],['/usr/local/bin/diary-config']),(['python','-m','agent.backup_restore','a','b','c'],['/usr/local/bin/diary-backup-restore','a','b','c']),(['python3','-m','agent.backup_restore','--help'],['/usr/local/bin/diary-backup-restore','--help']),(['uvicorn','agent.app:app','--port','8010'],['uvicorn','agent.app:app','--port','8010']),(['python','-m','unrelated'],['python','-m','unrelated'])]
for args,expected in cases:
    p=subprocess.run(['bash','-c','exec() { printf "%s\\0" "$@"; exit 0; }; script=$1; shift; . "$script"','test',str(entry),*args],capture_output=True)
    actual=p.stdout.decode().split('\0')[:-1]
    assert p.returncode==0 and actual==expected,(args,p.returncode,actual,expected,p.stderr)
# Verify the actual compatibility module's execvp, avoiding imports of app/storage.
package=evidence/'shim/agent';package.mkdir(parents=True,exist_ok=True)
(package/'__init__.py').write_text('')
shutil.copyfile(root/'agent/backup_restore.py',package/'backup_restore.py')
tree=ast.parse((root/'agent/managed_storage.py').read_text());safe=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='safe_key')
(package/'managed_storage.py').write_text(ast.unparse(safe)+'\n')
bin_dir=evidence/'shim/bin';bin_dir.mkdir(parents=True,exist_ok=True)
stub=bin_dir/'diary-backup-restore';stub.write_text('#!'+sys.executable+'\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(23)\n');stub.chmod(0o700)
env={**os.environ,'PYTHONPATH':str(package.parent),'PYTHONDONTWRITEBYTECODE':'1','PATH':str(bin_dir)+os.pathsep+os.environ.get('PATH','')}
p=subprocess.run([sys.executable,'-m','agent.backup_restore','a with spaces','b','c'],env=env,capture_output=True,text=True)
assert p.returncode==23 and json.loads(p.stdout)==['a with spaces','b','c'] and not p.stderr,(p.returncode,p.stdout,p.stderr)
print('11/11 offline dispatch checks passed')
