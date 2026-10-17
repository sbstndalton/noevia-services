//! Rust owns guard/startup cases; the fixed seam executes the actual Python app only.
#![forbid(unsafe_code)]
use std::{
    error::Error,
    fs,
    io::Read,
    os::{
        fd::OwnedFd,
        unix::{fs::DirBuilderExt, net::UnixStream},
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const KEY: &str = "NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED";
const SEAM: &str = r#"
import json,os,sys
from app import main,db,bench,services
from fastapi.testclient import TestClient
seen=[]
main.hw.start_sampler=lambda:seen.append('sampler')
bench.reap_orphans=lambda:seen.append('reap')
services._docker_client=lambda:None
client=TestClient(main.app,raise_server_exceptions=False)
functions={'add':db.add_prompt,'delete':db.delete_prompt,'seed':db.seed_bench_prompts,'setting':db.set_setting,'get':db.get_setting,'init':db.init,'startup':main._startup}
for q in json.loads(sys.argv[1]):
 try:
  if q['op']=='http':
   r=client.request(q['method'],q['path'],headers={'X-Model-Loader-Token':q['token']},content=q.get('body',''),follow_redirects=False)
   print(json.dumps({'status':r.status_code,'body':r.text},separators=(',',':')))
  elif q['op']=='count':
   with db._conn() as c:print(json.dumps(c.execute(q['sql']).fetchone()[0]))
  elif q['op']=='seen':print(json.dumps(seen,separators=(',',':')))
  else:print(json.dumps(functions[q['op']](*q.get('args',[])),separators=(',',':')))
 except Exception as e:print(json.dumps({'error':type(e).__name__,'message':str(e)},separators=(',',':')))
"#;
struct Root(PathBuf);
impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn root() -> Result<Root> {
    static SERIAL: AtomicUsize = AtomicUsize::new(0);
    let tick = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = std::env::temp_dir().join(format!(
        "native-prompts-qualification-{}-{tick}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    fs::DirBuilder::new().mode(0o700).create(&path)?;
    Ok(Root(path))
}
// Cleanup retains ownership until reap; OS kill/wait completion has no hard time bound.
struct OwnedProcess(Child);
impl Drop for OwnedProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
// Fair bounded nonblocking capture: one syscall per stream per deadline check.
// The fixed actual-app seam launches no subprocess; cleanup owns only its Python leader.
fn read_step(pipe: &mut UnixStream, bytes: &mut Vec<u8>) -> Result<bool> {
    let mut chunk = [0; 4096];
    match pipe.read(&mut chunk) {
        Ok(0) => Ok(true),
        Ok(n) => {
            require(
                bytes.len() + n <= 65_536,
                "actual app qualification output cap",
            )?;
            bytes.try_reserve(n)?;
            bytes.extend_from_slice(chunk.get(..n).ok_or("invalid capture count")?);
            Ok(false)
        }
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
            ) =>
        {
            Ok(false)
        }
        Err(e) => Err(e.into()),
    }
}
fn call(root: &Root, confirmation: Option<&str>, cases: &str) -> Result<Vec<String>> {
    let source = Path::new(file!()).canonicalize()?;
    let app = source
        .parent()
        .and_then(Path::parent)
        .ok_or("app source root")?;
    let models = root.0.join("models");
    fs::create_dir_all(&models)?;
    fs::write(models.join("models.ini"), "version = 1\n")?;
    let mut command = Command::new(
        std::env::var_os("PROMPTS_TEST_PYTHON").ok_or("PROMPTS_TEST_PYTHON required")?,
    );
    command
        .args(["-B", "-c", SEAM, cases])
        .current_dir(app)
        .env("MODELS_DIR", &models)
        .env("MODELS_INI_PATH", models.join("models.ini"))
        .env("DATA_DIR", root.0.join("data"))
        .env(
            "MODEL_LOADER_TOKEN",
            "synthetic-prompt-token-0123456789abcdef",
        )
        .env("LLAMA_CONTAINERS", "")
        .env("DOCKER_HOST", "unix:///absent-synthetic.sock")
        .env("MIGRATE_CACHE_RAM_ON_START", "false")
        .env("GGUF_PARSER", "python")
        .env("MODEL_AUTOCONFIG", "python")
        .env_remove("NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED")
        .env_remove("NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED")
        .env_remove("NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED")
        .env_remove(KEY);
    if let Some(value) = confirmation {
        command.env(KEY, value);
    }
    let output = capture(command, Duration::from_secs(12))?;
    Ok(String::from_utf8(output)?
        .lines()
        .map(str::to_owned)
        .collect())
}
pub(crate) struct Output {
    pub status: std::process::ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}
pub(crate) fn capture(command: Command, timeout: Duration) -> Result<Vec<u8>> {
    let output = capture_output(command, timeout)?;
    require(
        output.stderr.len() <= 65_536,
        "actual app qualification diagnostic cap",
    )?;
    require(
        output.status.success(),
        "actual app qualification process failed",
    )?;
    Ok(output.stdout)
}
pub(crate) fn capture_output(mut command: Command, timeout: Duration) -> Result<Output> {
    let (mut stdout, stdout_child) = UnixStream::pair()?;
    let (mut stderr, stderr_child) = UnixStream::pair()?;
    stdout.set_nonblocking(true)?;
    stderr.set_nonblocking(true)?;
    let mut child = OwnedProcess(
        command
            .stdin(Stdio::null())
            .stdout(Stdio::from(OwnedFd::from(stdout_child)))
            .stderr(Stdio::from(OwnedFd::from(stderr_child)))
            .spawn()?,
    );
    drop(command);
    let deadline = Instant::now() + timeout;
    let mut output = Vec::new();
    let mut diagnostics = Vec::new();
    let mut out_done = false;
    let mut err_done = false;
    let mut status = None;
    loop {
        if Instant::now() >= deadline {
            return Err("actual app qualification deadline".into());
        }
        if !out_done {
            out_done = read_step(&mut stdout, &mut output)?;
        }
        if !err_done {
            err_done = read_step(&mut stderr, &mut diagnostics)?;
        }
        if status.is_none() {
            status = child.0.try_wait()?;
        }
        if status.is_some() && out_done && err_done {
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    Ok(Output {
        status: status.ok_or("missing qualification status")?,
        stdout: output,
        stderr: diagnostics,
    })
}
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
#[derive(Debug)]
struct PolicyFailure(&'static str);
impl std::fmt::Display for PolicyFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl Error for PolicyFailure {}
fn policy_require(ok: bool, message: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Box::new(PolicyFailure(message)))
    }
}
#[cfg(test)]
fn assert_policy(result: Result<()>) {
    if let Err(error) = result {
        if let Some(policy) = error.downcast_ref::<PolicyFailure>() {
            panic!("PROMPTS_POLICY_ASSERT:{}", policy.0);
        }
        panic!("PROMPTS_INFRASTRUCTURE_FAILURE:{error}");
    }
}
fn guards() -> Result<()> {
    let root = root()?;
    let guarded = call(
        &root,
        Some("1"),
        r#"[{"op":"add","args":["synthetic","body"]},{"op":"delete","args":[1]},{"op":"seed"},{"op":"setting","args":["bench_prompts_seeded","1"]}]"#,
    )?;
    let expected =
        r#"{"error":"RuntimeError","message":"Native model manager owns prompt writes"}"#;
    policy_require(
        guarded.len() == 4 && guarded.iter().all(|answer| answer == expected),
        "all direct prompt writers refuse exact confirmation",
    )?;
    policy_require(
        !root.0.join("data").exists(),
        "guarded calls opened or created DB",
    )?;
    let allowed = call(
        &root,
        Some("1"),
        r#"[{"op":"init"},{"op":"setting","args":["other","sentinel"]},{"op":"get","args":["other"]},{"op":"count","sql":"SELECT count(*) FROM prompts"}]"#,
    )?;
    policy_require(
        allowed == ["null", "null", "\"sentinel\"", "0"],
        "unrelated kv and schema init remain available",
    )?;
    Ok(())
}
fn startup() -> Result<()> {
    for value in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some(" 1"),
        Some("1"),
        Some("yes"),
    ] {
        let root = root()?;
        let answers = call(
            &root,
            value,
            r#"[{"op":"startup"},{"op":"count","sql":"SELECT count(*) FROM prompts"},{"op":"get","args":["bench_prompts_seeded",""]},{"op":"seen"}]"#,
        )?;
        let expected = if value == Some("1") {
            ["null", "0", "\"\"", "[\"sampler\",\"reap\"]"]
        } else {
            ["null", "4", "\"1\"", "[\"sampler\",\"reap\"]"]
        };
        policy_require(
            answers == expected,
            "startup seed ownership and other hooks",
        )?;
    }
    Ok(())
}
fn http_guards() -> Result<()> {
    let root = root()?;
    let answers = call(
        &root,
        Some("1"),
        r#"[{"op":"http","method":"GET","path":"/api/v1/prompts","token":"wrong"},{"op":"http","method":"POST","path":"/api/v1/prompts","token":"wrong","body":"{"},{"op":"http","method":"DELETE","path":"/api/v1/prompts/1","token":"wrong"},{"op":"http","method":"GET","path":"/api/v1/prompts","token":"synthetic-prompt-token-0123456789abcdef"},{"op":"http","method":"POST","path":"/api/v1/prompts","token":"synthetic-prompt-token-0123456789abcdef","body":"{"},{"op":"http","method":"DELETE","path":"/api/v1/prompts/1","token":"synthetic-prompt-token-0123456789abcdef"},{"op":"http","method":"PUT","path":"/api/v1/prompts","token":"synthetic-prompt-token-0123456789abcdef"},{"op":"http","method":"GET","path":"/api/v1/prompts/1","token":"synthetic-prompt-token-0123456789abcdef"},{"op":"http","method":"DELETE","path":"/api/v1/prompts/1/extra","token":"synthetic-prompt-token-0123456789abcdef"}]"#,
    )?;
    let unauthorized = r#"{"status":401,"body":"{\"error\":\"unauthorized\"}"}"#;
    let owned =
        r#"{"status":503,"body":"{\"detail\":\"Native model manager owns this endpoint\"}"}"#;
    policy_require(
        answers.len() == 9
            && answers.iter().take(3).all(|a| a == unauthorized)
            && answers.iter().skip(3).take(3).all(|a| a == owned),
        "auth-first exact three method/path pairs",
    )?;
    policy_require(
        answers.get(6).is_some_and(|a| a.contains("\"status\":405"))
            && answers.get(7).is_some_and(|a| a.contains("\"status\":405"))
            && answers.get(8).is_some_and(|a| a.contains("\"status\":404")),
        "guard is method-aware and one item segment",
    )?;
    policy_require(!root.0.join("data").exists(), "guarded HTTP created DB")?;
    Ok(())
}
fn main() -> Result<()> {
    guards()?;
    startup()?;
    http_guards()?;
    println!("actual prompt direct guards, auth-first HTTP and once-only startup ownership pass");
    Ok(())
}
#[cfg(test)]
mod tests {
    fn synthetic(code: &str, timeout: std::time::Duration) -> super::Result<Vec<u8>> {
        let mut command = std::process::Command::new(
            std::env::var_os("PROMPTS_TEST_PYTHON").ok_or("PROMPTS_TEST_PYTHON required")?,
        );
        command.args(["-B", "-c", code]);
        super::capture(command, timeout)
    }
    #[test]
    fn only_typed_policy_errors_produce_policy_assertions() {
        fn payload(error: super::Result<()>) -> String {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                super::assert_policy(error)
            }))
            .unwrap_err();
            panic.downcast_ref::<String>().unwrap().clone()
        }
        assert_eq!(
            payload(super::policy_require(false, "fixed policy")),
            "PROMPTS_POLICY_ASSERT:fixed policy"
        );
        for message in ["setup", "spawn", "capture cap", "deadline", "fixed policy"] {
            assert_eq!(
                payload(Err(message.into())),
                format!("PROMPTS_INFRASTRUCTURE_FAILURE:{message}")
            );
        }
    }
    #[test]
    fn capture_caps_each_stream_and_observes_deadline_and_status() {
        use std::time::Duration;
        for code in [
            "import sys;sys.stdout.write('x'*65537)",
            "import sys;sys.stderr.write('x'*65537)",
        ] {
            assert_eq!(
                synthetic(code, Duration::from_secs(2))
                    .unwrap_err()
                    .to_string(),
                "actual app qualification output cap"
            );
        }
        assert_eq!(
            synthetic("raise SystemExit(7)", Duration::from_secs(2))
                .unwrap_err()
                .to_string(),
            "actual app qualification process failed"
        );
        assert_eq!(
            synthetic("import time;time.sleep(2)", Duration::from_millis(50))
                .unwrap_err()
                .to_string(),
            "actual app qualification deadline"
        );
        assert_eq!(
            synthetic("print('fixed')", Duration::from_secs(2)).unwrap(),
            b"fixed\n"
        );
    }
    #[test]
    fn actual_direct_writers_are_guarded_before_database() {
        super::assert_policy(super::guards());
    }
    #[test]
    fn actual_startup_seed_has_exact_confirmation_and_preserves_hooks() {
        super::assert_policy(super::startup());
    }
    #[test]
    fn actual_private_prompt_http_is_authenticated_and_method_aware() {
        super::assert_policy(super::http_guards());
    }
}
