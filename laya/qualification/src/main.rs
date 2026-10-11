#![forbid(unsafe_code)]
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    io::{Read, Write},
    net::TcpStream,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const CAP: usize = 16_384;
const OUTPUT_CAP: u64 = 65_536;
const SERVER_HASH: &str = "fa9965aa8d2606699e0cb1a4c16a71106e7fd0762b54e30656dfd224a17383e3";
const ENGINE_DOCKER_HASH: &str = "3e63e74b0265b9a524d601a1ad1222335122a4988671c1a22924574d7d20d964";
// Fixed ML collaborators: the actual retained server remains the policy/IPC
// oracle. No case generation, HTTP client or test assertions run in Python.
const TORCH: &str = "def set_num_threads(n):pass\ndef set_num_interop_threads(n):pass\n";
const ENGINE: &str = r#"import os,time,json,subprocess
ROOT='/tmp/laya-synthetic'
class Tok:
 def encode(self,s):return list(s)
class Agent:
 cfg={'max_len':8192,'head_max_len':4096}
 tok=Tok()
 def predict(self,state,questions):
  with open(ROOT+'/events','a') as f:f.write(json.dumps(state)+'\n')
  if state=='__crash':os._exit(2)
  if state.startswith('__sleep:'):time.sleep(float(state.split(':')[1]))
  if state=='__child':
   child=subprocess.Popen(['python','-c','import time;time.sleep(30)'])
   with open(ROOT+'/child-pid','w') as f:f.write(str(child.pid))
  keys=list(questions['decision']['criteria'])
  choice=keys[-1] if state.startswith('__sleep:') else keys[0]
  return {'answers':{'decision':{'choice':choice,'probabilities':{k:int(k==choice) for k in keys}}}}
def load(path,device):
 assert (path,device)==('/model','cpu')
 assert open('/model/synthetic-marker').read()=='synthetic-only\n'
 assert os.environ.get('HF_HUB_OFFLINE')=='1'
 assert os.environ.get('TRANSFORMERS_OFFLINE')=='1'
 os.makedirs(ROOT,exist_ok=True)
 with open(ROOT+'/starts','a') as f:f.write(str(os.getpid())+'\n')
 if os.environ.get('LAYA_SYNTHETIC_FAIL')=='1':raise ValueError('synthetic startup failure')
 time.sleep(float(os.environ.get('LAYA_SYNTHETIC_STARTUP_SLEEP','0')))
 os.environ['LAYA_KEEP_WARM_S']='0'
 print('synthetic engine diagnostic',flush=True)
 os.write(1,b'synthetic native diagnostic\n')
 return Agent()
"#;
fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn guard(directory: &Path) -> Result<()> {
    require(
        digest(&fs::read(directory.join("server.py"))?) == SERVER_HASH,
        "actual merged worker hash drift",
    )?;
    let docker = fs::read_to_string(directory.join("Dockerfile"))?;
    let (global, _) = docker
        .split_once("FROM rust:")
        .ok_or("missing pinned native builder")?;
    require(
        global
            .lines()
            .filter(|line| *line == "ARG RUNTIME_IMAGE=python:3.12-slim")
            .count()
            == 1,
        "runtime ARG must remain global",
    )?;
    let (_, production) = docker
        .split_once("\nFROM ${RUNTIME_IMAGE}\n")
        .ok_or("missing production stage")?;
    let baseline = format!(
        "ARG RUNTIME_IMAGE=python:3.12-slim\nFROM ${{RUNTIME_IMAGE}}\n{}",
        production
    )
    .replace(
        "COPY --from=native-builder /native/release/noevia-laya /usr/local/bin/noevia-laya\n",
        "",
    )
    .replace(
        "ENTRYPOINT [\"noevia-laya\"]\n",
        "ENTRYPOINT [\"python\", \"/service/server.py\"]\n",
    );
    require(
        digest(baseline.as_bytes()) == ENGINE_DOCKER_HASH,
        "production engine installation bytes changed",
    )?;
    for stage in ["synthetic", "synthetic-featureless"] {
        require(
            docker.contains(&format!("FROM synthetic-base AS {stage}\n")),
            "synthetic stage depends on production engine",
        )?;
    }
    require(
        docker.contains("FROM python:3.12-slim AS synthetic-base\n"),
        "synthetic base changed",
    )?;
    for (key, size) in [("NOEVIA_RS_REF", 40), ("NOEVIA_RS_SHA256", 64)] {
        let value = docker
            .lines()
            .find_map(|line| line.strip_prefix(&format!("ARG {key}=")))
            .ok_or("missing source pin")?;
        require(
            value.len() == size
                && value
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "invalid source pin",
        )?;
    }
    Ok(())
}
fn prepare(directory: &Path) -> Result<()> {
    fs::create_dir_all(directory.join("modules"))?;
    fs::create_dir_all(directory.join("model"))?;
    fs::write(directory.join("modules/torch.py"), TORCH)?;
    fs::write(directory.join("modules/laya.py"), ENGINE)?;
    fs::write(directory.join("model/synthetic-marker"), "synthetic-only\n")?;
    Ok(())
}
// Each reader allocates at most64KiB+1. Closed pipes and process-group cleanup
// prevent a timed-out tool from retaining readers or writing unbounded files.
struct OwnedChild {
    child: std::process::Child,
    complete: bool,
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if !self.complete {
            terminate(&mut self.child);
        }
    }
}
fn command(command: &mut Command, duration: Duration) -> Result<(bool, String)> {
    use std::os::unix::process::CommandExt;
    command
        .process_group(0)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut owned = OwnedChild {
        child: command.spawn()?,
        complete: false,
    };
    let stdout = owned.child.stdout.take().ok_or("missing owned stdout")?;
    let stderr = owned.child.stderr.take().ok_or("missing owned stderr")?;
    let (send, receive) = std::sync::mpsc::channel();
    let first = send.clone();
    let out = thread::Builder::new().spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout
            .take(OUTPUT_CAP + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes);
        let _ = first.send(result);
    })?;
    let err = thread::Builder::new().spawn(move || {
        let mut bytes = Vec::new();
        let result = stderr
            .take(OUTPUT_CAP + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes);
        let _ = send.send(result);
    })?;
    let end = Instant::now() + duration;
    let mut collected = 0;
    let mut bytes = Vec::new();
    let mut status = None;
    let result = (|| -> Result<(bool, String)> {
        loop {
            match receive.recv_timeout(Duration::from_millis(10)) {
                Ok(result) => {
                    let part = result?;
                    require(
                        bytes.len() + part.len() <= OUTPUT_CAP as usize,
                        "synthetic command output cap exceeded",
                    )?;
                    bytes.extend_from_slice(&part);
                    collected += 1;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    thread::sleep(Duration::from_millis(10))
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => (),
            }
            if status.is_none() {
                status = owned.child.try_wait()?;
            }
            if collected == 2 {
                if let Some(status) = status {
                    owned.complete = true;
                    return Ok((
                        status.success(),
                        String::from_utf8_lossy(&bytes).into_owned(),
                    ));
                }
            }
            require(Instant::now() < end, "synthetic command deadline exceeded")?;
        }
    })();
    if result.is_err() {
        terminate(&mut owned.child);
        owned.complete = true;
    }
    out.join().map_err(|_| "owned stdout reader failed")?;
    err.join().map_err(|_| "owned stderr reader failed")?;
    result
}
fn terminate(child: &mut std::process::Child) {
    if let Ok(pid) = i32::try_from(child.id()) {
        let _ = nix::sys::signal::killpg(
            nix::unistd::Pid::from_raw(pid),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
    let _ = child.kill();
    let _ = child.wait();
}
fn docker(args: &[&str]) -> Result<String> {
    let (success, output) = command(Command::new("docker").args(args), Duration::from_secs(8))?;
    require(success, &format!("owned Docker operation failed: {output}"))?;
    Ok(output)
}
#[derive(Debug)]
struct Response {
    status: u16,
    retry: Option<String>,
    body: Value,
}
fn body(state: &str) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(
        &json!({"state":state,"question":"Next?","options":[{"id":"continue","label":"Continue"},{"id":"verify","label":"Verify"}]}),
    )?)
}
fn exchange(method: &str, data: &[u8]) -> Result<Response> {
    require(data.len() <= CAP, "probe request cap")?;
    let mut stream =
        TcpStream::connect_timeout(&"127.0.0.1:8040".parse()?, Duration::from_secs(1))?;
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    stream.set_write_timeout(Some(Duration::from_secs(1)))?;
    let path = if method == "GET" {
        "/health"
    } else {
        "/v1/decisions"
    };
    stream.write_all(
        format!(
            "{method} {path} HTTP/1.0\r\nContent-Length: {}\r\n\r\n",
            data.len()
        )
        .as_bytes(),
    )?;
    stream.write_all(data)?;
    let mut bytes = Vec::new();
    let end = Instant::now() + Duration::from_secs(3);
    let mut chunk = [0u8; 1024];
    loop {
        let remaining = end
            .checked_duration_since(Instant::now())
            .ok_or("probe HTTP deadline exceeded")?;
        stream.set_read_timeout(Some(remaining))?;
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        require(bytes.len() + count <= 2 * CAP, "probe response cap")?;
        bytes.extend_from_slice(chunk.get(..count).ok_or("invalid socket read count")?);
    }
    parse_response(&bytes)
}
fn parse_response(bytes: &[u8]) -> Result<Response> {
    require(bytes.len() <= 2 * CAP, "probe response cap")?;
    let text = std::str::from_utf8(bytes)?;
    let (header, body) = text
        .split_once("\r\n\r\n")
        .ok_or("missing HTTP header terminator")?;
    require(header.len() <= CAP, "probe header cap")?;
    require(body.len() <= CAP, "probe body cap")?;
    let status = header
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or("missing HTTP status")?
        .parse()?;
    let mut retry = None;
    let mut length = None;
    for line in header.lines().skip(1) {
        if let Some((key, value)) = line.split_once(':') {
            if key.eq_ignore_ascii_case("retry-after") {
                retry = Some(value.trim().into());
            }
            if key.eq_ignore_ascii_case("content-length") {
                length = Some(value.trim().parse::<usize>()?);
            }
        }
    }
    require(length == Some(body.len()), "HTTP content length mismatch")?;
    Ok(Response {
        status,
        retry,
        body: serde_json::from_str(body)?,
    })
}
fn healthy() -> Result<()> {
    let response = exchange("GET", &[])?;
    require(
        response.status == 200 && response.body == json!({"ready":true}),
        "synthetic owner not ready",
    )
}
fn contract() -> Result<()> {
    healthy()?;
    let valid = exchange("POST", &body("ok")?)?;
    require(
        valid.status == 200
            && valid.body.get("selected") == Some(&json!("continue"))
            && valid.body.get("calibrated") == Some(&json!(false)),
        "typed synthetic decision failed",
    )?;
    require(
        exchange("POST", b"{}")?.status == 422,
        "invalid request accepted",
    )?;
    let decision = thread::spawn(|| exchange("POST", &body("__sleep:0.8")?));
    let started = Instant::now() + Duration::from_secs(1);
    while !fs::read_to_string("/tmp/laya-synthetic/events")?.contains("__sleep:0.8") {
        require(
            Instant::now() < started,
            "synthetic decision dispatch marker missing",
        )?;
        thread::sleep(Duration::from_millis(5));
    }
    healthy()?;
    let busy = exchange("POST", &body("busy-projection")?)?;
    require(
        busy.status == 503 && busy.retry.as_deref() == Some("1"),
        "busy admission contract failed",
    )?;
    require(
        decision
            .join()
            .map_err(|_| "synthetic decision thread failed")??
            .status
            == 503,
        "decision deadline not enforced",
    )?;
    // Late verify answer must be discarded without sending behind it.
    require(
        exchange("POST", &body("__sleep:1.5")?)?.status == 503,
        "late synthetic decision failed",
    )?;
    let blocked = exchange("POST", &body("blocked-projection")?)?;
    require(
        blocked.status == 503,
        "request dispatched behind unfinished late work",
    )?;
    thread::sleep(Duration::from_millis(650));
    require(
        exchange("POST", &body("after-late")?)?.body.get("selected") == Some(&json!("continue")),
        "stale answer leaked",
    )?;
    let events = fs::read_to_string("/tmp/laya-synthetic/events")?;
    require(
        !events.contains("blocked-projection") && !events.contains("busy-projection"),
        "abandoned/busy request executed",
    )?;
    // A disconnect releases admission but retains the pending engine frame.
    let mut socket = TcpStream::connect("127.0.0.1:8040")?;
    let data = body("__sleep:0.8")?;
    socket.write_all(
        format!(
            "POST /v1/decisions HTTP/1.0\r\nContent-Length: {}\r\n\r\n",
            data.len()
        )
        .as_bytes(),
    )?;
    socket.write_all(&data)?;
    thread::sleep(Duration::from_millis(50));
    drop(socket);
    thread::sleep(Duration::from_millis(850));
    require(
        exchange("POST", &body("after-disconnect")?)?
            .body
            .get("selected")
            == Some(&json!("continue")),
        "disconnect damaged frame/lease",
    )?;
    require(
        exchange("POST", &body("__crash")?)?.status == 503,
        "crash did not refuse decision",
    )?;
    let end = Instant::now() + Duration::from_secs(3);
    while healthy().is_err() {
        require(Instant::now() < end, "replacement deadline exceeded")?;
        thread::sleep(Duration::from_millis(20));
    }
    require(
        exchange("POST", &body("after-recovery")?)?
            .body
            .get("selected")
            == Some(&json!("continue")),
        "replacement not healthy",
    )?;
    require(
        exchange("POST", &body("__child")?)?.status == 200,
        "synthetic grandchild creation failed",
    )?;
    Ok(())
}
fn refusal(featureless: bool) -> Result<()> {
    use std::os::unix::{ffi::OsStringExt, fs::PermissionsExt};
    let directory = tempfile::tempdir()?;
    let marker = directory.path().join("interpreter-marker");
    let python = directory.path().join("python");
    fs::write(
        &python,
        format!(
            "#!/bin/sh\nprintf '%s' \"$*\" > '{}'\nexit 17\n",
            marker.display()
        ),
    )?;
    fs::set_permissions(&python, fs::Permissions::from_mode(0o700))?;
    for selector in [
        std::ffi::OsString::from("unknown"),
        std::ffi::OsString::from_vec(vec![0xff]),
    ] {
        let (success, output) = command(
            Command::new("/usr/local/bin/noevia-laya")
                .env("NOEVIA_LAYA_IMPL", selector)
                .env("PATH", directory.path()),
            Duration::from_secs(2),
        )?;
        require(
            !success && output.contains("NOEVIA_LAYA_IMPL must be"),
            "invalid selector silently fell back",
        )?;
        require(!marker.exists(), "invalid selector spawned interpreter")?;
    }
    if featureless {
        let (success, output) = command(
            Command::new("/usr/local/bin/noevia-laya")
                .env("NOEVIA_LAYA_IMPL", "rust")
                .env("PATH", directory.path()),
            Duration::from_secs(2),
        )?;
        require(
            !success && output.contains("native-laya feature required"),
            "feature omission did not refuse before startup",
        )?;
        require(!marker.exists(), "featureless native spawned interpreter")?;
    }
    for selector in [None, Some("python")] {
        let mut owner = Command::new("/usr/local/bin/noevia-laya");
        owner
            .env_remove("NOEVIA_LAYA_IMPL")
            .env("PATH", directory.path());
        if let Some(selector) = selector {
            owner.env("NOEVIA_LAYA_IMPL", selector);
        }
        let (success, _) = command(&mut owner, Duration::from_secs(2))?;
        require(
            !success && fs::read_to_string(&marker)? == "/service/server.py",
            "legacy argv changed",
        )?;
        fs::remove_file(&marker)?;
    }
    require(
        exchange("GET", &[]).is_err(),
        "refused owner unexpectedly opened listener",
    )?;
    Ok(())
}
struct Container {
    name: String,
}
fn container_name(directory: &Path, suffix: &str) -> Result<String> {
    let nonce = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("owned temporary nonce missing")?;
    Ok(format!(
        "laya-synthetic-{}-{nonce}-{suffix}",
        std::process::id()
    ))
}
impl Drop for Container {
    fn drop(&mut self) {
        // A failed name collision must not remove another container. Only our
        // random owned name and exact synthetic label authorize removal.
        let label = command(
            Command::new("docker").args([
                "inspect",
                "--format",
                "{{index .Config.Labels \"noevia.synthetic.owner\"}}",
                &self.name,
            ]),
            Duration::from_secs(8),
        );
        if label.is_ok_and(|(success, value)| success && value.trim() == self.name) {
            let _ = command(
                Command::new("docker").args(["rm", "-f", &self.name]),
                Duration::from_secs(8),
            );
        }
    }
}
impl Container {
    fn start(
        image: &str,
        directory: &Path,
        suffix: &str,
        selector: Option<&str>,
        startup: &str,
    ) -> Result<Self> {
        let name = container_name(directory, suffix)?;
        let mut owned = Self { name };
        let label = format!("noevia.synthetic.owner={}", owned.name);
        let modules = format!("{}:/synthetic:ro", directory.join("modules").display());
        let model = format!("{}:/model:ro", directory.join("model").display());
        let mut args = vec![
            "run",
            "-d",
            "--name",
            &owned.name,
            "--label",
            &label,
            "--network",
            "none",
            "--read-only",
            "--user",
            "1000:1000",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "32",
            "--tmpfs",
            "/tmp:rw,size=16777216,mode=1777",
            "-v",
            &modules,
            "-v",
            &model,
            "-e",
            "PYTHONPATH=/synthetic",
            "-e",
            "LAYA_QUEUE_TIMEOUT_MS=20",
            "-e",
            "LAYA_DECISION_TIMEOUT_S=0.5",
            "-e",
            "LAYA_KEEP_WARM_S=0",
        ];
        let selected = selector.map(|value| format!("NOEVIA_LAYA_IMPL={value}"));
        if let Some(selected) = &selected {
            args.extend(["-e", selected]);
        }
        if startup == "delay" {
            args.extend(["-e", "LAYA_SYNTHETIC_STARTUP_SLEEP=2"]);
        }
        if startup == "fail" {
            args.extend(["-e", "LAYA_SYNTHETIC_FAIL=1"]);
        }
        args.push(image);
        docker(&args)?;
        owned.restrictions()?;
        // This binding owns only the newly named synthetic container.
        owned.name.shrink_to_fit();
        Ok(owned)
    }
    fn restrictions(&self) -> Result<()> {
        let config: Value = serde_json::from_str(&docker(&[
            "inspect",
            "--format",
            "{{json .HostConfig}}",
            &self.name,
        ])?)?;
        require(
            config.get("ReadonlyRootfs") == Some(&json!(true)),
            "container root is writable",
        )?;
        require(
            config.get("NetworkMode") == Some(&json!("none")),
            "container network is enabled",
        )?;
        require(
            config.get("Memory") == Some(&json!(268435456)),
            "container memory bound missing",
        )?;
        require(
            config.get("NanoCpus") == Some(&json!(1000000000)),
            "container CPU bound missing",
        )?;
        require(
            config.get("PidsLimit") == Some(&json!(32)),
            "container PID bound missing",
        )?;
        require(
            config.get("Tmpfs").and_then(|value| value.get("/tmp"))
                == Some(&json!("rw,size=16777216,mode=1777")),
            "container tmpfs bound missing",
        )?;
        require(
            config.get("CapDrop") == Some(&json!(["ALL"])),
            "container capabilities retained",
        )?;
        let security = config
            .get("SecurityOpt")
            .and_then(Value::as_array)
            .ok_or("security options missing")?;
        require(
            security.iter().any(|v| {
                v.as_str()
                    .is_some_and(|s| s.starts_with("no-new-privileges"))
            }),
            "no-new-privileges missing",
        )?;
        require(
            config
                .get("PortBindings")
                .is_none_or(|v| v.is_null() || v.as_object().is_some_and(|map| map.is_empty())),
            "container exposes host port",
        )?;
        require(
            docker(&["inspect", "--format", "{{.Config.User}}", &self.name])?.trim() == "1000:1000",
            "container runs as root",
        )?;
        let mounts: Value = serde_json::from_str(&docker(&[
            "inspect",
            "--format",
            "{{json .Mounts}}",
            &self.name,
        ])?)?;
        for destination in ["/model", "/synthetic"] {
            let mount = mounts
                .as_array()
                .and_then(|items| {
                    items
                        .iter()
                        .find(|item| item.get("Destination") == Some(&json!(destination)))
                })
                .ok_or("synthetic mount missing")?;
            require(
                mount.get("RW") == Some(&json!(false)),
                "synthetic fixture mount writable",
            )?;
        }
        Ok(())
    }
    fn ready(&self) -> Result<()> {
        let end = Instant::now() + Duration::from_secs(10);
        loop {
            let (success, _) = command(
                Command::new("docker").args(["exec", &self.name, "laya-image-probe", "health"]),
                Duration::from_secs(4),
            )?;
            if success {
                return Ok(());
            }
            require(
                Instant::now() < end,
                "synthetic image startup deadline exceeded",
            )?;
            thread::sleep(Duration::from_millis(20));
        }
    }
    fn stop(&self, native: bool) -> Result<()> {
        docker(&["stop", "--time", "4", &self.name])?;
        let state: Value = serde_json::from_str(&docker(&[
            "inspect",
            "--format",
            "{{json .State}}",
            &self.name,
        ])?)?;
        require(
            state.get("Running") == Some(&json!(false)) && state.get("Pid") == Some(&json!(0)),
            "owned container process tree survived stop",
        )?;
        if native {
            require(
                state.get("ExitCode") == Some(&json!(0)),
                "native SIGTERM failed to complete owned cleanup",
            )?;
        }
        Ok(())
    }
}
fn run_images(image: &str, featureless: &str, source: &Path) -> Result<()> {
    require(
        image == "noevia-laya-synthetic:ci" && featureless == "noevia-laya-featureless:ci",
        "qualification accepts only owned synthetic CI image tags",
    )?;
    guard(source)?;
    let directory = tempfile::tempdir()?;
    prepare(directory.path())?;
    for (suffix, selector) in [
        ("default", None),
        ("python", Some("python")),
        ("native", Some("rust")),
    ] {
        let owner = Container::start(image, directory.path(), suffix, selector, "")?;
        owner.ready()?;
        // Namespace-local Rust probe: no exposed port or host-network access.
        docker(&["exec", &owner.name, "laya-image-probe", "contract"])?;
        let argv = docker(&["exec", &owner.name, "laya-image-probe", "owner"])?;
        let expected = if selector == Some("rust") {
            "noevia-laya"
        } else {
            "python /service/server.py"
        };
        require(
            argv.trim() == expected,
            "default/native PID1 owner mismatch",
        )?;
        owner.stop(selector == Some("rust"))?;
    }
    for (suffix, selected_image, mode) in [
        ("selector", image, "native"),
        ("featureless", featureless, "featureless"),
    ] {
        let name = container_name(directory.path(), suffix)?;
        let owner = Container { name };
        let label = format!("noevia.synthetic.owner={}", owner.name);
        let mut args = vec![
            "run",
            "--name",
            &owner.name,
            "--label",
            &label,
            "--network",
            "none",
            "--read-only",
            "--user",
            "1000:1000",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "32",
            "--tmpfs",
            "/tmp:rw,size=16777216,mode=1777",
            "--entrypoint",
            "laya-image-probe",
            selected_image,
            "refusal",
            mode,
        ];
        // No worker/model mount needed: every refusal occurs before engine start.
        docker(&args)?;
        args.clear();
    }
    let failed = Container::start(image, directory.path(), "failed", Some("rust"), "fail")?;
    let exit = docker(&["wait", &failed.name])?;
    require(
        exit.trim().parse::<i32>()? != 0,
        "startup failure silently fell back",
    )?;
    let startup = Container::start(image, directory.path(), "startup", Some("rust"), "delay")?;
    let end = Instant::now() + Duration::from_secs(5);
    loop {
        let (success, _) = command(
            Command::new("docker").args(["exec", &startup.name, "laya-image-probe", "started"]),
            Duration::from_secs(3),
        )?;
        if success {
            break;
        }
        require(Instant::now() < end, "synthetic startup marker deadline")?;
        thread::sleep(Duration::from_millis(20));
    }
    startup.stop(true)?;
    println!("synthetic image qualification PASS: default/explicit-python/native, typed HTTP, bounds, late/cancel/recovery, refusal/omission, startup/ready SIGTERM; no actual ML");
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("guard") if args.len() == 2 => guard(Path::new(args.get(1).ok_or("source path missing")?)),
        Some("images") if args.len() == 4 => run_images(args.get(1).ok_or("image missing")?, args.get(2).ok_or("featureless image missing")?, Path::new(args.get(3).ok_or("source path missing")?)),
        Some("health") if args.len() == 1 => healthy(),
        Some("contract") if args.len() == 1 => contract(),
        Some("refusal") if args.len() == 2 => refusal(args.get(1).is_some_and(|mode| mode == "featureless")),
        Some("owner") if args.len() == 1 => {
            let bytes = fs::read("/proc/1/cmdline")?;
            let args: Vec<_> = bytes.split(|b| *b == 0).filter(|part| !part.is_empty()).map(String::from_utf8_lossy).collect();
            println!("{}", args.join(" ")); Ok(())
        }
        Some("started") if args.len() == 1 => require(!fs::read_to_string("/tmp/laya-synthetic/starts")?.is_empty(), "synthetic worker not started"),
        _ => Err("usage: laya-image-probe guard SOURCE | images IMAGE FEATURELESS SOURCE | health | contract | refusal native|featureless | owner | started".into()),
    }
}
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::*;
    #[test]
    fn ci_installs_required_rust_components() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let workflow = fs::read_to_string(source.join(".github/workflows/ci.yml")).unwrap();
        let job = workflow
            .split_once("  laya-image:\n")
            .unwrap()
            .1
            .split_once("\n  model-manager:\n")
            .unwrap()
            .0;
        let required = "      - uses: dtolnay/rust-toolchain@70dec6087f2cb6bab9fbac4da23160afbe848e91 # 1.99.0\n        with:\n          components: rustfmt, clippy";
        assert!(job.contains(required));
        for replacement in ["components: clippy", "components: rustfmt"] {
            let mutant = job.replace("components: rustfmt, clippy", replacement);
            assert_ne!(mutant, job);
            assert!(
                !mutant.contains(required),
                "missing component mutation survived"
            );
        }
    }
    #[test]
    fn cargo_target_directory_uses_supported_step_context() {
        fn valid(workflow: &str) -> bool {
            let Some((_, rest)) = workflow.split_once("  laya-image:\n") else {
                return false;
            };
            let Some((job, _)) = rest.split_once("\n  model-manager:\n") else {
                return false;
            };
            let Some((header, _)) = job.split_once("    steps:\n") else {
                return false;
            };
            if header.contains("runner.temp") || header.contains("CARGO_TARGET_DIR") {
                return false;
            }
            let environment = "        env:\n          CARGO_TARGET_DIR: ${{ runner.temp }}/laya-image-target\n        run:";
            job.matches(environment).count() == 2
                && [
                    "Rust-owned source guard and bounded probe controls",
                    "Nonroot read-only network-none image qualification, synthetic ML only",
                ]
                .iter()
                .all(|name| job.contains(&format!("      - name: {name}\n{environment}")))
        }
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let workflow = fs::read_to_string(source.join(".github/workflows/ci.yml")).unwrap();
        assert!(
            valid(&workflow),
            "Cargo target requires supported step environment"
        );
        let mutant = workflow.replace(
            "  laya-image:\n    name: Laya synthetic native image contract\n    runs-on: ubuntu-latest\n    steps:",
            "  laya-image:\n    name: Laya synthetic native image contract\n    runs-on: ubuntu-latest\n    env:\n      CARGO_TARGET_DIR: ${{ runner.temp }}/laya-image-target\n    steps:",
        );
        assert_ne!(mutant, workflow, "job-level mutation must apply");
        assert!(
            !valid(&mutant),
            "unsupported runner context mutation survived"
        );
    }
    #[test]
    fn production_image_names_are_refused_before_docker_or_source_access() {
        assert!(run_images(
            "cowork-laya:production",
            "noevia-laya-featureless:ci",
            Path::new("/missing-source")
        )
        .unwrap_err()
        .to_string()
        .contains("only owned synthetic"));
        assert!(run_images(
            "noevia-laya-synthetic:ci",
            "cowork-laya:production",
            Path::new("/missing-source")
        )
        .unwrap_err()
        .to_string()
        .contains("only owned synthetic"));
    }
    #[test]
    fn bounded_http_decoder_refuses_length_shape_and_size_mutations() {
        let valid = b"HTTP/1.0 200 OK\r\nContent-Length: 14\r\n\r\n{\"ready\":true}";
        let response = parse_response(valid).unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body, json!({"ready":true}));
        for invalid in [
            b"HTTP/1.0 200 OK\r\nContent-Length: 13\r\n\r\n{\"ready\":true}".as_slice(),
            b"HTTP/1.0 200 OK\r\n\r\n{}".as_slice(),
            b"HTTP/1.0 200 OK\r\nContent-Length: 1\r\n\r\nx".as_slice(),
            b"HTTP/1.0\r\nContent-Length: 2\r\n\r\n{}".as_slice(),
        ] {
            assert!(parse_response(invalid).is_err());
        }
        assert!(parse_response(&vec![b'a'; 2 * CAP + 1]).is_err());
        assert!(
            exchange("POST", &vec![b'a'; CAP + 1]).is_err(),
            "oversized input reached socket"
        );
    }
    #[test]
    fn actual_worker_and_engine_instruction_guard_and_mutations() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        guard(source).unwrap();
        let directory = tempfile::tempdir().unwrap();
        fs::copy(source.join("server.py"), directory.path().join("server.py")).unwrap();
        let original = fs::read_to_string(source.join("Dockerfile")).unwrap();
        for mutant in [
            original.replace("torch==2.14.0", "torch==0.0.0"),
            original.replace("ARG RUNTIME_IMAGE=python:3.12-slim", "# missing global ARG"),
            original.replace(
                "FROM synthetic-base AS synthetic\n",
                "FROM production AS synthetic\n",
            ),
            original.replace("ARG REUSE_CPU_RUNTIME=false", "ARG REUSE_CPU_RUNTIME=true"),
        ] {
            fs::write(directory.path().join("Dockerfile"), mutant).unwrap();
            assert!(
                guard(directory.path()).is_err(),
                "instruction mutation survived guard"
            );
        }
        fs::write(directory.path().join("Dockerfile"), original).unwrap();
        fs::write(directory.path().join("server.py"), b"not actual source").unwrap();
        assert!(guard(directory.path()).is_err());
    }
    #[test]
    fn fixture_generation_is_fixed_synthetic_and_separate_from_model_assets() {
        let directory = tempfile::tempdir().unwrap();
        prepare(directory.path()).unwrap();
        assert_eq!(
            fs::read_to_string(directory.path().join("model/synthetic-marker")).unwrap(),
            "synthetic-only\n"
        );
        assert_eq!(
            fs::read_dir(directory.path().join("model"))
                .unwrap()
                .count(),
            1
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("modules/laya.py")).unwrap(),
            ENGINE
        );
        assert!(ENGINE.contains("('/model','cpu')"));
        assert!(!ENGINE.contains("import torch"));
    }
    #[test]
    fn owned_command_timeout_and_output_cap_are_enforced() {
        assert!(command(Command::new("sleep").arg("1"), Duration::from_millis(20)).is_err());
        assert!(command(Command::new("yes").arg("synthetic"), Duration::from_secs(1)).is_err());
        assert_eq!(
            command(
                Command::new("sh").args(["-c", "printf synthetic"]),
                Duration::from_secs(1)
            )
            .unwrap(),
            (true, "synthetic".into())
        );
    }
}
