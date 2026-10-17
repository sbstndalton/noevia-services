#![forbid(unsafe_code)]
use std::{
    error::Error,
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TOKEN: &str = "synthetic-image-token-0000000000000000000";
const ROOT: &str = "/tmp/sections-qualification";
const INI: &str = "version = 1\n[tiny]\nctx-size = 4096\n";
// Fixed existing-app seam: import the actual app, suppress all startup hooks and
// hardware/engine discovery, record ownership, then run the actual ASGI app.
// Rust owns fixtures, requests, assertions, deadlines and container cleanup.
const SHIM: &str = r#"#!/usr/local/bin/python
import json,os,sys
from pathlib import Path
from app import main,services
main.app.router.on_startup.clear()
services._docker_client=lambda:None
Path('/tmp/sections-qualification/child.json').write_text(json.dumps({'sections':os.environ.get('NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED'),'host':os.environ.get('NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED'),'native':os.environ.get('NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED'),'argv':sys.argv[1:]}))
import uvicorn
uvicorn.run(main.app,host=sys.argv[3],port=int(sys.argv[5]),log_level='error')
"#;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
struct OwnedProcess(std::process::Child);
impl Drop for OwnedProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn command(args: &[&str]) -> Result<bool> {
    command_named("docker", args)
}
fn command_named(program: &str, args: &[&str]) -> Result<bool> {
    let mut child = OwnedProcess(
        Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?,
    );
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.0.try_wait()? {
            return Ok(status.success());
        }
        if Instant::now() >= until {
            return Err("Docker command deadline".into());
        }
        thread::sleep(Duration::from_millis(20));
    }
}
fn omission() -> Result<()> {
    refusal(
        "/usr/local/bin/model-manager-sections-omission",
        "rust",
        "rust",
        "native-sections requires rust model manager and compiled native-sections feature",
    )
}
fn refusal(binary: &str, manager: &str, sections: &str, expected: &str) -> Result<()> {
    prepare()?;
    let diagnostic = format!("{ROOT}/omission.stderr");
    let mut child = OwnedProcess(
        Command::new(binary)
            .env("MODEL_MANAGER_IMPL", manager)
            .env("MODEL_MANAGER_SECTIONS_IMPL", sections)
            .env(
                "PATH",
                "/tmp/sections-qualification/bin:/usr/local/bin:/usr/bin:/bin",
            )
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(fs::File::create(&diagnostic)?)
            .spawn()?,
    );
    let until = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.0.try_wait()? {
            break status;
        }
        if Instant::now() >= until || fs::metadata(&diagnostic)?.len() > 65_536 {
            return Err("omission deadline or diagnostic cap".into());
        }
        thread::sleep(Duration::from_millis(20));
    };
    let mut bytes = Vec::new();
    fs::File::open(diagnostic)?
        .take(65_537)
        .read_to_end(&mut bytes)?;
    require(bytes.len() <= 65_536, "omission diagnostic cap")?;
    exact_refusal(
        status.code(),
        &String::from_utf8(bytes)?,
        expected,
        Path::new(&format!("{ROOT}/child.json")).exists(),
    )
}
fn exact_refusal(
    status: Option<i32>,
    diagnostic: &str,
    expected: &str,
    child_started: bool,
) -> Result<()> {
    require(!child_started, "refusal spawned actual app")?;
    require(status == Some(1), "expected application rejection status 1")?;
    require(
        diagnostic == format!("Error: {expected:?}\n"),
        "exact application rejection diagnostic",
    )
}
const OWNER_LABEL: &str = "noevia.sections-qualification.owner";
fn nonce() -> Result<String> {
    let mut bytes = [0u8; 32];
    fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}
// Capture to private files rather than pipes: a verbose or stalled child cannot
// block the parent, and both output size and elapsed time are bounded.
fn capture(program: &str, args: &[&str], deadline: Duration) -> Result<(Option<i32>, String)> {
    let dir = std::env::temp_dir().join(format!("sections-command-{}", nonce()?));
    fs::create_dir(&dir)?;
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
    let result = (|| {
        let output = dir.join("output");
        let file = fs::File::create(&output)?;
        let mut child = OwnedProcess(
            Command::new(program)
                .args(args)
                .stdin(Stdio::null())
                .stdout(file.try_clone()?)
                .stderr(file)
                .spawn()?,
        );
        let until = Instant::now() + deadline;
        loop {
            require(fs::metadata(&output)?.len() <= 65_536, "command output cap")?;
            require(Instant::now() < until, "command deadline")?;
            if let Some(status) = child.0.try_wait()? {
                let mut bytes = Vec::new();
                fs::File::open(&output)?
                    .take(65_537)
                    .read_to_end(&mut bytes)?;
                require(bytes.len() <= 65_536, "command output cap")?;
                return Ok((status.code(), String::from_utf8(bytes)?));
            }
            thread::sleep(Duration::from_millis(20));
        }
    })();
    let _ = fs::remove_dir_all(dir);
    result
}
struct Container {
    id: String,
    token: String,
    program: String,
}
impl Container {
    fn acquire(args: &[&str]) -> Result<Self> {
        Self::acquire_with("docker", args, Duration::from_secs(30))
    }
    fn acquire_with(program: &str, args: &[&str], deadline: Duration) -> Result<Self> {
        let token = nonce()?;
        let label = format!("{OWNER_LABEL}={token}");
        let mut create = vec!["create", "--label", &label];
        create.extend_from_slice(args);
        let (status, output) = capture(program, &create, deadline)?;
        require(status == Some(0), "owned container acquisition failed")?;
        let id = output.trim();
        require(
            id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit()),
            "immutable container ID",
        )?;
        let owned = Self {
            id: id.to_owned(),
            token,
            program: program.to_owned(),
        };
        require(owned.verified()?, "container ownership label mismatch")?;
        Ok(owned)
    }
    fn verified(&self) -> Result<bool> {
        let format = format!("{{{{.Id}}}} {{{{index .Config.Labels \"{OWNER_LABEL}\"}}}}");
        let (status, output) = capture(
            &self.program,
            &["inspect", "--format", &format, &self.id],
            Duration::from_secs(30),
        )?;
        Ok(status == Some(0) && output.trim() == format!("{} {}", self.id, self.token))
    }
}
impl Drop for Container {
    fn drop(&mut self) {
        if self.verified().unwrap_or(false) {
            let _ = command_named(&self.program, &["rm", "-f", &self.id]);
        }
    }
}
fn request(port: u16, method: &str, path: &str, authorized: bool) -> Result<(u16, String)> {
    let mut stream = TcpStream::connect_timeout(
        &SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(250),
    )?;
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    stream.set_write_timeout(Some(Duration::from_secs(3)))?;
    let auth = if authorized {
        format!("X-Model-Loader-Token: {TOKEN}\r\n")
    } else {
        String::new()
    };
    write!(stream,"{method} {path} HTTP/1.0\r\nHost: localhost\r\n{auth}Connection: close\r\nContent-Length: 0\r\n\r\n")?;
    let until = Instant::now() + Duration::from_secs(3);
    let mut bytes = Vec::new();
    loop {
        let remaining = until
            .checked_duration_since(Instant::now())
            .ok_or("HTTP response deadline")?;
        stream.set_read_timeout(Some(remaining))?;
        let mut chunk = [0u8; 8192];
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        require(bytes.len() + count <= 8_388_608, "HTTP response cap")?;
        bytes.extend_from_slice(chunk.get(..count).ok_or("HTTP read length")?);
    }
    let text = String::from_utf8(bytes)?;
    let (headers, body) = text.split_once("\r\n\r\n").ok_or("HTTP framing")?;
    let status = headers
        .split_whitespace()
        .nth(1)
        .ok_or("HTTP status")?
        .parse()?;
    Ok((status, body.to_owned()))
}
fn check(
    port: u16,
    method: &str,
    path: &str,
    authorized: bool,
    status: u16,
    body: Option<&str>,
) -> Result<String> {
    let (actual, text) = request(port, method, path, authorized)
        .map_err(|_| format!("HTTP {method} {path}: transport/framing/budget failure"))?;
    require(
        actual == status,
        &format!("HTTP {method} {path}: expected status {status}, actual status {actual}"),
    )?;
    if let Some(expected) = body {
        require(
            text == expected,
            &format!("HTTP {method} {path}: body assertion mismatch"),
        )?;
    }
    Ok(text)
}
fn safe_probe_diagnostic(reason: &str) -> bool {
    if [
        "owned image readiness deadline",
        "child marker cap",
        "selected child confirmation",
        "default clears stale confirmations",
        "actual collection fixture",
        "actual named fixture",
        "actual defaults branch",
        "read and rejected writes preserve INI",
        "startup hooks stayed suppressed",
    ]
    .contains(&reason)
    {
        return true;
    }
    let Some((request, assertion)) = reason
        .strip_prefix("HTTP ")
        .and_then(|s| s.split_once(": "))
    else {
        return false;
    };
    let known_request = [
        "GET /api/v1/sections",
        "GET /api/v1/sections/tiny",
        "GET /api/v1/sections/tiny?defaults=true",
        "GET /api/v1/sections/tiny?defaults=bad",
        "GET /api/v1/sections/bad!name",
        "HEAD /api/v1/sections/tiny",
        "PUT /api/v1/sections/tiny",
    ]
    .contains(&request);
    if assertion == "body assertion mismatch" || assertion == "transport/framing/budget failure" {
        return known_request;
    }
    let Some((expected, actual)) = assertion
        .strip_prefix("expected status ")
        .and_then(|s| s.split_once(", actual status "))
    else {
        return false;
    };
    known_request
        && [expected, actual]
            .iter()
            .all(|s| s.len() == 3 && s.bytes().all(|b| b.is_ascii_digit()))
}
fn prepare() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::create_dir_all(format!("{ROOT}/bin"))?;
    fs::create_dir_all(format!("{ROOT}/models"))?;
    fs::create_dir_all(format!("{ROOT}/data"))?;
    fs::write(format!("{ROOT}/models/models.ini"), INI)?;
    fs::write(format!("{ROOT}/bin/uvicorn"), SHIM)?;
    fs::set_permissions(
        format!("{ROOT}/bin/uvicorn"),
        fs::Permissions::from_mode(0o700),
    )?;
    Ok(())
}
fn probe(mode: &str) -> Result<()> {
    // noevia#1071: the native parent always supervises the private child.
    wait_ready(8090, Some(8091), Path::new(&format!("{ROOT}/child.json")))?;
    let marker = fs::read_to_string(format!("{ROOT}/child.json"))?;
    require(marker.len() < 1024, "child marker cap")?;
    probe_ready(mode, &marker)
}
fn wait_ready(public: u16, private: Option<u16>, marker: &Path) -> Result<()> {
    let until = Instant::now() + Duration::from_secs(12);
    loop {
        if request(public, "GET", "/api/v1/sections", true).is_ok()
            && marker.is_file()
            && private.is_none_or(|port| request(port, "GET", "/api/v1/sections", true).is_ok())
        {
            break;
        }
        require(Instant::now() < until, "owned image readiness deadline")?;
        thread::sleep(Duration::from_millis(50));
    }
    Ok(())
}
fn probe_ready(mode: &str, marker: &str) -> Result<()> {
    if mode == "rust" {
        require(
            marker.contains("\"sections\": \"1\"")
                && marker.contains("127.0.0.1")
                && marker.contains("8091"),
            "selected child confirmation",
        )?;
        for path in ["/api/v1/sections", "/api/v1/sections/tiny"] {
            check(
                8091,
                "GET",
                path,
                false,
                401,
                Some("{\"error\":\"unauthorized\"}"),
            )?;
            check(
                8091,
                "GET",
                path,
                true,
                503,
                Some("{\"detail\":\"Native model manager owns this endpoint\"}"),
            )?;
        }
        check(8091, "HEAD", "/api/v1/sections/tiny", true, 405, Some(""))?;
        check(8091, "PUT", "/api/v1/sections/tiny", true, 422, None)?;
    } else {
        // Sections off (noevia#1071: under the always-native parent): stale confirmations
        // are replaced with "0" and health/logs stay native ("1").
        require(
            marker.contains("\"sections\": \"0\"")
                && marker.contains("\"native\": \"1\"")
                && marker.contains("\"host\": \"0\"")
                && marker.contains("127.0.0.1")
                && marker.contains("8091"),
            "default clears stale confirmations",
        )?;
    }
    check(
        8090,
        "GET",
        "/api/v1/sections",
        false,
        401,
        Some("{\"error\":\"unauthorized\"}"),
    )?;
    let collection = check(8090, "GET", "/api/v1/sections", true, 200, None)?;
    require(
        collection.contains("\"name\":\"tiny\"")
            && collection.contains("\"raw\":\"version = 1\\n[tiny]\\nctx-size = 4096\\n\""),
        "actual collection fixture",
    )?;
    let section = check(8090, "GET", "/api/v1/sections/tiny", true, 200, None)?;
    require(
        section.contains("\"ctx-size\":\"4096\"") && section.contains("\"schema\""),
        "actual named fixture",
    )?;
    let defaults = check(
        8090,
        "GET",
        "/api/v1/sections/tiny?defaults=true",
        true,
        200,
        None,
    )?;
    require(
        defaults.contains("\"hints\"") && defaults.contains("\"exists\":true"),
        "actual defaults branch",
    )?;
    check(
        8090,
        "GET",
        "/api/v1/sections/tiny?defaults=bad",
        true,
        422,
        None,
    )?;
    check(
        8090,
        "GET",
        "/api/v1/sections/bad!name",
        true,
        400,
        Some("{\"detail\":\"invalid section name\"}"),
    )?;
    check(8090, "HEAD", "/api/v1/sections/tiny", true, 405, Some(""))?;
    check(8090, "PUT", "/api/v1/sections/tiny", true, 422, None)?;
    if mode == "rust" {
        fs::write(format!("{ROOT}/outside.gguf"), "outside synthetic sentinel")?;
        std::os::unix::fs::symlink(
            format!("{ROOT}/outside.gguf"),
            format!("{ROOT}/models/escape.gguf"),
        )?;
        check(
            8090,
            "GET",
            "/api/v1/sections",
            true,
            503,
            Some("{\"detail\":\"Native section configuration is unavailable\"}"),
        )?;
        fs::remove_file(format!("{ROOT}/models/escape.gguf"))?;
    }
    require(
        fs::read_to_string(format!("{ROOT}/models/models.ini"))? == INI,
        "read and rejected writes preserve INI",
    )?;
    require(
        fs::read_dir(format!("{ROOT}/data"))?.next().is_none(),
        "startup hooks stayed suppressed",
    )?;
    Ok(())
}
fn images(image: &str) -> Result<()> {
    require(
        image == "noevia-model-manager-qualification:ci",
        "expected fixed qualification image",
    )?;
    for mode in ["python", "rust"] {
        let sections = format!("MODEL_MANAGER_SECTIONS_IMPL={mode}");
        let mut args = vec![
            "--network",
            "none",
            "--log-driver",
            "none",
            "--read-only",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--pids-limit",
            "64",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,exec,size=16777216,mode=1777",
            "-e",
            "PATH=/tmp/sections-qualification/bin:/usr/local/bin:/usr/bin:/bin",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTHONPATH=/srv",
            "-e",
            "MODELS_DIR=/tmp/sections-qualification/models",
            "-e",
            "MODELS_INI_PATH=/tmp/sections-qualification/models/models.ini",
            "-e",
            "DATA_DIR=/tmp/sections-qualification/data",
            "-e",
            "DOCKER_HOST=unix:///tmp/absent-synthetic-docker.sock",
            "-e",
            "LLAMA_CONTAINERS=synthetic-only",
            "-e",
            "MIGRATE_CACHE_RAM_ON_START=false",
            "-e",
            "CHAT_MODEL_IDLE_UNLOAD_SECONDS=0",
            "-e",
            "NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED=1",
            "-e",
            "NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED=1",
            "-e",
            "NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED=1",
            "-e",
            "MODEL_LOADER_TOKEN=synthetic-image-token-0000000000000000000",
        ];
        // noevia#1071: the parent is native in both modes; only the sections selector differs.
        args.extend(["-e", "MODEL_MANAGER_IMPL=rust", "-e", &sections]);
        args.extend(["--entrypoint", "sections-image-probe", image, "serve"]);
        let owned = Container::acquire(&args)?;
        require(command(&["start", &owned.id])?, "owned image start")?;
        let (status, diagnostic) = capture(
            "docker",
            &["exec", &owned.id, "sections-image-probe", "probe", mode],
            Duration::from_secs(30),
        ).map_err(|_| format!("actual image HTTP qualification mode={mode} stage=exec: bounded capture/launch failed"))?;
        if status != Some(0) {
            // Never forward Docker/application output. Only our own allowlisted
            // fixed probe errors can be returned; bodies and environment are absent.
            let reason = diagnostic
                .lines()
                .find_map(|line| line.strip_prefix("SECTIONS_PROBE_FAILURE "))
                .filter(|reason| safe_probe_diagnostic(reason));
            return Err(format!(
                "actual image HTTP qualification mode={mode} status={status:?}: {}",
                reason.unwrap_or("probe/transport failed without safe diagnostic")
            )
            .into());
        }
        drop(owned);
    }
    for mode in ["omission", "refuse-invalid"] {
        let owned = Container::acquire(&[
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "128m",
            "--pids-limit",
            "32",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,exec,size=16777216,mode=1777",
            "--entrypoint",
            "sections-image-probe",
            image,
            mode,
        ])?;
        require(
            command(&["start", "-a", &owned.id])?,
            "actual image refusal qualification",
        )?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or("missing mode")?;
    let value = args.next();
    require(args.next().is_none(), "unexpected arguments")?;
    match (mode.as_str(), value.as_deref()) {
        ("images", Some(image)) => images(image),
        ("probe", Some(mode @ ("python" | "rust"))) => {
            let result = probe(mode);
            if let Err(error) = &result {
                let reason = error.to_string();
                if safe_probe_diagnostic(&reason) {
                    eprintln!("SECTIONS_PROBE_FAILURE {reason}");
                }
            }
            result
        }
        ("omission", None) => omission(),
        ("refuse-invalid", None) => refusal(
            "/usr/local/bin/noevia-model-manager",
            "rust",
            "invalid",
            "MODEL_MANAGER_SECTIONS_IMPL must be python or rust",
        ),
        ("serve", None) => {
            prepare()?;
            use std::os::unix::process::CommandExt;
            Err(Command::new("noevia-model-manager").exec().into())
        }
        _ => Err("invalid mode".into()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn consume_fixed_request(stream: &mut TcpStream) {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut bytes = Vec::new();
        while !bytes.ends_with(b"\r\n\r\n") {
            let mut byte = [0u8; 1];
            stream.read_exact(&mut byte).unwrap();
            bytes.extend(byte);
            assert!(bytes.len() <= 4096);
        }
    }
    #[test]
    fn image_readiness_waits_for_actual_private_child_not_early_marker() {
        use std::net::TcpListener;
        let dir = std::env::temp_dir().join(format!("sections-ready-control-{}", nonce().unwrap()));
        fs::create_dir(&dir).unwrap();
        let marker = dir.join("child.json");
        fs::write(&marker, "synthetic early marker").unwrap();
        let public = TcpListener::bind("127.0.0.1:0").unwrap();
        let public_port = public.local_addr().unwrap().port();
        let private = TcpListener::bind("127.0.0.1:0").unwrap();
        let private_port = private.local_addr().unwrap().port();
        drop(private);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stopped = stop.clone();
        public.set_nonblocking(true).unwrap();
        let server = thread::spawn(move || {
            let until = Instant::now() + Duration::from_secs(5);
            while !stopped.load(std::sync::atomic::Ordering::Relaxed) && Instant::now() < until {
                if let Ok((mut stream, _)) = public.accept() {
                    consume_fixed_request(&mut stream);
                    stream
                        .write_all(b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                        .unwrap();
                } else {
                    thread::sleep(Duration::from_millis(5));
                }
            }
        });
        // The previous public+marker predicate succeeds while private ASGI is absent.
        assert!(request(public_port, "GET", "/api/v1/sections", true).is_ok() && marker.is_file());
        assert!(request(private_port, "GET", "/api/v1/sections", true).is_err());
        let child = thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            let listener = TcpListener::bind(("127.0.0.1", private_port)).unwrap();
            let (mut stream, _) = listener.accept().unwrap();
            consume_fixed_request(&mut stream);
            stream
                .write_all(b"HTTP/1.0 503 Unavailable\r\nContent-Length: 2\r\n\r\n{}")
                .unwrap();
        });
        let start = Instant::now();
        wait_ready(public_port, Some(private_port), &marker).unwrap();
        assert!(start.elapsed() >= Duration::from_millis(150));
        child.join().unwrap();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        server.join().unwrap();
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn image_diagnostics_report_fixed_assertions_without_body_or_environment() {
        assert!(safe_probe_diagnostic(
            "HTTP GET /api/v1/sections: expected status 200, actual status 503"
        ));
        assert!(safe_probe_diagnostic(
            "HTTP PUT /api/v1/sections/tiny: body assertion mismatch"
        ));
        assert!(safe_probe_diagnostic("actual named fixture"));
        for unsafe_text in [
            "body contains token secret",
            "HTTP GET /private-user-path: body assertion mismatch",
            "HTTP GET /api/v1/sections: expected status 200, actual status 503 secret",
            "actual named fixture\nuser body",
        ] {
            assert!(!safe_probe_diagnostic(unsafe_text));
        }
    }
    #[test]
    fn image_python_http_contract_with_actual_startup_suppressed_app() {
        // Uses the same fixed seam, with Rust-owned temporary paths and an
        // ephemeral loopback listener. Never starts the app's startup hooks.
        let dir = std::env::temp_dir().join(format!("sections-app-control-{}", nonce().unwrap()));
        fs::create_dir_all(dir.join("models")).unwrap();
        fs::create_dir(dir.join("data")).unwrap();
        fs::write(dir.join("models/models.ini"), INI).unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let python = std::env::var("SECTIONS_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
        let seam = SHIM.replace(ROOT, dir.to_str().unwrap());
        // Resolve the source before taking parents: CI's qualification/main.rs
        // has an empty lexical grandparent, but an absolute app root exists.
        let source = Path::new(file!()).canonicalize().unwrap();
        let app_root = source.parent().unwrap().parent().unwrap();
        assert!(app_root.is_absolute() && app_root.join("app/main.py").is_file());
        let _child = OwnedProcess(
            Command::new(python)
                .args([
                    "-B",
                    "-c",
                    &seam,
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    &port.to_string(),
                ])
                .current_dir(app_root)
                .env("MODELS_DIR", dir.join("models"))
                .env("MODELS_INI_PATH", dir.join("models/models.ini"))
                .env("DATA_DIR", dir.join("data"))
                .env("MODEL_LOADER_TOKEN", TOKEN)
                .env("DOCKER_HOST", "unix:///tmp/absent-synthetic-docker.sock")
                .env("LLAMA_CONTAINERS", "synthetic-only")
                .env("MIGRATE_CACHE_RAM_ON_START", "false")
                .env_remove("NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED")
                .env_remove("NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED")
                .env_remove("NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let until = Instant::now() + Duration::from_secs(12);
        while request(port, "GET", "/api/v1/sections", true).is_err() {
            assert!(Instant::now() < until, "actual app readiness");
            thread::sleep(Duration::from_millis(50));
        }
        check(
            port,
            "GET",
            "/api/v1/sections",
            false,
            401,
            Some("{\"error\":\"unauthorized\"}"),
        )
        .unwrap();
        let collection = check(port, "GET", "/api/v1/sections", true, 200, None).unwrap();
        assert!(
            collection.contains("\"name\":\"tiny\""),
            "actual collection name"
        );
        assert!(
            collection.contains("\"raw\":\"version = 1\\n[tiny]\\nctx-size = 4096\\n\""),
            "actual collection raw INI"
        );
        let section = check(port, "GET", "/api/v1/sections/tiny", true, 200, None).unwrap();
        assert!(
            section.contains("\"ctx-size\":\"4096\""),
            "actual named ctx value"
        );
        let defaults = check(
            port,
            "GET",
            "/api/v1/sections/tiny?defaults=true",
            true,
            200,
            None,
        )
        .unwrap();
        assert!(
            defaults.contains("\"hints\"") && defaults.contains("\"exists\":true"),
            "actual defaults"
        );
        check(
            port,
            "GET",
            "/api/v1/sections/tiny?defaults=bad",
            true,
            422,
            None,
        )
        .unwrap();
        check(
            port,
            "GET",
            "/api/v1/sections/bad!name",
            true,
            400,
            Some("{\"detail\":\"invalid section name\"}"),
        )
        .unwrap();
        check(port, "HEAD", "/api/v1/sections/tiny", true, 405, Some("")).unwrap();
        check(port, "PUT", "/api/v1/sections/tiny", true, 422, None).unwrap();
        assert_eq!(
            fs::read_to_string(dir.join("models/models.ini")).unwrap(),
            INI
        );
        assert!(fs::read_dir(dir.join("data")).unwrap().next().is_none());
        drop(_child);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn repair_owned_cleanup_controls() {
        let dir = std::env::temp_dir().join(format!("sections-controls-{}", nonce().unwrap()));
        fs::create_dir(&dir).unwrap();
        let source = dir.join("fake.rs");
        fs::write(&source, r#"
use std::{env,fs,io::Write};
fn main() {
 let args:Vec<String>=env::args().skip(1).collect();
 let dir=env::current_exe().unwrap().parent().unwrap().to_owned();
 let mut log=fs::OpenOptions::new().create(true).append(true).open(dir.join("log")).unwrap();
 writeln!(log,"{}",args.join(" ")).unwrap();
 let scenario=fs::read_to_string(dir.join("scenario")).unwrap();
 let id="a".repeat(64);
 match args[0].as_str() {
  "create" => {
   fs::write(dir.join("label"),args[2].split_once('=').unwrap().1).unwrap();
   if scenario=="failure" || scenario=="collision" {std::process::exit(125)}
   if scenario=="late" {std::thread::sleep(std::time::Duration::from_millis(300));}
   println!("{id}");
  },
  "inspect" => {
   let label=if scenario=="mismatch" {"unowned".into()} else {fs::read_to_string(dir.join("label")).unwrap()};
   println!("{id} {label}");
  },
  "rm" => {assert_eq!(args[2],id);},
  _ => std::process::exit(127)
 }
}
"#).unwrap();
        let program = dir.join("fake-docker");
        assert!(Command::new("rustc")
            .arg(&source)
            .arg("-o")
            .arg(&program)
            .status()
            .unwrap()
            .success());
        for launch in ["serve", "omission", "refuse-mismatch", "refuse-invalid"] {
            for scenario in ["success", "failure", "collision", "mismatch", "late"] {
                fs::write(dir.join("scenario"), scenario).unwrap();
                fs::write(dir.join("log"), "").unwrap();
                let result = Container::acquire_with(
                    program.to_str().unwrap(),
                    &[launch],
                    Duration::from_millis(150),
                );
                assert_eq!(result.is_ok(), scenario == "success");
                drop(result);
                let log = fs::read_to_string(dir.join("log")).unwrap();
                assert_eq!(
                    log.lines().any(|line| line.starts_with("rm -f ")),
                    scenario == "success",
                    "{launch}/{scenario}: {log}"
                );
            }
        }
        // Ownership is revalidated at cleanup, even after successful acquisition.
        fs::write(dir.join("scenario"), "success").unwrap();
        fs::write(dir.join("log"), "").unwrap();
        let owned = Container::acquire_with(
            program.to_str().unwrap(),
            &["serve"],
            Duration::from_secs(1),
        )
        .unwrap();
        fs::write(dir.join("scenario"), "mismatch").unwrap();
        drop(owned);
        assert!(!fs::read_to_string(dir.join("log"))
            .unwrap()
            .contains("rm -f"));
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn repair_refusal_rejects_infrastructure_and_diagnostic_mutants() {
        let expected = "MODEL_MANAGER_SECTIONS_IMPL must be python or rust";
        let diagnostic = format!("Error: {expected:?}\n");
        assert!(exact_refusal(Some(1), &diagnostic, expected, false).is_ok());
        assert!(exact_refusal(Some(1), &diagnostic, expected, true).is_err());
        for status in [Some(0), Some(125), Some(126), Some(127), None] {
            assert!(exact_refusal(status, &diagnostic, expected, false).is_err());
        }
        for wrong in ["", "unrelated infrastructure failure", "Error: wrong\n"] {
            assert!(exact_refusal(Some(1), wrong, expected, false).is_err());
        }
        let dir = std::env::temp_dir().join(format!("sections-timeout-{}", nonce().unwrap()));
        fs::create_dir(&dir).unwrap();
        let source = dir.join("late.rs");
        fs::write(
            &source,
            "fn main(){std::thread::sleep(std::time::Duration::from_secs(2));}",
        )
        .unwrap();
        let program = dir.join("late");
        assert!(Command::new("rustc")
            .arg(source)
            .arg("-o")
            .arg(&program)
            .status()
            .unwrap()
            .success());
        assert!(capture(program.to_str().unwrap(), &[], Duration::from_millis(50)).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn actual_http_probe_sends_auth_and_rejects_body_and_status_mutants() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let peer = thread::spawn(move || {
            for _ in 0..3 {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut bytes = Vec::new();
                loop {
                    let mut byte = [0u8; 1];
                    socket.read_exact(&mut byte).unwrap();
                    bytes.extend(byte);
                    assert!(bytes.len() <= 4096);
                    if bytes.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                let text = String::from_utf8(bytes).unwrap();
                assert!(text.starts_with("GET /synthetic HTTP/1.0\r\n"));
                assert!(text.contains(&format!("X-Model-Loader-Token: {TOKEN}\r\n")));
                socket
                    .write_all(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 2\r\n\r\n{}")
                    .unwrap();
            }
        });
        assert!(check(port, "GET", "/synthetic", true, 503, Some("{}")).is_ok());
        assert!(check(port, "GET", "/synthetic", true, 200, Some("{}")).is_err());
        assert!(check(port, "GET", "/synthetic", true, 503, Some("wrong")).is_err());
        peer.join().unwrap();
    }
    #[test]
    fn response_budget_refuses_oversized_peer() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let peer = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut received = Vec::new();
            while !received.ends_with(b"\r\n\r\n") {
                let mut byte = [0u8; 1];
                socket.read_exact(&mut byte).unwrap();
                received.extend(byte);
                assert!(received.len() <= 4096);
            }
            let body = vec![b'x'; 8_388_609];
            let _ = socket.write_all(&body);
        });
        assert!(request(port, "GET", "/synthetic", false)
            .unwrap_err()
            .to_string()
            .contains("HTTP response cap"));
        peer.join().unwrap();
    }
}
