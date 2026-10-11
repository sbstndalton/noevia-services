//! Whole synthetic runtime qualification; Rust owns launch, fixtures, IO and assertions.
#![forbid(unsafe_code)]
#[path = "prompts.rs"]
#[allow(dead_code)]
mod guards;
use std::{
    error::Error,
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    os::unix::{
        fs::{DirBuilderExt, PermissionsExt},
        process::CommandExt,
    },
    path::Path,
    process::Command,
    thread,
    time::{Duration, Instant},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const ROOT: &str = "/tmp/noevia-diagnosis-image";
const TOKEN: &str = "synthetic-diagnosis-image-token-0123456789abcdef";
const IMAGE: &str = "noevia-model-manager-qualification:ci";
const LABEL: &str = "noevia.diagnosis-qualification.owner";
const SHIM: &str = r#"#!/usr/local/bin/python
import json,os,sys
from pathlib import Path
from app import main,api,services
main.app.router.on_startup.clear()
services._docker_client=lambda:None
services._effective_container_names=lambda:['synthetic-only']
api._logs_since_start=lambda name:''
root=Path('/tmp/noevia-diagnosis-image')
(root/'child.json').write_text(json.dumps({'diagnose':os.environ.get('NOEVIA_NATIVE_MODEL_MANAGER_DIAGNOSE_CONFIRMED'),'tz':os.environ.get('TZ'),'argv':sys.argv[1:]}))
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
fn prepare() -> Result<()> {
    fs::DirBuilder::new().mode(0o700).create(ROOT)?;
    for name in ["bin", "data", "models"] {
        fs::DirBuilder::new()
            .mode(0o700)
            .create(format!("{ROOT}/{name}"))?;
    }
    fs::write(format!("{ROOT}/models/models.ini"), "version = 1\n")?;
    let shim = format!("{ROOT}/bin/uvicorn");
    fs::write(&shim, SHIM)?;
    fs::set_permissions(shim, fs::Permissions::from_mode(0o700))?;
    Ok(())
}
fn serve() -> Result<()> {
    prepare()?;
    let error = Command::new("/usr/local/bin/noevia-model-manager").exec();
    Err(error.into())
}
trait SendStep {
    fn send(&mut self, bytes: &[u8], remaining: Duration) -> std::io::Result<usize>;
}
impl SendStep for TcpStream {
    fn send(&mut self, bytes: &[u8], remaining: Duration) -> std::io::Result<usize> {
        self.set_write_timeout(Some(remaining))?;
        self.write(bytes)
    }
}
fn send_request(stream: &mut impl SendStep, request: &[u8], deadline: Instant) -> Result<()> {
    require(request.len() <= 1_065_000, "image HTTP request cap")?;
    let mut sent = 0;
    while sent < request.len() {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or("image HTTP deadline")?;
        let end = sent.saturating_add(4096).min(request.len());
        let chunk = request.get(sent..end).ok_or("image HTTP send range")?;
        match stream.send(chunk, remaining) {
            Ok(0) => return Err("image HTTP zero write".into()),
            Ok(n) if n <= chunk.len() => sent += n,
            Ok(_) => return Err("image HTTP invalid write count".into()),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}
fn http(port: u16, method: &str, path: &str, body: &str, auth: bool) -> Result<(u16, String)> {
    let mut stream = TcpStream::connect_timeout(
        &SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(250),
    )?;
    require(
        method.len() <= 16 && method.bytes().all(|b| b.is_ascii_uppercase()),
        "image HTTP method",
    )?;
    require(
        path.len() <= 8192 && !path.bytes().any(|b| b <= 32 || b == 127),
        "image HTTP path",
    )?;
    require(body.len() <= 1_048_576, "image HTTP body cap")?;
    let token = if auth {
        format!("X-Model-Loader-Token: {TOKEN}\r\n")
    } else {
        String::new()
    };
    let request = format!("{method} {path} HTTP/1.0\r\nHost: localhost\r\n{token}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    require(request.len() <= 1_065_000, "image HTTP request cap")?;
    let deadline = Instant::now() + Duration::from_secs(3);
    send_request(&mut stream, request.as_bytes(), deadline)?;
    let mut bytes = Vec::new();
    loop {
        stream.set_read_timeout(Some(
            deadline
                .checked_duration_since(Instant::now())
                .ok_or("image HTTP deadline")?,
        ))?;
        let mut chunk = [0; 4096];
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        require(bytes.len() + n <= 65_536, "image HTTP response cap")?;
        bytes.try_reserve(n)?;
        bytes.extend_from_slice(chunk.get(..n).ok_or("invalid HTTP count")?);
    }
    let text = String::from_utf8(bytes)?;
    let (header, body) = text.split_once("\r\n\r\n").ok_or("image HTTP framing")?;
    Ok((
        header
            .split_whitespace()
            .nth(1)
            .ok_or("image HTTP status")?
            .parse()?,
        body.to_owned(),
    ))
}
fn check(
    port: u16,
    method: &str,
    path: &str,
    body: &str,
    auth: bool,
    status: u16,
    expected: Option<&str>,
) -> Result<String> {
    let (actual, value) = http(port, method, path, body, auth)?;
    require(actual == status, "image HTTP status assertion")?;
    if let Some(expected) = expected {
        require(value == expected, "image HTTP body assertion")?;
    }
    Ok(value)
}
fn bounded_text(path: &Path) -> Result<String> {
    let mut bytes = Vec::new();
    fs::File::open(path)?.take(4097).read_to_end(&mut bytes)?;
    require(bytes.len() <= 4096, "image marker cap")?;
    Ok(String::from_utf8(bytes)?)
}
fn validate_marker(mode: &str, marker: &str) -> Result<()> {
    let confirmation = match mode {
        "rust" => "\"diagnose\": \"1\"",
        "off" => "\"diagnose\": \"0\"",
        "python" => "\"diagnose\": null",
        _ => return Err("unknown image mode".into()),
    };
    require(
        marker.contains(confirmation),
        "exact child diagnosis confirmation",
    )?;
    require(marker.contains("\"tz\": \"UTC\""), "child exact UTC")?;
    let (host, port) = if mode == "python" {
        ("0.0.0.0", "8090")
    } else {
        ("127.0.0.1", "8091")
    };
    require(
        marker.contains(host) && marker.contains(port),
        "exact child bind",
    )?;
    Ok(())
}

fn readiness_ready(
    mode: &str,
    marker: bool,
    parent: Option<&(u16, String)>,
    private: Option<&(u16, String)>,
    within_deadline: bool,
) -> bool {
    within_deadline
        && marker
        && parent.is_some_and(|(status, _)| *status == 200)
        && (mode != "rust"
            || private.is_some_and(|(status, body)| {
                *status == 503 && body == "{\"detail\":\"Native model manager owns this endpoint\"}"
            }))
}
fn probe(mode: &str) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(12);
    let marker = Path::new(ROOT).join("child.json");
    loop {
        require(Instant::now() < deadline, "image readiness deadline")?;
        let present = marker.is_file();
        let parent = present
            .then(|| {
                http(
                    8090,
                    "GET",
                    "/api/v1/backends/synthetic-only/diagnose",
                    "",
                    true,
                )
            })
            .and_then(std::result::Result::ok);
        let private = (present && mode == "rust")
            .then(|| {
                http(
                    8091,
                    "GET",
                    "/api/v1/backends/synthetic-only/diagnose",
                    "",
                    true,
                )
            })
            .and_then(std::result::Result::ok);
        if readiness_ready(
            mode,
            present,
            parent.as_ref(),
            private.as_ref(),
            Instant::now() < deadline,
        ) {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    validate_marker(mode, &bounded_text(&marker)?)?;
    for port in if mode == "python" {
        vec![8090]
    } else {
        vec![8090, 8091]
    } {
        check(
            port,
            "GET",
            "/api/v1/backends/synthetic-only/diagnose",
            "",
            false,
            401,
            Some("{\"error\":\"unauthorized\"}"),
        )?;
        let owned = mode == "rust" && port == 8091;
        check(
            port,
            "GET",
            "/api/v1/backends/synthetic-only/diagnose",
            "",
            true,
            if owned { 503 } else { 200 },
            Some(if owned {
                "{\"detail\":\"Native model manager owns this endpoint\"}"
            } else {
                "{\"failures\":[]}"
            }),
        )?;
        check(
            port,
            "GET",
            "/api/v1/backends/unknown/diagnose",
            "",
            true,
            if owned { 503 } else { 404 },
            Some(if owned {
                "{\"detail\":\"Native model manager owns this endpoint\"}"
            } else {
                "{\"detail\":\"unknown backend\"}"
            }),
        )?;
        check(
            port,
            "POST",
            "/api/v1/backends/synthetic-only/diagnose",
            "",
            true,
            405,
            None,
        )?;
        check(
            port,
            "GET",
            "/api/v1/backends/synthetic-only/diagnose/extra",
            "",
            true,
            404,
            None,
        )?;
    }
    Ok(())
}

fn refusal(mode: &str) -> Result<()> {
    prepare()?;
    let binary = if mode == "omission" {
        "/usr/local/bin/model-manager-diagnosis-omission"
    } else {
        "/usr/local/bin/noevia-model-manager"
    };
    let manager = if mode == "mismatch" { "python" } else { "rust" };
    let selected = if mode == "invalid" { "invalid" } else { "rust" };
    let expected = if mode == "timezone" {
        "native-diagnose requires exact TZ=UTC"
    } else if mode == "invalid" {
        "MODEL_MANAGER_DIAGNOSE_IMPL must be python or rust"
    } else {
        "native-diagnose requires rust model manager and compiled native-diagnose feature"
    };
    if mode == "omission" {
        let mut features = Command::new(binary);
        features.arg("--features");
        let features = String::from_utf8(guards::capture(features, Duration::from_secs(5))?)?;
        require(
            features.lines().any(|line| line == "native-host")
                && !features.lines().any(|line| line == "native-diagnose"),
            "compiled diagnosis omission preserves host",
        )?;
    }
    let mut command = Command::new(binary);
    command
        .env("PATH", format!("{ROOT}/bin:/usr/local/bin:/usr/bin:/bin"))
        .env("MODEL_MANAGER_IMPL", manager)
        .env("MODEL_MANAGER_DIAGNOSE_IMPL", selected)
        .env("MODEL_MANAGER_HOST_IMPL", "python")
        .env("MODEL_MANAGER_SECTIONS_IMPL", "python")
        .env("MODEL_MANAGER_PROMPTS_IMPL", "python")
        .env(
            "TZ",
            if mode == "timezone" {
                "America/New_York"
            } else {
                "UTC"
            },
        )
        .env("DATA_DIR", format!("{ROOT}/data"));
    let output = guards::capture_output(command, Duration::from_secs(5))?;
    require(
        output.status.code() == Some(1),
        "exact selector refusal exit",
    )?;
    require(
        output.stdout.is_empty() && output.stderr == format!("Error: {expected:?}\n").as_bytes(),
        "exact selector refusal diagnostic",
    )?;
    require(
        !Path::new(ROOT).join("child.json").exists()
            && fs::read_dir(format!("{ROOT}/data"))?.next().is_none(),
        "selector refusal before child and database",
    )?;
    Ok(())
}
fn docker(args: &[&str]) -> Result<Vec<u8>> {
    let mut command = Command::new("docker");
    command.args(args);
    let output = guards::capture_output(command, Duration::from_secs(30))?;
    docker_output(args, output)
}
fn docker_output(args: &[&str], output: guards::Output) -> Result<Vec<u8>> {
    if !output.status.success() {
        return Err(docker_failure(args, &output.stderr).into());
    }
    Ok(output.stdout)
}
fn docker_failure(args: &[&str], stderr: &[u8]) -> String {
    let operation = match args.first().copied() {
        Some("create") => "create",
        Some("inspect") => "inspect",
        Some("start") => "start",
        Some("exec") => "exec",
        Some("rm") => "cleanup",
        _ => "unknown",
    };
    let mode = if operation == "exec" {
        match args.last().copied() {
            Some("python") => "python",
            Some("rust") => "rust",
            Some("off") => "off",
            Some("omission") => "omission",
            Some("mismatch") => "mismatch",
            Some("invalid") => "invalid",
            Some("timezone") => "timezone",
            _ => "unknown",
        }
    } else {
        "none"
    };
    // Accept only a complete, single registered Rust diagnostic. Never relay
    // arbitrary Docker/Python output, environment, request bodies or paths.
    let diagnostic = [
        "image HTTP status assertion",
        "image HTTP body assertion",
        "image HTTP deadline",
        "image HTTP response cap",
        "image readiness deadline",
        "image marker cap",
        "exact child diagnosis confirmation",
        "child exact UTC",
        "exact child bind",
        "compiled diagnosis omission preserves host",
        "exact selector refusal exit",
        "exact selector refusal diagnostic",
        "selector refusal before child and database",
        "actual app qualification process failed",
        "actual app qualification deadline",
        "actual app qualification output cap",
    ]
    .into_iter()
    .find(|message| stderr.len() <= 65_536 && stderr == format!("Error: {message:?}\n").as_bytes())
    .unwrap_or("unrecognized bounded diagnostic");
    format!("diagnosis image Docker {operation} mode={mode} failed: {diagnostic}")
}
fn nonce() -> Result<String> {
    let mut bytes = [0; 32];
    fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}
struct Container {
    id: String,
    token: String,
}
fn owns_container(output: &str, id: &str, token: &str) -> bool {
    let hex = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    hex(id) && hex(token) && output.trim() == format!("{id} {token}")
}
impl Container {
    fn acquire(args: &[&str]) -> Result<Self> {
        let token = nonce()?;
        let label = format!("{LABEL}={token}");
        let mut create = vec!["create", "--label", &label];
        create.extend_from_slice(args);
        let output = String::from_utf8(docker(&create)?)?;
        let id = output.trim();
        require(
            id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit()),
            "immutable container ID",
        )?;
        let owned = Self {
            id: id.to_owned(),
            token,
        };
        require(owned.verified()?, "owned container label")?;
        Ok(owned)
    }
    fn verified(&self) -> Result<bool> {
        let format = format!("{{{{.Id}}}} {{{{index .Config.Labels \"{LABEL}\"}}}}");
        let output = String::from_utf8(docker(&["inspect", "--format", &format, &self.id])?)?;
        Ok(owns_container(&output, &self.id, &self.token))
    }
}
impl Drop for Container {
    fn drop(&mut self) {
        if self.verified().unwrap_or(false) {
            let _ = docker(&["rm", "-f", &self.id]);
        }
    }
}
fn images(image: &str) -> Result<()> {
    require(image == IMAGE, "fixed qualification image")?;
    for mode in ["python", "rust", "off"] {
        let manager = if mode == "python" {
            "MODEL_MANAGER_IMPL=python"
        } else {
            "MODEL_MANAGER_IMPL=rust"
        };
        let diagnosis = if mode == "rust" {
            "MODEL_MANAGER_DIAGNOSE_IMPL=rust"
        } else {
            "MODEL_MANAGER_DIAGNOSE_IMPL=python"
        };
        let owned = Container::acquire(&[
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
            "PATH=/tmp/noevia-diagnosis-image/bin:/usr/local/bin:/usr/bin:/bin",
            "-e",
            "PYTHONPATH=/srv",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "MODELS_DIR=/tmp/noevia-diagnosis-image/models",
            "-e",
            "MODELS_INI_PATH=/tmp/noevia-diagnosis-image/models/models.ini",
            "-e",
            "DATA_DIR=/tmp/noevia-diagnosis-image/data",
            "-e",
            "DOCKER_HOST=unix:///tmp/absent-synthetic.sock",
            "-e",
            "LLAMA_CONTAINERS=synthetic-only",
            "-e",
            "MIGRATE_CACHE_RAM_ON_START=false",
            "-e",
            "MODEL_MANAGER_HOST_IMPL=python",
            "-e",
            "MODEL_MANAGER_SECTIONS_IMPL=python",
            "-e",
            "MODEL_MANAGER_PROMPTS_IMPL=python",
            "-e",
            "TZ=UTC",
            "-e",
            manager,
            "-e",
            diagnosis,
            "-e",
            "NOEVIA_NATIVE_MODEL_MANAGER_DIAGNOSE_CONFIRMED=hostile",
            "-e",
            "MODEL_LOADER_TOKEN=synthetic-diagnosis-image-token-0123456789abcdef",
            "--entrypoint",
            "diagnosis-image-probe",
            image,
            "serve",
        ])?;
        docker(&["start", &owned.id])?;
        docker(&["exec", &owned.id, "diagnosis-image-probe", "probe", mode])?;
    }
    for mode in ["omission", "mismatch", "invalid", "timezone"] {
        let owned = Container::acquire(&[
            "--network",
            "none",
            "--log-driver",
            "none",
            "--read-only",
            "--memory",
            "128m",
            "--pids-limit",
            "32",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,exec,size=16777216,mode=1777",
            "--entrypoint",
            "diagnosis-image-probe",
            image,
            mode,
        ])?;
        docker(&["start", "-a", &owned.id])?;
        require(
            String::from_utf8(docker(&[
                "inspect",
                "--format",
                "{{.State.ExitCode}}",
                &owned.id,
            ])?)?
            .trim()
                == "0",
            "actual refusal probe completed successfully",
        )?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or("image mode required")?;
    let value = args.next();
    require(args.next().is_none(), "unexpected image arguments")?;
    match (mode.as_str(), value.as_deref()) {
        ("serve", None) => serve(),
        (mode @ ("omission" | "mismatch" | "invalid" | "timezone"), None) => refusal(mode),
        ("probe", Some(mode)) => probe(mode),
        ("images", Some(image)) => images(image),
        _ => Err("invalid image mode".into()),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn confirmation_and_binding_are_exact() {
        let selected = r#"{"diagnose": "1", "tz": "UTC", "argv": ["127.0.0.1", "8091"]}"#;
        assert!(super::validate_marker("rust", selected).is_ok());
        for changed in [
            selected.replace("\"1\"", "\"true\""),
            selected.replace("UTC", "Etc/UTC"),
            selected.replace("8091", "8090"),
            selected.replace("127.0.0.1", "0.0.0.0"),
        ] {
            assert!(super::validate_marker("rust", &changed).is_err());
        }
        assert!(super::validate_marker("off", &selected.replace("\"1\"", "\"0\"")).is_ok());
        assert!(super::validate_marker(
            "python",
            r#"{"diagnose": null, "tz": "UTC", "argv": ["0.0.0.0", "8090"]}"#
        )
        .is_ok());
    }
}
