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
const ROOT: &str = "/tmp/noevia-prompts-image";
const TOKEN: &str = "synthetic-prompts-image-token-0123456789abcdef";
const IMAGE: &str = "noevia-model-manager-qualification:ci";
const LABEL: &str = "noevia.prompts-qualification.owner";
const SHIM: &str = r#"#!/usr/local/bin/python
import json,os,sqlite3,sys
from pathlib import Path
from app import main,bench,services
main.hw.start_sampler=lambda:None
bench.reap_orphans=lambda:None
services._docker_client=lambda:None
root=Path('/tmp/noevia-prompts-image');database=Path(os.environ['DATA_DIR'])/'model_loader.db'
pre=None
if database.exists():
 with sqlite3.connect(database) as c:pre=[c.execute('SELECT count(*) FROM prompts').fetchone()[0],c.execute("SELECT value FROM kv WHERE key='bench_prompts_seeded'").fetchone()[0]]
(root/'child.json').write_text(json.dumps({'prompts':os.environ.get('NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED'),'data':os.environ.get('DATA_DIR'),'before':pre,'argv':sys.argv[1:]}))
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
    require(
        marker.contains(&format!("\"data\": \"{ROOT}/data\"")),
        "canonical child DATA_DIR",
    )?;
    match mode {
        "rust" => require(
            marker.contains("\"prompts\": \"1\"")
                && marker.contains("\"before\": [4, \"1\"]")
                && marker.contains("127.0.0.1")
                && marker.contains("8091"),
            "native seed precedes exact child confirmation",
        )?,
        "off" => require(
            marker.contains("\"prompts\": \"0\"") && marker.contains("\"before\": null"),
            "off child owns startup seed",
        )?,
        "python" => require(
            marker.contains("\"prompts\": null")
                && marker.contains("\"before\": null")
                && marker.contains("0.0.0.0"),
            "default clears inherited prompt confirmation",
        )?,
        _ => return Err("unknown image mode".into()),
    }
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
            .then(|| http(8090, "GET", "/api/v1/prompts", "", true))
            .and_then(std::result::Result::ok);
        let private = (present && mode == "rust")
            .then(|| http(8091, "GET", "/api/v1/prompts", "", true))
            .and_then(std::result::Result::ok);
        let within_deadline = Instant::now() < deadline;
        require(within_deadline, "image readiness deadline")?;
        if readiness_ready(
            mode,
            present,
            parent.as_ref(),
            private.as_ref(),
            within_deadline,
        ) {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let marker = bounded_text(&marker)?;
    validate_marker(mode, &marker)?;
    if mode == "rust" {
        for (method, path) in [
            ("GET", "/api/v1/prompts"),
            ("POST", "/api/v1/prompts"),
            ("DELETE", "/api/v1/prompts/5"),
        ] {
            check(
                8091,
                method,
                path,
                "{",
                false,
                401,
                Some("{\"error\":\"unauthorized\"}"),
            )?;
            check(
                8091,
                method,
                path,
                "{",
                true,
                503,
                Some("{\"detail\":\"Native model manager owns this endpoint\"}"),
            )?;
        }
        check(8091, "PUT", "/api/v1/prompts", "", true, 405, None)?;
        check(
            8091,
            "DELETE",
            "/api/v1/prompts/5/extra",
            "",
            true,
            404,
            None,
        )?;
    }
    check(
        8090,
        "POST",
        "/api/v1/prompts",
        "{",
        false,
        401,
        Some("{\"error\":\"unauthorized\"}"),
    )?;
    let collection = check(8090, "GET", "/api/v1/prompts", "", true, 200, None)?;
    for name in [
        "Bench: creative writing",
        "Bench: coding",
        "Bench: reasoning",
        "Bench: summarise long input",
    ] {
        require(collection.contains(name), "whole four literal startup seed")?;
    }
    let added = check(
        8090,
        "POST",
        "/api/v1/prompts",
        r#"{"name":"synthetic-image-added","body":"synthetic body"}"#,
        true,
        200,
        None,
    )?;
    require(
        added.starts_with("{\"id\":5,") && added.contains("synthetic-image-added"),
        "native or default whole POST and ID",
    )?;
    let mut reader = Command::new("/usr/local/bin/python");
    reader.args(["-B","-c","from app import db;print(len(db.list_prompts()));print([r['name'] for r in db.prompts_by_ids([5])]);print(db.get_setting('bench_prompts_seeded',''))"]);
    require(
        String::from_utf8(guards::capture(reader, Duration::from_secs(3))?)?
            == "5\n['synthetic-image-added']\n1\n",
        "actual indirect readers see same WAL commit",
    )?;
    let deleted = check(8090, "DELETE", "/api/v1/prompts/5", "", true, 200, None)?;
    require(
        !deleted.contains("synthetic-image-added"),
        "whole DELETE readback",
    )?;
    check(
        8090,
        "DELETE",
        "/api/v1/prompts/5",
        "",
        true,
        404,
        Some("{\"detail\":\"prompt not found\"}"),
    )?;
    let mut reader = Command::new("/usr/local/bin/python");
    reader.args(["-B","-c","from app import db;print(len(db.list_prompts()));print(len(db.prompts_by_ids([5])));print(db.get_setting('bench_prompts_seeded',''))"]);
    require(
        String::from_utf8(guards::capture(reader, Duration::from_secs(3))?)? == "4\n0\n1\n",
        "actual indirect readers see delete and fixed flag",
    )?;
    Ok(())
}
fn refusal(mode: &str) -> Result<()> {
    prepare()?;
    let binary = if mode == "omission" {
        "/usr/local/bin/model-manager-prompts-omission"
    } else {
        "/usr/local/bin/noevia-model-manager"
    };
    let manager = if mode == "mismatch" { "python" } else { "rust" };
    let selected = if mode == "invalid" { "invalid" } else { "rust" };
    let expected = if mode == "invalid" {
        "MODEL_MANAGER_PROMPTS_IMPL must be python or rust"
    } else {
        "native-prompts requires rust model manager and compiled native-prompts feature"
    };
    if mode == "omission" {
        let mut features = Command::new(binary);
        features.arg("--features");
        let features = String::from_utf8(guards::capture(features, Duration::from_secs(5))?)?;
        require(
            features.lines().any(|line| line == "native-host")
                && !features.lines().any(|line| line == "native-prompts"),
            "compiled prompt omission preserves host",
        )?;
    }
    let mut command = Command::new(binary);
    command
        .env("PATH", format!("{ROOT}/bin:/usr/local/bin:/usr/bin:/bin"))
        .env("MODEL_MANAGER_IMPL", manager)
        .env("MODEL_MANAGER_PROMPTS_IMPL", selected)
        .env("MODEL_MANAGER_HOST_IMPL", "python")
        .env("MODEL_MANAGER_SECTIONS_IMPL", "python")
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
        "canonical child DATA_DIR",
        "native seed precedes exact child confirmation",
        "off child owns startup seed",
        "default clears inherited prompt confirmation",
        "whole four literal startup seed",
        "native or default whole POST and ID",
        "actual indirect readers see same WAL commit",
        "whole DELETE readback",
        "actual indirect readers see delete and fixed flag",
        "compiled prompt omission preserves host",
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
    format!("prompt image Docker {operation} mode={mode} failed: {diagnostic}")
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
        let prompts = if mode == "rust" {
            "MODEL_MANAGER_PROMPTS_IMPL=rust"
        } else {
            "MODEL_MANAGER_PROMPTS_IMPL=python"
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
            "PATH=/tmp/noevia-prompts-image/bin:/usr/local/bin:/usr/bin:/bin",
            "-e",
            "PYTHONPATH=/srv",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "MODELS_DIR=/tmp/noevia-prompts-image/models",
            "-e",
            "MODELS_INI_PATH=/tmp/noevia-prompts-image/models/models.ini",
            "-e",
            "DATA_DIR=/tmp/noevia-prompts-image/data",
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
            manager,
            "-e",
            prompts,
            "-e",
            "NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED=1",
            "-e",
            "MODEL_LOADER_TOKEN=synthetic-prompts-image-token-0123456789abcdef",
            "--entrypoint",
            "prompts-image-probe",
            image,
            "serve",
        ])?;
        docker(&["start", &owned.id])?;
        docker(&["exec", &owned.id, "prompts-image-probe", "probe", mode])?;
    }
    for mode in ["omission", "mismatch", "invalid"] {
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
            "prompts-image-probe",
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
        (mode @ ("omission" | "mismatch" | "invalid"), None) => refusal(mode),
        ("probe", Some(mode)) => probe(mode),
        ("images", Some(image)) => images(image),
        _ => Err("invalid image mode".into()),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn readiness_requires_successful_owned_endpoints_before_deadline() {
        let public = (200, String::new());
        let owned = (
            503,
            "{\"detail\":\"Native model manager owns this endpoint\"}".to_owned(),
        );
        for mode in ["python", "off", "rust"] {
            assert!(super::readiness_ready(
                mode,
                true,
                Some(&public),
                Some(&owned),
                true
            ));
            assert!(!super::readiness_ready(
                mode,
                false,
                Some(&public),
                Some(&owned),
                true
            ));
            assert!(!super::readiness_ready(
                mode,
                true,
                None,
                Some(&owned),
                true
            ));
            assert!(!super::readiness_ready(
                mode,
                true,
                Some(&public),
                Some(&owned),
                false
            ));
            for status in [401, 404, 500, 502, 503] {
                assert!(!super::readiness_ready(
                    mode,
                    true,
                    Some(&(status, String::new())),
                    Some(&owned),
                    true
                ));
            }
        }
        assert!(!super::readiness_ready(
            "rust",
            true,
            Some(&public),
            None,
            true
        ));
        assert!(!super::readiness_ready(
            "rust",
            true,
            Some(&public),
            Some(&public),
            true
        ));
        assert!(!super::readiness_ready(
            "rust",
            true,
            Some(&public),
            Some(&(503, "unrelated refusal".to_owned())),
            true
        ));
        assert!(super::readiness_ready(
            "off",
            true,
            Some(&public),
            None,
            true
        ));
        assert!(super::readiness_ready(
            "python",
            true,
            Some(&public),
            None,
            true
        ));
    }
    #[test]
    fn failed_docker_diagnostics_are_fixed_and_reject_contamination() {
        use std::os::unix::process::ExitStatusExt;
        let args = [
            "exec",
            "private-container-id",
            "prompts-image-probe",
            "probe",
            "rust",
        ];
        let known = b"Error: \"image HTTP status assertion\"\n";
        let failed = super::docker_output(
            &args,
            super::guards::Output {
                status: std::process::ExitStatus::from_raw(256),
                stdout: b"private-body-and-token".to_vec(),
                stderr: known.to_vec(),
            },
        )
        .unwrap_err();
        assert_eq!(
            failed.to_string(),
            "prompt image Docker exec mode=rust failed: image HTTP status assertion"
        );
        assert_eq!(
            super::docker_output(
                &args,
                super::guards::Output {
                    status: std::process::ExitStatus::from_raw(0),
                    stdout: b"fixed-success".to_vec(),
                    stderr: Vec::new(),
                }
            )
            .unwrap(),
            b"fixed-success"
        );
        assert_eq!(
            super::docker_failure(&args, known),
            "prompt image Docker exec mode=rust failed: image HTTP status assertion"
        );
        for bytes in [
            b"secret-token".to_vec(),
            [known.as_slice(), b"private-body"].concat(),
            [b"private-prefix".as_slice(), known.as_slice()].concat(),
            vec![b'x'; 65_537],
        ] {
            assert_eq!(
                super::docker_failure(&args, &bytes),
                "prompt image Docker exec mode=rust failed: unrecognized bounded diagnostic"
            );
        }
        assert_eq!(
            super::docker_failure(&["private-operation", "private-mode"], known),
            "prompt image Docker unknown mode=none failed: image HTTP status assertion"
        );
        assert_eq!(
            super::docker_failure(&["exec", "private-mode"], known),
            "prompt image Docker exec mode=unknown failed: image HTTP status assertion"
        );
    }
    #[test]
    fn send_budget_bounds_slow_progress_and_no_progress() {
        struct Slow {
            zero: bool,
            interrupted: bool,
            calls: usize,
        }
        impl super::SendStep for Slow {
            fn send(
                &mut self,
                bytes: &[u8],
                remaining: std::time::Duration,
            ) -> std::io::Result<usize> {
                assert!(bytes.len() <= 4096);
                assert!(remaining <= std::time::Duration::from_millis(20));
                self.calls += 1;
                std::thread::sleep(std::time::Duration::from_millis(3));
                if self.interrupted {
                    Err(std::io::ErrorKind::Interrupted.into())
                } else {
                    Ok(if self.zero { 0 } else { 1 })
                }
            }
        }
        for interrupted in [false, true] {
            let mut slow = Slow {
                zero: false,
                interrupted,
                calls: 0,
            };
            let error = super::send_request(
                &mut slow,
                &[0; 8192],
                std::time::Instant::now() + std::time::Duration::from_millis(20),
            )
            .unwrap_err();
            assert_eq!(error.to_string(), "image HTTP deadline");
            assert!(slow.calls < 20);
        }
        let mut zero = Slow {
            zero: true,
            interrupted: false,
            calls: 0,
        };
        assert_eq!(
            super::send_request(
                &mut zero,
                b"fixed",
                std::time::Instant::now() + std::time::Duration::from_millis(20)
            )
            .unwrap_err()
            .to_string(),
            "image HTTP zero write"
        );
        assert_eq!(zero.calls, 1);
    }
    #[test]
    fn cleanup_requires_exact_immutable_id_and_owned_label() {
        let id = "a".repeat(64);
        let token = "b".repeat(64);
        let output = format!("{id} {token}\n");
        assert!(super::owns_container(&output, &id, &token));
        for incorrect in [
            String::new(),
            format!("{} {token}", "c".repeat(64)),
            format!("{id} {}", "c".repeat(64)),
            format!("{output}extra"),
            format!("{id} <no value>"),
        ] {
            assert!(!super::owns_container(&incorrect, &id, &token));
        }
        assert!(!super::owns_container(&output, "name", &token));
        assert!(!super::owns_container(&output, &id, ""));
    }
    #[test]
    fn marker_requires_canonical_data_preseed_and_exact_confirmation() {
        let native = r#"{"data": "/tmp/noevia-prompts-image/data", "prompts": "1", "before": [4, "1"], "argv": ["127.0.0.1", "8091"]}"#;
        assert!(super::validate_marker("rust", native).is_ok());
        for broken in [
            native.replace("\"prompts\": \"1\"", "\"prompts\": \"true\""),
            native.replace("[4, \"1\"]", "null"),
            native.replace("8091", "8090"),
            native.replace("/data", "/other"),
        ] {
            assert!(super::validate_marker("rust", &broken).is_err());
        }
        let off = native
            .replace("\"prompts\": \"1\"", "\"prompts\": \"0\"")
            .replace("[4, \"1\"]", "null");
        assert!(super::validate_marker("off", &off).is_ok());
        assert!(super::validate_marker("rust", &off).is_err());
        let python = off
            .replace("\"prompts\": \"0\"", "\"prompts\": null")
            .replace("127.0.0.1", "0.0.0.0");
        assert!(super::validate_marker("python", &python).is_ok());
        assert!(super::validate_marker("python", native).is_err());
    }
}
