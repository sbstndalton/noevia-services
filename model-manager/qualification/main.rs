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
    let mut child = OwnedProcess(
        Command::new("docker")
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
    prepare()?;
    let diagnostic = format!("{ROOT}/omission.stderr");
    let mut child = OwnedProcess(
        Command::new("/usr/local/bin/model-manager-sections-omission")
            .env("MODEL_MANAGER_IMPL", "rust")
            .env("MODEL_MANAGER_SECTIONS_IMPL", "rust")
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
    require(!status.success(), "omission binary accepted sections")?;
    let mut bytes = Vec::new();
    fs::File::open(diagnostic)?
        .take(65_537)
        .read_to_end(&mut bytes)?;
    require(bytes.len() <= 65_536, "omission diagnostic cap")?;
    require(
        String::from_utf8(bytes)?.contains(
            "native-sections requires rust model manager and compiled native-sections feature",
        ),
        "specific omitted capability refusal",
    )?;
    require(
        !Path::new(&format!("{ROOT}/child.json")).exists(),
        "omission spawned actual app",
    )?;
    Ok(())
}
struct Container(String);
impl Drop for Container {
    fn drop(&mut self) {
        let _ = command(&["rm", "-f", &self.0]);
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
    let (actual, text) = request(port, method, path, authorized)?;
    require(actual == status, "HTTP status mismatch")?;
    if let Some(expected) = body {
        require(text == expected, "HTTP body mismatch")?;
    }
    Ok(text)
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
    let until = Instant::now() + Duration::from_secs(12);
    loop {
        if request(8090, "GET", "/api/v1/sections", true).is_ok()
            && Path::new(&format!("{ROOT}/child.json")).is_file()
        {
            break;
        }
        require(Instant::now() < until, "owned image readiness deadline")?;
        thread::sleep(Duration::from_millis(50));
    }
    let marker = fs::read_to_string(format!("{ROOT}/child.json"))?;
    require(marker.len() < 1024, "child marker cap")?;
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
        require(
            marker.contains("\"sections\": null")
                && marker.contains("\"native\": null")
                && marker.contains("\"host\": null")
                && marker.contains("0.0.0.0")
                && marker.contains("8090"),
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
        let name = format!("sections-qualification-{}-{mode}", std::process::id());
        let owned = Container(name.clone());
        let selected = format!("MODEL_MANAGER_IMPL={mode}");
        let sections = format!("MODEL_MANAGER_SECTIONS_IMPL={mode}");
        let mut args = vec![
            "run",
            "-d",
            "--name",
            &name,
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
            "NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED=1",
            "-e",
            "NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED=1",
            "-e",
            "NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED=1",
            "-e",
            "MODEL_LOADER_TOKEN=synthetic-image-token-0000000000000000000",
        ];
        if mode == "rust" {
            args.extend(["-e", &selected, "-e", &sections]);
        }
        args.extend(["--entrypoint", "sections-image-probe", image, "serve"]);
        require(command(&args)?, "owned image start")?;
        require(
            command(&["exec", &name, "sections-image-probe", "probe", mode])?,
            "actual image HTTP qualification",
        )?;
        drop(owned);
    }
    let name = format!("sections-omission-{}", std::process::id());
    let _owned = Container(name.clone());
    require(
        command(&[
            "run",
            "--name",
            &name,
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
            "omission",
        ])?,
        "specific sections omission qualification",
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or("missing mode")?;
    let value = args.next();
    require(args.next().is_none(), "unexpected arguments")?;
    match (mode.as_str(), value.as_deref()) {
        ("images", Some(image)) => images(image),
        ("probe", Some(mode @ ("python" | "rust"))) => probe(mode),
        ("omission", None) => omission(),
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
