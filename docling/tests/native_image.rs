//! Test-only fixed HTTP image probe. It never imports an engine, prints bodies,
//! selects commands from requests, or accesses host/network-mounted state.
#![forbid(unsafe_code)]
use std::{
    io::{Read, Write},
    net::{Shutdown, TcpStream, UdpSocket},
    time::Duration,
};
type Error = Box<dyn std::error::Error>;
struct Reply {
    status: u16,
    body: String,
}
fn request(body: Option<&[u8]>) -> Result<TcpStream, Error> {
    let mut socket =
        TcpStream::connect_timeout(&"127.0.0.1:8031".parse()?, Duration::from_secs(5))?;
    socket.set_read_timeout(Some(Duration::from_secs(10)))?;
    socket.set_write_timeout(Some(Duration::from_secs(5)))?;
    let header = if let Some(data) = body {
        format!("POST /extract HTTP/1.0\r\nHost: localhost\r\nX-Document-Name: synthetic.docx\r\nContent-Length: {}\r\n\r\n",data.len())
    } else {
        "GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n".to_owned()
    };
    socket.write_all(header.as_bytes())?;
    if let Some(data) = body {
        socket.write_all(data)?;
    }
    Ok(socket)
}
fn response(mut socket: TcpStream) -> Result<Reply, Error> {
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match socket.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                if bytes
                    .len()
                    .checked_add(n)
                    .is_none_or(|size| size > 256 * 1024)
                {
                    return Err("image reply cap".into());
                }
                bytes.extend_from_slice(chunk.get(..n).ok_or("read count")?);
            }
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => break,
            Err(e) => return Err(e.into()),
        }
    }
    let split = bytes
        .windows(4)
        .position(|b| b == b"\r\n\r\n")
        .ok_or("HTTP headers missing")?;
    let headers = std::str::from_utf8(bytes.get(..split).ok_or("headers")?)?;
    let body = bytes
        .get(split.checked_add(4).ok_or("body offset")?..)
        .ok_or("body")?;
    let field = |name: &str| {
        headers
            .lines()
            .find_map(|line| {
                line.split_once(':')
                    .filter(|(key, _)| key.eq_ignore_ascii_case(name))
                    .map(|(_, v)| v.trim())
            })
            .ok_or("response header missing")
    };
    let size = field("Content-Length")?.parse::<usize>()?;
    if size != body.len() || field("Content-Type")? != "application/json" {
        return Err("incomplete/non-JSON response".into());
    }
    Ok(Reply {
        status: headers.split_whitespace().nth(1).ok_or("status")?.parse()?,
        body: std::str::from_utf8(body)?.to_owned(),
    })
}
// Fixed synthetic shapes, not a general JSON parser. Preserve quoted whitespace.
fn compact_json(body: &str) -> Result<String, Error> {
    let mut result = String::new();
    let (mut quoted, mut escaped) = (false, false);
    for ch in body.chars() {
        if quoted {
            result.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                quoted = false;
            }
        } else if ch == '"' {
            quoted = true;
            result.push(ch);
        } else if !ch.is_ascii_whitespace() {
            result.push(ch);
        }
    }
    if quoted {
        return Err("unterminated JSON string".into());
    }
    Ok(result)
}
fn pid(reply: &Reply) -> Result<u32, Error> {
    if reply.status != 200 {
        return Err("synthetic extraction failed".into());
    }
    let body = compact_json(&reply.body)?;
    let value = body
        .split_once("\"pid\":")
        .map(|(_, s)| s)
        .ok_or("worker PID missing")?;
    let number: String = value.chars().take_while(char::is_ascii_digit).collect();
    let pid: u32 = number.parse()?;
    if pid <= 1 {
        return Err("invalid worker PID".into());
    }
    let python = format!(
        r#"{{"pages":[{{"number":1,"text":"ok","status":"native","method":"docling","truncated":false}}],"total":1,"truncatedPages":false,"pid":{pid}}}"#
    );
    let native = format!(
        r#"{{"pages":[{{"method":"docling","number":1,"status":"native","text":"ok","truncated":false}}],"pid":{pid},"total":1,"truncatedPages":false}}"#
    );
    if body != python && body != native {
        return Err("synthetic extraction shape drift".into());
    }
    Ok(pid)
}
fn health_valid(reply: &Reply) -> Result<bool, Error> {
    let formats = r#"[".bmp",".csv",".docx",".epub",".htm",".html",".jpeg",".jpg",".odp",".ods",".odt",".pdf",".png",".pptx",".tif",".tiff",".xlsx"]"#;
    // noevia#1867: both owners advertise page ranges and the ranged body cap (default 96 MiB).
    let python = format!(r#"{{"service":"docling","formats":{formats},"ranges":true,"rangedMaxBytes":100663296}}"#);
    let native = format!(r#"{{"formats":{formats},"rangedMaxBytes":100663296,"ranges":true,"service":"docling"}}"#);
    let body = compact_json(&reply.body)?;
    Ok(reply.status == 200 && (body == python || body == native))
}
fn cpu_valid(cpu: &str) -> bool {
    let mut fields = cpu.split_whitespace();
    match (fields.next(), fields.next(), fields.next()) {
        (Some(quota), Some(period), None) => match (quota.parse::<u64>(), period.parse::<u64>()) {
            (Ok(q), Ok(p)) => q > 0 && q == p,
            _ => false,
        },
        _ => false,
    }
}
fn absent_worker(pid: u32) -> Result<(), Error> {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(s)
            if s.rsplit_once(')')
                .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z')) =>
        {
            Ok(())
        }
        _ => Err("cancelled worker remains running".into()),
    }
}
fn temporary_inputs_clean() -> Result<(), Error> {
    for entry in std::fs::read_dir("/tmp")? {
        if entry?
            .file_name()
            .to_string_lossy()
            .starts_with("noevia-docling-")
        {
            return Err("generated input survived".into());
        }
    }
    Ok(())
}
fn isolate() -> Result<(), Error> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    if !status
        .lines()
        .any(|l| l.starts_with("Uid:") && l.split_whitespace().nth(1) == Some("65534"))
        || !status.lines().any(|l| l == "CapEff:\t0000000000000000")
        || !status.lines().any(|l| l == "NoNewPrivs:\t1")
    {
        return Err("runtime identity/capability drift".into());
    }
    let mounts = std::fs::read_to_string("/proc/mounts")?;
    if !mounts.lines().any(|l| {
        let fields: Vec<_> = l.split_whitespace().collect();
        fields.get(1) == Some(&"/")
            && fields
                .get(3)
                .is_some_and(|flags| flags.split(',').any(|f| f == "ro"))
    }) {
        return Err("root filesystem writable".into());
    }
    for entry in std::fs::read_dir("/sys/class/net")? {
        if entry?.file_name() != "lo" {
            return Err("external network present".into());
        }
    }
    if std::fs::read_to_string("/sys/fs/cgroup/pids.max")?.trim() != "64"
        || std::fs::read_to_string("/sys/fs/cgroup/memory.max")?.trim() != "1073741824"
    {
        return Err("runtime resource bound drift".into());
    }
    let cpu = std::fs::read_to_string("/sys/fs/cgroup/cpu.max")?;
    if !cpu_valid(&cpu) {
        return Err("runtime CPU bound drift".into());
    }
    Ok(())
}
fn check(owner: &str) -> Result<(), Error> {
    if !matches!(owner, "python" | "rust") {
        return Err("fixed owner required".into());
    }
    isolate()?;
    let process = std::fs::read("/proc/1/cmdline")?;
    let command = String::from_utf8(process)?;
    if (owner == "python" && !command.starts_with("python\0server.py\0"))
        || (owner == "rust"
            && !command
                .split('\0')
                .next()
                .is_some_and(|s| s.ends_with("noevia-docling")))
    {
        return Err("selected/default owner not PID1".into());
    }
    let health = response(request(None)?)?;
    if !health_valid(&health)? {
        return Err("health failed".into());
    }
    let first = pid(&response(request(Some(b"x"))?)?)?;
    let error = response(request(Some(b"BOOM"))?)?;
    if error.status != 422
        || compact_json(&error.body)?
            != r#"{"error":"This document could not be read within its processing limits; the original is retained."}"#
    {
        return Err("conversion error contract failed".into());
    }
    if pid(&response(request(Some(b"x"))?)?)? != first {
        return Err("ordinary error lost warm worker".into());
    }
    for half in [false, true] {
        let old = pid(&response(request(Some(b"x"))?)?)?;
        let notice = UdpSocket::bind("127.0.0.1:18931")?;
        notice.set_read_timeout(Some(Duration::from_secs(5)))?;
        let mut socket = request(Some(b"SLEEP 30"))?;
        let mut buffer = [0u8; 64];
        let count = notice.recv(&mut buffer)?;
        if buffer.get(..count) != Some(b"started".as_slice()) {
            return Err("actual conversion start not observed".into());
        }
        if response(request(Some(b"x"))?)?.status != 503 {
            return Err("single conversion admission failed".into());
        }
        if half {
            socket.shutdown(Shutdown::Write)?;
            let mut bytes = [0u8; 1];
            if socket.read(&mut bytes)? != 0 {
                return Err("cancelled half-close replied".into());
            }
        } else {
            drop(socket);
            std::thread::sleep(Duration::from_millis(500));
        }
        absent_worker(old)?;
        temporary_inputs_clean()?;
        if pid(&response(request(Some(b"x"))?)?)? == old {
            return Err("cancelled worker reused".into());
        }
    }
    temporary_inputs_clean()?;
    println!(
        "synthetic image checks passed: owner={owner}, warm/error/full-close/half-close/isolation"
    );
    Ok(())
}
fn main() -> Result<(), Error> {
    let owner = std::env::args().nth(1).ok_or("fixed owner required")?;
    check(&owner)
}

#[cfg(test)]
mod tests {
    use super::*;
    const DOCKER: &str = include_str!("../Dockerfile");
    #[test]
    fn frozen_engine_layers_and_fixed_entrypoint_are_preserved() {
        let layers=DOCKER.split_once("FROM python:3.12-slim-bookworm AS production\n").and_then(|(_,s)|s.split_once("COPY --from=native-docling /out/noevia-docling /usr/local/bin/noevia-docling\n").map(|(s,_)|s.trim_start_matches('\n'))).expect("production stages");
        assert_eq!(
            layers.trim_end(),
            include_str!("fixtures/engine-layers.dockerfile").trim_end()
        );
        assert_eq!(DOCKER.matches("CMD [\"noevia-docling\"]").count(), 2);
        assert!(DOCKER.contains("--features native-docling"));
        assert!(DOCKER.contains("cp target/release/noevia-docling /out/noevia-docling-omission"));
        assert!(DOCKER.contains("FROM native-contract AS omission-contract\nCOPY --from=native-docling /out/noevia-docling-omission /usr/local/bin/noevia-docling"));
    }
    #[test]
    fn actual_overlay_guard_counts_only_production_copies() {
        let directory =
            std::env::temp_dir().join(format!("docling-overlay-guard-{}", std::process::id()));
        std::fs::create_dir_all(directory.join("services/docling")).expect("owned fixture");
        std::fs::create_dir_all(directory.join("docs")).expect("owned docs");
        std::fs::write(directory.join("docs/deployment.md"), "COPY server.py extract.py isolation.py isolated_worker.py selftest.py synthetic_pdfs.py /app/\n").expect("overlay docs");
        let loader = "import runpy,sys; from pathlib import Path; g=runpy.run_path(sys.argv[1]); f=g['test_overlay_docs_list_exactly_the_files_the_dockerfile_copies_to_app']; f.__globals__['HERE']=Path(sys.argv[2]); f()";
        let test = std::path::Path::new(file!())
            .parent()
            .expect("test directory")
            .join("../test_requirements_pinned.py");
        let production_copy = "COPY server.py extract.py isolation.py isolated_worker.py selftest.py synthetic_pdfs.py ./";
        let controls = [
            (DOCKER.to_owned(), true),
            (
                DOCKER.replace(production_copy, "# missing production copy"),
                false,
            ),
            (
                DOCKER.replace(
                    production_copy,
                    &format!("{production_copy}\n{production_copy}"),
                ),
                false,
            ),
            (
                DOCKER.replace(
                    "COPY tests/fixtures/native_extract.py ./extract.py",
                    "COPY tests/fixtures/native_extract.py ./extract.py\nCOPY synthetic_only.py ./",
                ),
                true,
            ),
        ];
        for (docker, valid) in controls {
            std::fs::write(directory.join("services/docling/Dockerfile"), docker)
                .expect("fixture Docker");
            let answer = std::process::Command::new("python3")
                .args(["-B", "-c", loader])
                .arg(&test)
                .arg(directory.join("services/docling"))
                .output()
                .expect("actual Python guard");
            assert_eq!(
                answer.status.success(),
                valid,
                "{}",
                String::from_utf8_lossy(&answer.stderr)
            );
            if !valid {
                assert!(String::from_utf8_lossy(&answer.stderr)
                    .contains("AssertionError: expected one `COPY"));
            }
        }
        std::fs::remove_dir_all(directory).expect("owned cleanup");
    }
    #[test]
    fn all_four_immutable_runtime_pins_match() {
        let expected_ref = "ARG NOEVIA_RS_REF=219b874b63f24ea1882514fd8bcabd5895045a98";
        let expected_hash =
            "ARG NOEVIA_RS_SHA256=da511923e90364a63163fc720a85b7ec5fbf1cb125c8bfbf7a5dc90713b96de1";
        let valid = |docker: &str| {
            docker.lines().any(|l| l == expected_ref)
                && docker.lines().any(|l| l == expected_hash)
                && docker.contains("sha256sum -c -")
        };
        for docker in [
            DOCKER,
            include_str!("../../ocr/Dockerfile"),
            include_str!("../../diary/Dockerfile"),
            include_str!("../../model-manager/Dockerfile"),
        ] {
            assert!(valid(docker));
            assert!(!valid(&docker.replace(
                expected_ref,
                "ARG NOEVIA_RS_REF=174078a20c214a609085bc44c1d668cd18a634eb"
            )));
            assert!(!valid(&docker.replace(expected_hash, "ARG NOEVIA_RS_SHA256=edeab963f31832122a303999fcd5acbc46132a7b15e93f50758066c04ddf2773")));
            assert!(!valid(
                &docker.replace("sha256sum -c -", "# verification omitted")
            ));
        }
    }
    #[test]
    fn actual_ci_requires_images_omission_probe_and_no_heavy_target() {
        let workflow = include_str!("../../.github/workflows/ci.yml");
        let job = workflow
            .split_once("  docling:\n")
            .and_then(|(_, s)| s.split_once("\n  ocr:\n").map(|(s, _)| s))
            .expect("Docling job");
        for line in [
            "target: native-contract",
            "target: omission-contract",
            "rustc --edition=2021 --test tests/native_image.rs",
            "working-directory: self/docling",
            "--wait --wait-timeout 60 --no-build",
            "exec -T python native-image-probe python",
            "exec -T rust native-image-probe rust",
            "native-docling feature required",
        ] {
            assert!(job.contains(line), "required image gate missing: {line}");
        }
        assert!(!job.contains("target: production"));
        assert!(!job.lines().any(|l| l.trim_start().starts_with("if:")));
        let compose = include_str!("native-runtime.compose.yml");
        for literal in [
            "network_mode: none",
            "read_only: true",
            "pids_limit: 64",
            "mem_limit: 1g",
            "cpus: 1",
            "no-new-privileges:true",
        ] {
            assert_eq!(compose.matches(literal).count(), 2);
        }
        assert_eq!(compose.matches("NOEVIA_DOCLING_IMPL: rust").count(), 1);
        assert!(!compose
            .split_once("  rust:\n")
            .expect("native service")
            .0
            .contains("NOEVIA_DOCLING_IMPL"));
    }
    #[test]
    fn fixed_test_adapter_formats_match_actual_engine_source() {
        fn formats(source: &str) -> std::collections::BTreeSet<&str> {
            let literal = source
                .split_once("SUPPORTED = {")
                .expect("format declaration")
                .1
                .split_once('}')
                .expect("format end")
                .0;
            literal
                .split('"')
                .enumerate()
                .filter_map(|(i, part)| (i % 2 == 1).then_some(part))
                .collect()
        }
        assert_eq!(
            formats(include_str!("fixtures/native_extract.py")),
            formats(include_str!("../extract.py"))
        );
    }
    #[test]
    fn fixed_shapes_and_resource_bounds_refuse_partial_proofs() {
        assert!(cpu_valid("100000 100000\n"));
        assert_eq!(pid(&Reply { status: 200, body: r#"{"pages":[{"number":1,"text":"ok","status":"native","method":"docling","truncated":false}],"total":1,"truncatedPages":false,"pid":42}"#.into() }).expect("complete extraction"), 42);
        assert!(health_valid(&Reply { status: 200, body: r#"{"service":"docling","formats":[".bmp",".csv",".docx",".epub",".htm",".html",".jpeg",".jpg",".odp",".ods",".odt",".pdf",".png",".pptx",".tif",".tiff",".xlsx"],"ranges":true,"rangedMaxBytes":100663296}"#.into() }).expect("complete health"));
        assert!(health_valid(&Reply { status: 200, body: r#"{"formats":[".bmp",".csv",".docx",".epub",".htm",".html",".jpeg",".jpg",".odp",".ods",".odt",".pdf",".png",".pptx",".tif",".tiff",".xlsx"],"rangedMaxBytes":100663296,"ranges":true,"service":"docling"}"#.into() }).expect("native health"));
        // noevia#1867: an image that does not advertise page ranges is stale.
        assert!(!health_valid(&Reply { status: 200, body: r#"{"service":"docling","formats":[".bmp",".csv",".docx",".epub",".htm",".html",".jpeg",".jpg",".odp",".ods",".odt",".pdf",".png",".pptx",".tif",".tiff",".xlsx"]}"#.into() }).expect("pre-range health"));
        // #1711: .md is no longer advertised; an image still listing it is stale.
        assert!(!health_valid(&Reply { status: 200, body: r#"{"service":"docling","formats":[".bmp",".csv",".docx",".epub",".htm",".html",".jpeg",".jpg",".md",".odp",".ods",".odt",".pdf",".png",".pptx",".tif",".tiff",".xlsx"]}"#.into() }).expect("stale health"));
        for bad in ["", "max 100000", "0 0", "100000 200000", "1 1 extra"] {
            assert!(!cpu_valid(bad));
        }
        assert!(pid(&Reply {
            status: 200,
            body: r#"{"pages":[],"text":"ok","pid":42}"#.into()
        })
        .is_err());
        assert!(!health_valid(&Reply {
            status: 200,
            body: r#"{"service":"docling"}"#.into()
        })
        .expect("shape"));
        assert_eq!(
            compact_json(" { \"text\": \"two words\" } ").expect("shape"),
            r#"{"text":"two words"}"#
        );
    }
    #[test]
    fn truncated_response_is_refused() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("loopback");
        let address = listener.local_addr().expect("address");
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("client");
            socket.write_all(b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\nContent-Type: application/json\r\n\r\nx").expect("reply");
        });
        assert!(response(TcpStream::connect(address).expect("connect")).is_err());
        server.join().expect("server");
    }
}
