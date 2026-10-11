//! Actual Python HTTP ownership guard qualification with synthetic dependencies only.
#![forbid(unsafe_code)]
#[path = "prompts.rs"]
#[allow(dead_code)]
mod capture;
use std::{
    error::Error,
    fs,
    os::unix::fs::DirBuilderExt,
    path::Path,
    process::Command,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const KEY: &str = "NOEVIA_NATIVE_MODEL_MANAGER_DIAGNOSE_CONFIRMED";
// No startup, Docker, hardware sampling, inference, sockets or subprocesses in this seam.
// Real middleware/router/diagnosis execute; acquisition collaborators are fixed fakes.
const SEAM: &str = r#"
import json,sys
from app import main,api,services
from fastapi.testclient import TestClient
seen=[]
def names():
 seen.append('names')
 return ['synthetic','broken']
def logs(name):
 seen.append('logs')
 if name=='broken':raise RuntimeError('synthetic failure')
 return ''
services._effective_container_names=names
api._logs_since_start=logs
client=TestClient(main.app,raise_server_exceptions=False)
for method,path,authorized in json.loads(sys.argv[1]):
 seen.clear()
 r=client.request(method,path,headers={'X-Model-Loader-Token':'synthetic-diagnosis-token-0123456789abcdef' if authorized else 'wrong'},follow_redirects=False)
 print(json.dumps([r.status_code,r.text,seen],separators=(',',':')))
"#;
struct Root(std::path::PathBuf);
impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn call(value: Option<&str>, composed: bool, cases: &str) -> Result<Vec<String>> {
    let tick = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root =
        Root(std::env::temp_dir().join(format!("diagnosis-guard-{}-{tick}", std::process::id())));
    fs::DirBuilder::new().mode(0o700).create(&root.0)?;
    let source = Path::new(file!()).canonicalize()?;
    let app = source
        .parent()
        .and_then(Path::parent)
        .ok_or("qualification source root")?;
    let mut command = Command::new(
        std::env::var_os("DIAGNOSIS_TEST_PYTHON").ok_or("DIAGNOSIS_TEST_PYTHON required")?,
    );
    command
        .args(["-B", "-c", SEAM, cases])
        .current_dir(app)
        .env("MODELS_DIR", root.0.join("models"))
        .env("MODELS_INI_PATH", root.0.join("models/models.ini"))
        .env("DATA_DIR", root.0.join("data"))
        .env(
            "MODEL_LOADER_TOKEN",
            "synthetic-diagnosis-token-0123456789abcdef",
        )
        .env("LLAMA_CONTAINERS", "")
        .env("DOCKER_HOST", "unix:///absent-synthetic.sock")
        .env("MIGRATE_CACHE_RAM_ON_START", "false")
        .env("GGUF_PARSER", "python")
        .env("MODEL_FILES_IMPL", "python")
        .env("MODEL_AUTOCONFIG", "python")
        .env_remove(KEY);
    for key in [
        "NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED",
        "NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED",
        "NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED",
        "NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED",
    ] {
        command.env(key, if composed { "1" } else { "0" });
    }
    if let Some(value) = value {
        command.env(KEY, value);
    }
    let answers = String::from_utf8(capture::capture(command, Duration::from_secs(12))?)?;
    if root.0.join("data").exists() || root.0.join("models").exists() {
        return Err("qualification unexpectedly accessed state".into());
    }
    Ok(answers.lines().map(str::to_owned).collect())
}
fn require(ok: bool, message: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn exact_confirmation() -> Result<()> {
    let cases = r#"[["GET","/api/v1/backends/synthetic/diagnose",true],["GET","/api/v1/backends/unknown/diagnose",true],["GET","/api/v1/backends/broken/diagnose",true]]"#;
    for value in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some(" 1"),
        Some("1 "),
        Some("yes"),
        Some("1"),
    ] {
        let answers = call(value, false, cases)?;
        let expected = if value == Some("1") {
            vec![r#"[503,"{\"detail\":\"Native model manager owns this endpoint\"}",[]]"#; 3]
        } else {
            vec![
                r#"[200,"{\"failures\":[]}",["names","logs"]]"#,
                r#"[404,"{\"detail\":\"unknown backend\"}",["names"]]"#,
                r#"[500,"Internal Server Error",["names","logs"]]"#,
            ]
        };
        require(
            answers == expected,
            "exact diagnosis confirmation before discovery and errors",
        )?;
    }
    Ok(())
}
fn auth_and_paths() -> Result<()> {
    let cases = r#"[["GET","/api/v1/backends/synthetic/diagnose",false],["GET","/api/v1/backends/unknown/diagnose",false],["GET","/api/v1/backends/synthetic/diagnose?synthetic=1",true],["GET","/api/v1/backends/%73ynthetic/diagnose",true],["POST","/api/v1/backends/synthetic/diagnose",true],["HEAD","/api/v1/backends/synthetic/diagnose",true],["GET","/api/v1/backends//diagnose",true],["GET","/api/v1/backends/synthetic/extra/diagnose",true],["GET","/api/v1/backends/synthetic/diagnose/extra",true],["GET","/api/v1/backends/synthetic/diagnose/",true]]"#;
    let answers = call(Some("1"), false, cases)?;
    require(answers.len() == 10, "qualification answer count")?;
    require(
        answers
            .iter()
            .take(2)
            .all(|a| a == r#"[401,"{\"error\":\"unauthorized\"}",[]]"#),
        "token precedes diagnosis ownership",
    )?;
    require(
        answers
            .iter()
            .skip(2)
            .take(2)
            .all(|a| a == r#"[503,"{\"detail\":\"Native model manager owns this endpoint\"}",[]]"#),
        "decoded exact path and query ownership",
    )?;
    for (index, status) in [(4, 405), (5, 405), (6, 404), (7, 404), (8, 404), (9, 307)] {
        require(
            answers
                .get(index)
                .is_some_and(|a| a.starts_with(&format!("[{status},")) && a.ends_with(",[]]")),
            "unowned methods and path shapes do not acquire",
        )?;
    }
    Ok(())
}
fn composed_guards() -> Result<()> {
    let cases = r#"[["GET","/api/v1/health",true],["GET","/api/v1/host",true],["GET","/api/v1/sections",true],["GET","/api/v1/sections/synthetic",true],["GET","/api/v1/prompts",true],["POST","/api/v1/prompts",true],["DELETE","/api/v1/prompts/1",true],["GET","/api/v1/backends/synthetic/logs",true],["GET","/api/v1/backends/synthetic/diagnose",true]]"#;
    let answers = call(Some("1"), true, cases)?;
    require(
        answers.len() == 9
            && answers.iter().all(|a| {
                a == r#"[503,"{\"detail\":\"Native model manager owns this endpoint\"}",[]]"#
            }),
        "all existing ownership guards compose",
    )
}
fn main() -> Result<()> {
    exact_confirmation()?;
    auth_and_paths()?;
    composed_guards()?;
    println!("actual diagnosis ownership qualification passed");
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn actual_exact_confirmation_preserves_default_errors_and_unknowns() {
        super::exact_confirmation().unwrap();
    }
    #[test]
    fn actual_token_precedes_method_and_path_ownership() {
        super::auth_and_paths().unwrap();
    }
    #[test]
    fn actual_diagnosis_composes_with_existing_guards() {
        super::composed_guards().unwrap();
    }
}
