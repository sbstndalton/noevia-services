//! Isolated checkout only: actual Python source mutations must reach named Rust assertions.
#![forbid(unsafe_code)]
#[path = "prompts.rs"]
#[allow(dead_code)]
mod guards;
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
struct Restore {
    path: PathBuf,
    bytes: Vec<u8>,
}
impl Drop for Restore {
    fn drop(&mut self) {
        let _ = fs::write(&self.path, &self.bytes);
    }
}
fn mutation(
    root: &Path,
    binary: &Path,
    file: &str,
    before: &str,
    after: &str,
    test: &str,
) -> Result<()> {
    let path = root.join(file);
    let bytes = fs::read(&path)?;
    let original = std::str::from_utf8(&bytes)?;
    if original.matches(before).count() != 1 {
        return Err("nonunique source mutation target".into());
    }
    let changed = original.replace(before, after);
    let restore = Restore {
        path: path.clone(),
        bytes,
    };
    fs::write(&path, changed)?;
    let mut command = Command::new(binary);
    command.args([&format!("tests::{test}"), "--exact", "--nocapture"]);
    let result = guards::capture_output(command, Duration::from_secs(30));
    fs::write(&restore.path, &restore.bytes)?;
    drop(restore);
    let result = result?;
    let stdout = String::from_utf8(result.stdout)?;
    let stderr = String::from_utf8(result.stderr)?;
    let policy = match test {
        "actual_direct_writers_are_guarded_before_database" => {
            "all direct prompt writers refuse exact confirmation"
        }
        "actual_startup_seed_has_exact_confirmation_and_preserves_hooks" => {
            "startup seed ownership and other hooks"
        }
        "actual_private_prompt_http_is_authenticated_and_method_aware" => {
            "auth-first exact three method/path pairs"
        }
        _ => return Err("unregistered mutation test".into()),
    };
    if !intended_failure(result.status.code(), &stdout, &stderr, test, policy) {
        return Err("mutation did not reach required Rust policy assertion".into());
    }
    println!("detected {file}: {test}");
    Ok(())
}
fn intended_failure(code: Option<i32>, out: &str, err: &str, test: &str, policy: &str) -> bool {
    let selected = format!("tests::{test}");
    let source = include_str!("prompts.rs");
    let line = source
        .lines()
        .position(|line| line.contains("panic!(\"PROMPTS_POLICY_ASSERT:"))
        .map(|line| line + 1);
    let Some(line) = line else {
        return false;
    };
    let location = format!("prompts.rs:{line}:");
    code == Some(101)
        && out.contains("running 1 test")
        && out.contains("0 passed; 1 failed;")
        && out
            .lines()
            .filter(|line| line.starts_with("    tests::"))
            .eq(std::iter::once(format!("    {selected}").as_str()))
        && (out.contains(&format!("test {selected} ... FAILED"))
            || out.contains(&format!("{selected} --- FAILED")))
        && err.matches("panicked at").count() == 1
        && err.contains(&format!("thread '{selected}'"))
        && err.contains(&location)
        && err
            .lines()
            .filter(|line| line.starts_with("PROMPTS_POLICY_ASSERT:"))
            .eq(std::iter::once(
                format!("PROMPTS_POLICY_ASSERT:{policy}").as_str(),
            ))
        && !out.contains("error[E")
        && !err.contains("error[E")
        && !err.contains("PROMPTS_INFRASTRUCTURE_FAILURE")
}
#[cfg(test)]
mod classifier_tests {
    #[test]
    fn exact_policy_rejects_infrastructure_and_other_assertions() {
        let name = "actual_direct_writers_are_guarded_before_database";
        let policy = "all direct prompt writers refuse exact confirmation";
        let line = include_str!("prompts.rs")
            .lines()
            .position(|l| l.contains("panic!(\"PROMPTS_POLICY_ASSERT:"))
            .unwrap()
            + 1;
        let out=format!("running 1 test\ntest tests::{name} ... FAILED\nfailures:\n    tests::{name}\ntest result: FAILED. 0 passed; 1 failed;");
        let err=format!("thread 'tests::{name}' (123) panicked at qualification/prompts.rs:{line}:13:\nPROMPTS_POLICY_ASSERT:{policy}\n");
        assert!(super::intended_failure(Some(101), &out, &err, name, policy));
        for bad in [
            err.replace(policy, "other policy"),
            err.replace("PROMPTS_POLICY_ASSERT", "PROMPTS_INFRASTRUCTURE_FAILURE"),
            err.replace(&format!(":{line}:"), ":1:"),
            format!("{err}thread 'other' panicked at setup.rs:1:1\n"),
            format!("{err}error[E0001]: compile\n"),
        ] {
            assert!(!super::intended_failure(
                Some(101),
                &out,
                &bad,
                name,
                policy
            ));
        }
        for code in [None, Some(1), Some(0)] {
            assert!(!super::intended_failure(code, &out, &err, name, policy));
        }
        assert!(!super::intended_failure(
            Some(101),
            &out.replace(name, "wrong_test"),
            &err,
            name,
            policy
        ));
        for bad in [
            "spawn failed",
            "capture cap",
            "deadline",
            "called Result::unwrap() on Err",
        ] {
            assert!(!super::intended_failure(Some(101), &out, bad, name, policy));
        }
    }
}
fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let binary =
        PathBuf::from(args.next().ok_or("compiled guard tests required")?).canonicalize()?;
    if args.next().is_some() {
        return Err("unexpected mutation arguments".into());
    }
    let source = Path::new(file!()).canonicalize()?;
    let root = source
        .parent()
        .and_then(Path::parent)
        .ok_or("source root")?;
    let guard="    if os.environ.get(\"NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED\") == \"1\":\n        raise RuntimeError(\"Native model manager owns prompt writes\")\n";
    for definition in ["def add_prompt(name: str, body: str) -> int:\n","def delete_prompt(pid: int) -> bool:\n","def seed_bench_prompts() -> int:\n    \"\"\"Add the built-in benchmark prompts once. Returns how many were added.\"\"\"\n"] {
        let before=format!("{definition}{guard}");mutation(root,&binary,"app/db.py",&before,definition,"actual_direct_writers_are_guarded_before_database")?;
    }
    mutation(root,&binary,"app/db.py","if key == _BENCH_SEED_KEY and os.environ.get(\"NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED\") == \"1\":","if False and key == _BENCH_SEED_KEY and os.environ.get(\"NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED\") == \"1\":","actual_direct_writers_are_guarded_before_database")?;
    mutation(
        root,
        &binary,
        "app/main.py",
        "if os.environ.get(\"NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED\") != \"1\":",
        "if os.environ.get(\"NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED\") == \"1\":",
        "actual_startup_seed_has_exact_confirmation_and_preserves_hooks",
    )?;
    mutation(
        root,
        &binary,
        "app/main.py",
        "if os.environ.get(\"NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED\") == \"1\":",
        "if False and os.environ.get(\"NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED\") == \"1\":",
        "actual_private_prompt_http_is_authenticated_and_method_aware",
    )?;
    mutation(root,&binary,"app/main.py","if not hmac.compare_digest(sent.encode(), token.encode()):","if request.url.path.startswith('/api/v1/prompts') is False and not hmac.compare_digest(sent.encode(), token.encode()):","actual_private_prompt_http_is_authenticated_and_method_aware")?;
    Ok(())
}
