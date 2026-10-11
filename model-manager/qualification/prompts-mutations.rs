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
fn registered_policy(test: &str) -> Option<&'static str> {
    match test {
        "actual_direct_writers_are_guarded_before_database" => {
            Some("all direct prompt writers refuse exact confirmation")
        }
        "actual_startup_seed_has_exact_confirmation_and_preserves_hooks" => {
            Some("startup seed ownership and other hooks")
        }
        "actual_private_prompt_http_is_authenticated_and_method_aware" => {
            Some("auth-first exact three method/path pairs")
        }
        _ => None,
    }
}
fn policy_location() -> Option<String> {
    let mut sites = include_str!("prompts.rs")
        .lines()
        .enumerate()
        .filter_map(|(line, text)| {
            text.find("panic!(\"PROMPTS_POLICY_ASSERT:")
                .map(|column| (line + 1, column + 1))
        });
    let (line, column) = sites.next()?;
    if sites.next().is_some() {
        return None;
    }
    // Both original rustc commands use the same qualification directory. Bind the
    // whole sibling compile-time path, including an absolute prefix when supplied.
    let path = Path::new(file!()).with_file_name("prompts.rs");
    Some(format!("{}:{line}:{column}", path.to_str()?))
}
fn decimal_duration(value: &str) -> bool {
    let Some((whole, fraction)) = value.split_once('.') else {
        return false;
    };
    !whole.is_empty()
        && whole.bytes().all(|b| b.is_ascii_digit())
        && fraction.len() == 2
        && fraction.bytes().all(|b| b.is_ascii_digit())
        && value.parse::<f64>().is_ok_and(f64::is_finite)
}
fn intended_failure(code: Option<i32>, out: &str, err: &str, test: &str, policy: &str) -> bool {
    if code != Some(101) || registered_policy(test) != Some(policy) {
        return false;
    }
    let Some(location) = policy_location() else {
        return false;
    };
    let selected = format!("tests::{test}");
    // Rust 1.99 pretty libtest with one --exact --nocapture test: the first
    // failures section is empty because its panic was already sent to stderr.
    let prefix = format!(
        "\nrunning 1 test\ntest {selected} ... FAILED\n\nfailures:\n\nfailures:\n    {selected}\n\ntest result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4 filtered out; finished in "
    );
    let Some(duration) = out
        .strip_prefix(&prefix)
        .and_then(|s| s.strip_suffix("s\n\n"))
    else {
        return false;
    };
    if !decimal_duration(duration) {
        return false;
    }
    let panic_prefix = format!("\nthread '{selected}' (");
    let Some(panic) = err.strip_prefix(&panic_prefix) else {
        return false;
    };
    let Some((pid, tail)) = panic.split_once(") panicked at ") else {
        return false;
    };
    !pid.is_empty()
        && pid.bytes().all(|b| b.is_ascii_digit())
        && pid.parse::<u64>().is_ok_and(|pid| pid != 0)
        && tail == format!(
            "{location}:\nPROMPTS_POLICY_ASSERT:{policy}\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n"
        )
}
#[cfg(test)]
mod classifier_tests {
    fn fixture(name: &str, policy: &str) -> (String, String) {
        let selected = format!("tests::{name}");
        let location = super::policy_location().unwrap();
        (
            format!("\nrunning 1 test\ntest {selected} ... FAILED\n\nfailures:\n\nfailures:\n    {selected}\n\ntest result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4 filtered out; finished in 0.00s\n\n"),
            format!("\nthread '{selected}' (123) panicked at {location}:\nPROMPTS_POLICY_ASSERT:{policy}\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n"),
        )
    }
    #[test]
    fn exact_policy_rejects_infrastructure_and_other_assertions() {
        // These pure fixtures are libtest grammar controls, not actual mutant captures.
        assert_eq!(
            include_str!("prompts.rs").matches("    #[test]\n").count(),
            5
        );
        for name in [
            "actual_direct_writers_are_guarded_before_database",
            "actual_startup_seed_has_exact_confirmation_and_preserves_hooks",
            "actual_private_prompt_http_is_authenticated_and_method_aware",
        ] {
            let policy = super::registered_policy(name).unwrap();
            let (out, err) = fixture(name, policy);
            assert!(super::intended_failure(Some(101), &out, &err, name, policy));
            assert!(super::intended_failure(
                Some(101),
                &out.replace("0.00s", "12.34s"),
                &err.replace("(123)", "(8157242)"),
                name,
                policy
            ));
            let location = super::policy_location().unwrap();
            let (path_line, column) = location.rsplit_once(':').unwrap();
            let (path, line) = path_line.rsplit_once(':').unwrap();
            for bad in [
                format!("prefix\n{err}"), format!("{err}tail\n"), format!("{err}{err}"),
                err.replace(policy, "other policy"),
                err.replace("PROMPTS_POLICY_ASSERT", "PROMPTS_INFRASTRUCTURE_FAILURE"),
                err.replace(&location, &format!("{path_line}:{}", column.parse::<usize>().unwrap() + 1)),
                err.replace(&location, &format!("{path}:{}:{column}", line.parse::<usize>().unwrap() + 1)),
                err.replace(&location, &format!("arbitrary/{location}")),
                err.replace(&location, &format!("prompts.rs:{line}:{column}")),
                err.replace(name, "other_test"), err.replace("(123)", "(0)"),
                err.replace("(123)", "(-1)"), err.replace("(123)", "(1x)"),
                err.replace("(123)", "(18446744073709551616)"),
                err.replace("note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n", ""),
                err.trim_start().to_owned(), err.trim_end().to_owned(),
                format!("{err}thread 'other' panicked at setup.rs:1:1\n"),
                format!("{err}error[E0001]: compile\n"),
                "spawn failed".into(), "capture cap".into(), "deadline".into(),
                "called Result::unwrap() on Err".into(),
            ] {
                assert!(!super::intended_failure(Some(101), &out, &bad, name, policy), "accepted stderr {bad:?}");
            }
            for bad in [
                format!("prefix\n{out}"),
                format!("{out}tail\n"),
                format!("{out}{out}"),
                out.replace("running 1 test", "running 2 tests"),
                out.replace(name, "other_test"),
                out.replace(" ... FAILED", " --- FAILED"),
                out.replace(&format!("    tests::{name}\n"), ""),
                out.replace(
                    &format!("    tests::{name}\n"),
                    &format!("    tests::{name}\n    tests::other\n"),
                ),
                out.replace("test result: FAILED.", "test result: ok."),
                out.replace("0 passed; 1 failed;", "1 passed; 0 failed;"),
                out.replace("4 filtered out", "5 filtered out"),
                out.split("test result:").next().unwrap().to_owned(),
                out.trim_start().to_owned(),
                out.trim_end().to_owned(),
                out.replace("0.00s", "NaNs"),
                out.replace("0.00s", "infs"),
                out.replace("0.00s", "-0.01s"),
                out.replace("0.00s", "1e2s"),
                out.replace("0.00s", "0.0s"),
                out.replace("0.00s", ".00s"),
                out.replace("0.00s", "0.00junk s"),
                out.replace("0.00s", &format!("{}.00s", "9".repeat(400))),
                format!("{out}error[E0001]: compile\n"),
                format!("{out}capture cap\n"),
                format!("{out}deadline\n"),
            ] {
                assert!(
                    !super::intended_failure(Some(101), &bad, &err, name, policy),
                    "accepted stdout {bad:?}"
                );
            }
            for code in [None, Some(1), Some(0), Some(102)] {
                assert!(!super::intended_failure(code, &out, &err, name, policy));
            }
            assert!(!super::intended_failure(
                Some(101),
                &out,
                &err,
                name,
                "other policy"
            ));
        }
        let (out, err) = fixture("unregistered", "unregistered policy");
        assert!(!super::intended_failure(
            Some(101),
            &out,
            &err,
            "unregistered",
            "unregistered policy"
        ));
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
