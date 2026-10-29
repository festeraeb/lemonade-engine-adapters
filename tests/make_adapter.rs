//! Port of tests/test_make_adapter.py: spec validation and the generated backend files.
use make_adapter::{register_cmake, render};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn base_spec() -> Value {
    serde_json::from_str(&std::fs::read_to_string(root().join("engines/freetoken.json")).unwrap()).unwrap()
}

/// Copy of the freetoken spec with fields replaced; `Null` removes the field (Python `bad()`).
fn bad(changes: Vec<(&str, Value)>) -> Value {
    let mut s = base_spec();
    for (k, v) in changes {
        if v.is_null() {
            s.as_object_mut().unwrap().remove(k);
        } else {
            s[k] = v;
        }
    }
    s
}

#[test]
fn shipped_specs_are_valid() {
    let mut n = 0;
    for e in std::fs::read_dir(root().join("engines")).unwrap() {
        let p = e.unwrap().path();
        if p.extension().map_or(false, |x| x == "json") {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            render(&v).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            n += 1;
        }
    }
    assert!(n >= 2);
}

#[test]
fn rejects() {
    let cases: Vec<(&str, Value)> = vec![
        ("missing field", bad(vec![("identity", Value::Null)])),
        ("no {model}", bad(vec![("args_template", json!("serve --port {port}"))])),
        ("shell syntax", bad(vec![("args_template", json!("serve --model {model} --port {port} | tee log"))])),
        ("unknown placeholder", bad(vec![("args_template", json!("serve --model {model} --port {port} --ctx {ctx}"))])),
        ("binary with path", bad(vec![("binary", json!("/usr/bin/ft"))])),
        ("bad recipe", bad(vec![("recipe", json!("Free_Token"))])),
        ("bad device family", bad(vec![("support", json!([{"os": ["linux"], "devices": {"cuda": []}}]))])),
        (
            "ready field without values",
            bad(vec![("ready", json!({"path": "/health", "timeout_s": 60, "json_field": "status"}))]),
        ),
        ("marker on a file", bad(vec![("checkpoint", json!({"kind": "file", "marker": "x"}))])),
        ("bad env name", bad(vec![("env", json!({"cuda visible": "{device}"}))])),
    ];
    for (name, spec) in cases {
        assert!(render(&spec).is_err(), "case {name} was accepted");
    }
}

fn files() -> std::collections::BTreeMap<String, String> {
    render(&base_spec()).unwrap().into_iter().collect()
}

#[test]
fn three_files_under_the_stem() {
    let keys: Vec<String> = files().into_keys().collect();
    assert_eq!(
        keys,
        vec![
            "src/cpp/include/lemon/backends/freetoken/freetoken.h",
            "src/cpp/include/lemon/backends/freetoken/freetoken_server.h",
            "src/cpp/server/backends/freetoken/freetoken_server.cpp",
        ]
    );
}

#[test]
fn descriptor_uses_the_2026_43_support_row() {
    let f = files();
    let h = &f["src/cpp/include/lemon/backends/freetoken/freetoken.h"];
    assert!(h.contains(r#"{"system", BackendTier::Experimental, BackendFormat::Native, {"linux"},"#));
    assert!(h.contains(r#"/*binary*/          "ft""#));
    assert!(!h.contains("/*experimental*/")); // 2026.43.x has no such field (it broke burn.h)
}

#[test]
fn server_checks_readiness_and_identity() {
    let f = files();
    let cpp = &f["src/cpp/server/backends/freetoken/freetoken_server.cpp"];
    assert!(cpp.contains(r#"kOwnedBy = "FreeToken""#));
    assert!(cpp.contains(r#"kReadyField = "status""#));
    assert!(cpp.contains(r#"kReadyValues = {"ok"}"#));
    assert!(cpp.contains(r#"{"CUDA_VISIBLE_DEVICES", "{device}"}"#));
    assert!(cpp.contains("start_server("));
    assert!(cpp.contains("stop_server();"));
    assert!(!cpp.contains("set_process_handle")); // the 11.9 API that does not build on 2026.43.x
}

#[test]
fn download_accepts_an_existing_checkpoint() {
    let f = files();
    let cpp = &f["src/cpp/server/backends/freetoken/freetoken_server.cpp"];
    assert!(cpp.contains("if (is_downloaded(info, ctx)) {"));
}

#[test]
fn register_is_idempotent() {
    let d = std::env::temp_dir().join(format!("make-adapter-test-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let cm = d.join("CMakeLists.txt");
    std::fs::write(&cm, "x\nset(LEMON_BACKENDS\n    # comment\n    \"llamacpp|llamacpp\"\n)\ny\n").unwrap();
    assert!(register_cmake(Path::new(&d), "freetoken", "freetoken").unwrap());
    assert!(!register_cmake(Path::new(&d), "freetoken", "freetoken").unwrap());
    let t = std::fs::read_to_string(&cm).unwrap();
    assert!(t.contains("    \"llamacpp|llamacpp\"\n    \"freetoken|freetoken\"\n)"));
    std::fs::remove_dir_all(&d).ok();
}
