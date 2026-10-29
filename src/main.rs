//! make-adapter CLI: same flags as make_adapter.py.
//!
//!   make-adapter engines/freetoken.json --check
//!   make-adapter engines/freetoken.json --out /tmp/preview
//!   make-adapter engines/freetoken.json --lemonade-src <fork checkout>

use make_adapter::{disp, register_cmake, render};
use serde_json::Value;
use std::path::PathBuf;
use std::process::exit;

const USAGE: &str = "usage: make-adapter [-h] (--lemonade-src LEMONADE_SRC | --out OUT | --check) spec";
const HELP: &str = "\
Adapter maker: turn one engine spec (JSON) into a native Lemonade backend.

positional arguments:
  spec

options:
  -h, --help            show this help message and exit
  --lemonade-src LEMONADE_SRC
                        Lemonade source tree to install into (also edits CMakeLists.txt)
  --out OUT             write the generated files under this directory only
  --check               validate the spec and print the file list";

fn usage_error(msg: &str) -> ! {
    eprintln!("{USAGE}\nmake-adapter: error: {msg}");
    exit(2)
}

const LONGS: [&str; 4] = ["--lemonade-src", "--out", "--check", "--help"];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut spec, mut src, mut out, mut check) = (Vec::<String>::new(), None::<String>, None::<String>, false);
    let mut chosen: Vec<&str> = vec![];
    let mut i = 0;
    let mut only_pos = false;
    while i < args.len() {
        let a = &args[i];
        i += 1;
        if only_pos || !a.starts_with('-') || a == "-" {
            spec.push(a.clone());
            continue;
        }
        if a == "--" {
            only_pos = true;
            continue;
        }
        if a == "-h" {
            println!("{USAGE}\n\n{HELP}");
            exit(0);
        }
        if !a.starts_with("--") {
            usage_error(&format!("unrecognized arguments: {a}"));
        }
        let (name, inline) = match a.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (a.as_str(), None),
        };
        let matches: Vec<&str> = LONGS.iter().copied().filter(|l| l.starts_with(name)).collect();
        let opt = if LONGS.contains(&name) {
            name
        } else if matches.len() == 1 {
            matches[0]
        } else if matches.is_empty() {
            usage_error(&format!("unrecognized arguments: {a}"))
        } else {
            usage_error(&format!("ambiguous option: {name} could match {}", matches.join(", ")))
        };
        if opt == "--help" {
            println!("{USAGE}\n\n{HELP}");
            exit(0);
        }
        if opt == "--check" && inline.is_some() {
            usage_error("argument --check: ignored explicit argument");
        }
        if let Some(prev) = chosen.first() {
            usage_error(&format!("argument {opt}: not allowed with argument {prev}"));
        }
        chosen.push(opt);
        match opt {
            "--check" => check = true,
            _ => {
                let v = match inline {
                    Some(v) => v,
                    None if i < args.len() && !(args[i].starts_with('-') && args[i] != "-") => {
                        i += 1;
                        args[i - 1].clone()
                    }
                    None => usage_error(&format!("argument {opt}: expected one argument")),
                };
                if opt == "--out" { out = Some(v) } else { src = Some(v) }
            }
        }
    }
    if chosen.is_empty() {
        usage_error("one of the arguments --lemonade-src --out --check is required");
    }
    if spec.is_empty() {
        usage_error("the following arguments are required: spec");
    }
    if spec.len() > 1 {
        usage_error(&format!("unrecognized arguments: {}", spec[1..].join(" ")));
    }

    let text = std::fs::read_to_string(&spec[0]).unwrap_or_else(|e| {
        eprintln!("{}: {e}", spec[0]);
        exit(1)
    });
    let parsed: Result<Value, _> = serde_json::from_str(&text);
    let (value, files) = match parsed {
        Ok(v) => match render(&v) {
            Ok(f) => (v, f),
            Err(e) => {
                eprintln!("spec error: {e}");
                exit(2)
            }
        },
        Err(e) => {
            eprintln!("spec error: {e}");
            exit(2)
        }
    };
    if check {
        println!("{}", files.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>().join("\n"));
        return;
    }
    let root = PathBuf::from(src.clone().or(out).unwrap());
    for (rel, body) in &files {
        let p = root.join(rel);
        let io = |e: std::io::Error| -> ! {
            eprintln!("{}: {e}", disp(&p));
            exit(1)
        };
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap_or_else(|e| io(e));
        }
        std::fs::write(&p, body).unwrap_or_else(|e| io(e));
        println!("wrote {}", disp(&p));
    }
    if src.is_some() {
        let recipe = value["recipe"].as_str().unwrap();
        let stem = value["stem"].as_str().unwrap();
        match register_cmake(&root, recipe, stem) {
            Ok(changed) => println!(
                "CMakeLists.txt LEMON_BACKENDS: {} \"{recipe}|{stem}\"",
                if changed { "added" } else { "already has" }
            ),
            Err(m) => {
                eprintln!("{m}");
                exit(1)
            }
        }
    }
}
