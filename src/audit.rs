//! Structured, append-only audit trail for every write operation.
//!
//! Lives OUTSIDE the vault so it is never indexed, never committed by the
//! vault's automatic git backup, and never written back into notes. The page
//! itself only holds current truth; this file plus the vault's git history
//! record why and when it changed.
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct AuditLog {
    path: PathBuf,
    max_bytes: u64,
    lock: Mutex<()>,
}

impl AuditLog {
    pub fn new(dir: &Path, max_bytes: u64) -> Result<Self> {
        fs::create_dir_all(dir)
            .with_context(|| format!("创建日志目录失败: {}", dir.display()))?;
        Ok(Self {
            path: dir.join("audit.jsonl"),
            max_bytes: max_bytes.max(1024),
            lock: Mutex::new(()),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one operation. A size-based rotation keeps the file bounded;
    /// older rotations live in the same daemon-owned log directory.
    pub fn record(&self, entry: Value) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        if let Ok(meta) = fs::metadata(&self.path) {
            if meta.len() >= self.max_bytes {
                // Timestamped archive: never overwrite an earlier rotation.
                // Audit history must not disappear just because the file grew.
                let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(0);
                let name = format!("audit.{stamp}-{nanos:09}.jsonl");
                let backup = self.path.with_file_name(name);
                fs::rename(&self.path, &backup)
                    .with_context(|| format!("审计日志轮转失败: {}", self.path.display()))?;
            }
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("打开审计日志失败: {}", self.path.display()))?;
        writeln!(file, "{entry}")
            .with_context(|| format!("写入审计日志失败: {}", self.path.display()))?;
        Ok(())
    }

    pub fn write_event(
        &self,
        op: &str,
        target: &str,
        path: &str,
        reason: Option<&str>,
        version_before: Option<&str>,
        version_after: &str,
        changes: Vec<String>,
    ) -> Result<()> {
        self.record(json!({
            "ts": chrono::Utc::now().to_rfc3339(),
            "op": op,
            "target": target,
            "path": path,
            "reason": reason,
            "version_before": version_before,
            "version_after": version_after,
            "changes": changes,
        }))
    }
}
