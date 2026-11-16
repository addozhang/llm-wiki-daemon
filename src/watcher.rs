//! Filesystem watcher for incremental index updates (read-only).
//!
//! Human edits in Obsidian land on disk as normal file events. This watcher
//! turns them into `VaultIndex::upsert` / `remove` calls inside the SAME process
//! that owns the SQLite index, so no second daemon or second index exists.
//! It never writes to the vault.
use crate::index::VaultIndex;
use anyhow::{Context, Result};
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher, recommended_watcher};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

fn flush_pending(index: &VaultIndex, pending: &mut HashSet<PathBuf>) {
    if pending.is_empty() {
        return;
    }
    let batch: Vec<PathBuf> = pending.drain().collect();
    let mut changed = 0usize;
    for path in batch {
        match index.apply_path(&path) {
            Ok(true) => changed += 1,
            Ok(false) => {}
            Err(err) => tracing::warn!("索引增量更新失败 {}: {err}", path.display()),
        }
    }
    if changed > 0 {
        tracing::info!("watcher: 增量更新 {changed} 篇笔记");
    }
}

pub fn spawn(index: Arc<VaultIndex>, debounce_ms: u64) -> Result<()> {
    let root = index.root().to_path_buf();
    let (tx, rx) = mpsc::channel::<notify::Result<Event>>();

    let mut watcher: RecommendedWatcher = recommended_watcher(move |res: notify::Result<Event>| {
        let _ = tx.send(res);
    })
    .context("创建 file watcher 失败")?;
    watcher
        .watch(&root, RecursiveMode::Recursive)
        .with_context(|| format!("监听 vault 失败: {}", root.display()))?;

    std::thread::Builder::new()
        .name("vault-watcher".to_string())
        .spawn(move || {
            // The watcher must stay alive for the whole process lifetime.
            let _watcher = watcher;
            let window = Duration::from_millis(debounce_ms.max(20));
            // A pure "silence period" debounce can be starved forever by a
            // steady stream of events (Obsidian autosave, iCloud-like sync,
            // build tools). Force a flush after a bounded wait regardless.
            let max_wait = window.saturating_mul(10);
            let mut pending: HashSet<PathBuf> = HashSet::new();
            let mut first: Option<Instant> = None;

            loop {
                match rx.recv_timeout(window) {
                    Ok(Ok(event)) => {
                        // A rename can carry both the old and the new path.
                        for path in event.paths {
                            index.note_watcher_event();
                            pending.insert(path);
                        }
                        if first.is_none() {
                            first = Some(Instant::now());
                        }
                        if first.is_some_and(|t| t.elapsed() >= max_wait) {
                            flush_pending(&index, &mut pending);
                            first = None;
                        }
                    }
                    Ok(Err(err)) => tracing::warn!("watcher 事件错误: {err}"),
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        flush_pending(&index, &mut pending);
                        first = None;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        flush_pending(&index, &mut pending);
                        break;
                    }
                }
            }
        })
        .context("启动 watcher 线程失败")?;

    Ok(())
}
