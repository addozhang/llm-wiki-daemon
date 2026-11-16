use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server: ServerSection,
    pub auth: AuthSection,
    pub security: SecuritySection,
    pub vault: VaultSection,
    #[serde(default)]
    pub watcher: WatcherSection,
    #[serde(default)]
    pub writes: WritesSection,
    #[serde(default)]
    pub wiki: WikiSection,
    #[serde(default)]
    pub logging: LoggingSection,
}

/// Daemon-owned logs live OUTSIDE the vault: rolling tracing log plus a
/// structured JSONL audit trail for every write operation.
#[derive(Debug, Clone, Deserialize)]
pub struct LoggingSection {
    #[serde(default = "default_log_dir")]
    pub dir: String,
    /// Rotate audit.jsonl once it reaches this size (bytes).
    #[serde(default = "default_audit_max_bytes")]
    pub audit_max_bytes: u64,
}

fn default_log_dir() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    format!("{home}/Library/Logs/llm-wiki-daemon")
}

fn default_audit_max_bytes() -> u64 {
    8 * 1024 * 1024
}

impl Default for LoggingSection {
    fn default() -> Self {
        Self {
            dir: default_log_dir(),
            audit_max_bytes: default_audit_max_bytes(),
        }
    }
}

impl LoggingSection {
    pub fn path(&self) -> PathBuf {
        PathBuf::from(self.dir.replace('~', &std::env::var("HOME").unwrap_or_default()))
    }
}

/// Vault-relative directory that owns every wiki page. Writes are only ever
/// allowed inside this subtree, so the value is validated at startup.
#[derive(Debug, Clone, Deserialize)]
pub struct WikiSection {
    #[serde(default = "default_wiki_root")]
    pub root: String,
}

fn default_wiki_root() -> String {
    "wiki".to_string()
}

impl Default for WikiSection {
    fn default() -> Self {
        Self {
            root: default_wiki_root(),
        }
    }
}

/// Write tools are ON by default; `[writes] enabled` is only an emergency kill
/// switch. Real safety comes from the wiki scope checks in `WriteStore`.
#[derive(Debug, Clone, Deserialize)]
pub struct WritesSection {
    // Field-level defaults must match `Default` below: a partial [writes]
    // section must behave the same as an omitted one.
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub require_token: bool,
}

impl Default for WritesSection {
    fn default() -> Self {
        Self {
            // Writes are ON by default, but the store restricts every write to
            // `layer: wiki` pages. This flag is only an emergency kill switch.
            enabled: true,
            require_token: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WatcherSection {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_debounce")]
    pub debounce_ms: u64,
}

impl Default for WatcherSection {
    fn default() -> Self {
        Self {
            enabled: true,
            debounce_ms: 200,
        }
    }
}

fn default_debounce() -> u64 {
    200
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerSection {
    pub bind: String,
    pub port: u16,
    #[serde(default)]
    pub allow_non_loopback: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthSection {
    pub token_env: String,
    #[serde(default = "default_true")]
    pub required_when_non_loopback: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct SecuritySection {
    #[serde(default)]
    pub allowed_hosts: Vec<String>,
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    #[serde(default)]
    pub deny_paths: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VaultSection {
    pub root: String,
    #[serde(default = "default_schema")]
    pub schema: String,
    pub index_db: String,
}

fn default_schema() -> String {
    "AGENTS.md".to_string()
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("读取配置失败: {}", path.display()))?;
        let cfg: Config = toml::from_str(&text).context("解析 config.toml 失败")?;
        Ok(cfg)
    }

    /// Returns the canonicalized vault root. Fails fast on unsafe bind settings.
    pub fn validate(&self) -> Result<PathBuf> {
        let ip: IpAddr = self
            .server
            .bind
            .parse()
            .with_context(|| format!("无效的 bind 地址: {}", self.server.bind))?;

        if !ip.is_loopback() {
            if !self.server.allow_non_loopback {
                bail!("bind={} 不是回环地址；必须显式设置 server.allow_non_loopback = true", self.server.bind);
            }
            if self.auth.required_when_non_loopback && self.token().is_none() {
                bail!("非回环绑定必须在环境变量 {} 中提供 token", self.auth.token_env);
            }
            tracing::warn!("绑定非回环地址 {}，请确认已启用 TLS 与 token", self.server.bind);
        }

        let root = std::fs::canonicalize(&self.vault.root)
            .with_context(|| format!("vault 目录不存在: {}", self.vault.root))?;
        if !root.is_dir() {
            bail!("vault root 不是目录: {}", root.display());
        }
        if self.writes.enabled && self.writes.require_token && self.token().is_none() {
            bail!(
                "writes.enabled = true 且 require_token = true 时，必须在环境变量 {} 中提供 token",
                self.auth.token_env
            );
        }
        let wr = self.wiki_root();
        if wr.is_empty()
            || wr.starts_with('/')
            || wr.starts_with('.')
            || wr.contains('\\')
            || wr.split('/').any(|p| p.is_empty() || p == "." || p == "..")
        {
            bail!("wiki.root 必须是 vault 内的安全相对路径: {:?}", self.wiki.root);
        }
        for denied in &self.security.deny_paths {
            let d = denied.trim_end_matches('/');
            if wr == d
                || wr.starts_with(&format!("{d}/"))
                || d.starts_with(&format!("{wr}/"))
            {
                bail!("wiki.root 与排除区重叠: {wr} / {d}");
            }
        }
        if !root.join(&self.vault.schema).exists() {
            bail!("schema 文件缺失: {}", root.join(&self.vault.schema).display());
        }
        Ok(root)
    }

    /// Normalized vault-relative wiki directory (no trailing slash).
    pub fn wiki_root(&self) -> String {
        self.wiki.root.trim().trim_end_matches('/').to_string()
    }

    pub fn token(&self) -> Option<String> {
        std::env::var(&self.auth.token_env)
            .ok()
            .filter(|s| !s.is_empty())
    }

    /// Token enforcement is a REQUEST-level decision. A stray
    /// LLM_WIKI_DAEMON_TOKEN in the environment must not silently turn on
    /// Bearer auth when `require_token = false` (that would 401 every client,
    /// since neither harness sends an Authorization header). Non-loopback
    /// binds always enforce it, as validated at startup.
    pub fn request_token(&self) -> Option<String> {
        let loopback = self
            .server
            .bind
            .parse::<IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false);
        if self.writes.require_token || !loopback {
            self.token()
        } else {
            None
        }
    }
}
