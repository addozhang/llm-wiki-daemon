mod audit;
mod config;
mod index;
mod mcp;
mod store;
mod watcher;

use anyhow::Result;
use axum::{
    Router,
    extract::State,
    http::{Request, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::net::TcpListener;

#[derive(Clone)]
struct SecurityState {
    allowed_hosts: Vec<String>,
    allowed_origins: Vec<String>,
    token: Option<String>,
}

/// Byte-wise comparison with no early exit on content mismatch, so token
/// comparison does not leak where two values first differ. Length is checked
/// first; token length is not treated as secret here.
fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Host/Origin validation prevents DNS rebinding even when bound to loopback.
/// Token is optional on loopback, mandatory on non-loopback (checked at startup).
async fn security_guard(
    State(sec): State<SecurityState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if !sec.allowed_hosts.is_empty() {
        let host = req
            .headers()
            .get(header::HOST)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !sec
            .allowed_hosts
            .iter()
            .any(|h| h.eq_ignore_ascii_case(host))
        {
            return (StatusCode::FORBIDDEN, "invalid Host header").into_response();
        }
    }

    // When an origin allow-list is configured it is MANDATORY: a request with
    // no Origin header is rejected too. Otherwise a non-browser local process
    // (and DNS-rebinding tricks that omit the header) can talk to the daemon
    // while write tools are always on. Both configured harnesses already send
    // an allowed Origin, so this needs no client-side change.
    if !sec.allowed_origins.is_empty() {
        let origin = req
            .headers()
            .get(header::ORIGIN)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !sec.allowed_origins.iter().any(|o| o == origin) {
            return (StatusCode::FORBIDDEN, "missing or invalid Origin header").into_response();
        }
        if let Some(site) = req
            .headers()
            .get("sec-fetch-site")
            .and_then(|v| v.to_str().ok())
        {
            if site.eq_ignore_ascii_case("cross-site") {
                return (StatusCode::FORBIDDEN, "cross-site request rejected").into_response();
            }
        }
    }

    if let Some(token) = &sec.token {
        let expected = format!("Bearer {token}");
        let ok = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(|v| constant_time_eq(v, &expected))
            .unwrap_or(false);
        if !ok {
            return (StatusCode::UNAUTHORIZED, "missing or invalid bearer token").into_response();
        }
    }

    next.run(req).await
}

#[tokio::main]
async fn main() -> Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config.toml"));
    let cfg = config::Config::load(&config_path)?;
    let root = cfg.validate()?;

    // Rolling file appender outside the vault: the launchd stdout.log stays
    // tiny (panic-only), and daily rotation bounds the tracing log.
    let log_dir = cfg.logging.path();
    std::fs::create_dir_all(&log_dir)?;
    let appender = tracing_appender::rolling::RollingFileAppender::new(
        tracing_appender::rolling::Rotation::DAILY,
        &log_dir,
        "daemon.log",
    );
    let (writer, log_guard) = tracing_appender::non_blocking(appender);
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_writer(writer)
        .init();
    let _log_guard = log_guard;

    let index = Arc::new(index::VaultIndex::open(
        root,
        Path::new(&cfg.vault.index_db),
        cfg.security.deny_paths.clone(),
        &cfg.vault.schema,
    )?);
    let count = index.rebuild()?;
    tracing::info!("indexed {count} notes from {}", cfg.vault.root);

    if cfg.watcher.enabled {
        watcher::spawn(Arc::clone(&index), cfg.watcher.debounce_ms)?;
        tracing::info!(
            "watcher enabled (debounce {} ms, macOS FSEvents)",
            cfg.watcher.debounce_ms
        );
    }

    let writes_enabled = cfg.writes.enabled;
    let wiki_root = cfg.wiki_root();
    let audit = Arc::new(audit::AuditLog::new(
        &log_dir,
        cfg.logging.audit_max_bytes,
    )?);
    tracing::info!("audit trail: {}", audit.path().display());
    let store = Arc::new(store::WriteStore::new(
        Arc::clone(&index),
        writes_enabled,
        wiki_root.clone(),
        audit,
    ));
    let handler = mcp::VaultServer::new(index, store, wiki_root.clone());
    tracing::info!(
        "writes {} | wiki root: {}/ (scope-gated) | origin required: {}",
        if writes_enabled { "ENABLED" } else { "disabled" },
        wiki_root,
        !cfg.security.allowed_origins.is_empty(),
    );
    let http_config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true);
    let service = StreamableHttpService::new(
        move || Ok(handler.clone()),
        LocalSessionManager::default().into(),
        http_config,
    );

    let sec = SecurityState {
        allowed_hosts: cfg.security.allowed_hosts.clone(),
        allowed_origins: cfg.security.allowed_origins.clone(),
        token: cfg.request_token(),
    };
    let app = Router::new()
        .nest_service("/mcp", service)
        .layer(axum::middleware::from_fn_with_state(
            sec.clone(),
            security_guard,
        ));

    let addr = format!("{}:{}", cfg.server.bind, cfg.server.port);
    let listener = TcpListener::bind(&addr).await?;
    tracing::info!(
        "llm-wiki-daemon listening on http://{addr}/mcp (mode: {})",
        if cfg.writes.enabled {
            "read-write"
        } else {
            "read-only"
        }
    );
    axum::serve(listener, app).await?;
    Ok(())
}
