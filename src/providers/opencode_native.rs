//! Native OpenCode 1.x / 2.x writers.
//!
//! The 1.x and 2.x tables are projections of OpenCode's own event log, so
//! casr never grafts rows into `opencode.db` behind its back (see
//! [`crate::providers::opencode`]). Instead these providers convert a
//! canonical session into the provider-native transfer JSON and hand it to
//! each CLI's own `import` command — the same path `opencode export` feeds —
//! so every row, index and projection is created by OpenCode itself.
//!
//! - **1.x** (`alias: oc1`) shells out to `opencode import <file>` run from
//!   the target workspace. The importer overrides `projectID`/`directory`/
//!   `path` from its invocation, grafting the session into the right project.
//! - **2.x** (`alias: oc2`) shells out to `opencode2 import <file>
//!   --directory <workspace>`, which resolves the project and writes through
//!   OpenCode 2's own services.
//!
//! Both write fresh `ses_`-prefixed session ids per conversion: the native
//! importers are append-oriented (re-importing an existing id keeps the old
//! messages), so a stable id would corrupt the transcript instead of
//! overwriting it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use tracing::{debug, info, warn};

use crate::discovery::DetectionResult;
use crate::model::{
    CanonicalMessage, CanonicalSession, MessageRole, ToolResult, effective_workspace,
    native_name_from_metadata, reindex_messages,
};
use crate::providers::opencode::OpenCode;
use crate::providers::{Provider, WriteOptions, WrittenSession};

/// Which native OpenCode generation this provider imports into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeVariant {
    /// OpenCode 1.x: `opencode import`, singular `session`/`message`/`part`.
    V1,
    /// OpenCode 2.x: `opencode2 import`, `session_v2`/`session_message`.
    V2,
}

impl NativeVariant {
    const fn cli_program(self) -> &'static str {
        match self {
            Self::V1 => "opencode",
            Self::V2 => "opencode2",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::V1 => "OpenCode 1.x (native import)",
            Self::V2 => "OpenCode 2.x (native import)",
        }
    }
}

/// Provider that imports canonical sessions through OpenCode's own CLIs.
pub struct OpenCodeNative {
    /// Which OpenCode generation to target.
    pub variant: NativeVariant,
}

/// Registry-exposed instances (not part of the default registry: they share
/// session discovery with [`OpenCode`] and would double-match every id).
pub static OPENCODE_V1: OpenCodeNative = OpenCodeNative {
    variant: NativeVariant::V1,
};
pub static OPENCODE_V2: OpenCodeNative = OpenCodeNative {
    variant: NativeVariant::V2,
};

/// Fresh native id: `<prefix>_<32 hex>` (`ses_…`, `msg_…`, `prt_…`).
fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

/// Monotonic per-transfer id generator for messages and parts.
///
/// OpenCode orders rows by `(time_created, id)` (messages) and by `id`
/// (parts), so entries sharing a timestamp must still sort in the order they
/// were written. A zero-padded sequence prefix guarantees that.
struct IdGen(usize);

impl IdGen {
    fn next(&mut self, prefix: &str) -> String {
        self.0 += 1;
        format!("{prefix}_{:012}_{}", self.0, uuid::Uuid::new_v4().simple())
    }
}

/// Split a canonical model name into OpenCode's `(providerID, modelID)`.
///
/// OpenCode only ever displays these; a historical session converted from
/// another agent is never re-run, so a best-effort mapping is enough.
fn split_model(name: Option<&str>) -> (String, String) {
    let full = name
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("casr-converted")
        .to_string();
    let provider = if full.starts_with("claude") {
        "anthropic"
    } else if full.starts_with("gpt")
        || full.starts_with("o1")
        || full.starts_with("o3")
        || full.starts_with("o4")
        || full.starts_with("codex")
        || full.starts_with("chatgpt")
    {
        "openai"
    } else if full.starts_with("gemini") || full.starts_with('i') && full.contains("mage") {
        "google"
    } else if full.starts_with("grok") {
        "xai"
    } else {
        "casr"
    };
    (provider.to_string(), full)
}

/// Join tool-result contents exactly the way the 1.x/2.x readers fall back
/// to when an assistant turn has no text parts.
fn results_join(results: &[ToolResult]) -> String {
    results
        .iter()
        .map(|result| result.content.as_str())
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<&str>>()
        .join("\n")
}

/// Coerce tool-call arguments into the JSON object the native tool states
/// require (`state.input` is a record).
fn arguments_object(arguments: &serde_json::Value) -> serde_json::Value {
    match arguments {
        serde_json::Value::Object(_) => arguments.clone(),
        serde_json::Value::Null => serde_json::json!({}),
        serde_json::Value::String(raw) => match serde_json::from_str::<serde_json::Value>(raw) {
            Ok(value @ serde_json::Value::Object(_)) => value,
            _ => serde_json::json!({ "value": raw }),
        },
        other => serde_json::json!({ "value": other }),
    }
}

/// Whether to serialize this message's content as a text part.
///
/// Content that the readers would derive from the tool results (their
/// text→reasoning→results fallback) is not duplicated as a text part, so the
/// transcript shows each output once: on the tool card itself.
fn should_emit_text_part(message: &CanonicalMessage) -> bool {
    if message.content.trim().is_empty() {
        return false;
    }
    if message.tool_results.is_empty() {
        return true;
    }
    message.content != results_join(&message.tool_results)
}

/// Reshape a canonical session so write ↔ native read-back agree.
///
/// The native formats keep a tool call and its result on one assistant turn,
/// and their readers derive `content` from the results when no text part
/// exists. Mirroring that on the canonical side before the write makes the
/// pipeline's read-back verification an exact comparison:
///
/// 1. Anchor: 1.x requires every assistant message to reference a parent
///    user message, so a transcript that opens with an assistant turn gets a
///    synthetic leading user entry.
/// 2. Merge: contiguous assistant entries sharing a source `message.id`
///    (Claude Code streams one logical turn as several JSONL entries) become
///    one native assistant message.
/// 3. Move: tool results living on user-side turns are reattached to the
///    assistant turn that issued the call, matched by call id.
/// 4. Drop: turns left with no text, calls or results disappear.
/// 5. Materialize: turns with results but no text get `content` set to the
///    readers' derived join, and the writer then suppresses the text part.
pub(super) fn prepare_native_session(session: &mut CanonicalSession) -> Vec<String> {
    let mut warnings = Vec::new();

    // 0. Monotonize timestamps. Source transcripts (e.g. a Claude Code
    //    session continued from an earlier one) can carry out-of-order
    //    timestamps, but native OpenCode orders rows by `(time_created, id)`
    //    — a regression would reorder the transcript on read-back. Clamp
    //    each message to at least its predecessor's time.
    let mut floor = i64::MIN;
    for message in &mut session.messages {
        if let Some(ts) = message.timestamp {
            if ts < floor {
                message.timestamp = Some(floor);
            } else {
                floor = ts;
            }
        }
    }

    // 1. Anchor a transcript that opens with an assistant turn.
    if session.messages.first().map(|msg| &msg.role) == Some(&MessageRole::Assistant) {
        // The anchor must also sort first by time: OpenCode orders rows by
        // `(time_created, id)`, so a started_at that post-dates the earliest
        // message would still push the anchor behind it.
        let anchor_time = session
            .messages
            .iter()
            .filter_map(|message| message.timestamp)
            .min()
            .or(session.started_at);
        session.messages.insert(
            0,
            CanonicalMessage {
                idx: 0,
                role: MessageRole::User,
                content: format!(
                    "[casr] imported from {}; the source transcript opens with an assistant turn",
                    session.provider_slug
                ),
                timestamp: anchor_time,
                author: None,
                tool_calls: Vec::new(),
                tool_results: Vec::new(),
                extra: serde_json::Value::Null,
            },
        );
        warnings.push(
            "Added a synthetic leading user turn: the source transcript starts with an assistant \
             message, which native OpenCode transcripts cannot reference."
                .to_string(),
        );
    }

    // 2. Merge contiguous assistant entries sharing a source message id.
    let mut merged_runs = 0usize;
    let mut merged_entries = 0usize;
    let mut index = 0;
    while index < session.messages.len() {
        let Some(source_id) = claude_source_message_id(&session.messages[index]) else {
            index += 1;
            continue;
        };
        if session.messages[index].role != MessageRole::Assistant {
            index += 1;
            continue;
        }
        let mut run_end = index + 1;
        while run_end < session.messages.len()
            && session.messages[run_end].role == MessageRole::Assistant
            && claude_source_message_id(&session.messages[run_end]).as_deref() == Some(&source_id)
        {
            run_end += 1;
        }
        if run_end > index + 1 {
            let run: Vec<CanonicalMessage> = session.messages.drain(index..run_end).collect();
            let mut head = run[0].clone();
            for tail in &run[1..] {
                if !tail.content.trim().is_empty() {
                    if !head.content.trim().is_empty() {
                        head.content.push('\n');
                    }
                    head.content.push_str(&tail.content);
                }
                head.tool_calls.extend(tail.tool_calls.iter().cloned());
                head.tool_results.extend(tail.tool_results.iter().cloned());
                head.author = head.author.or_else(|| tail.author.clone());
                head.timestamp = head.timestamp.or(tail.timestamp);
            }
            merged_runs += 1;
            merged_entries += run.len() - 1;
            session.messages.insert(index, head);
        }
        index += 1;
    }
    if merged_runs > 0 {
        warnings.push(format!(
            "Merged {merged_entries} streamed assistant entr{ies} into {merged_runs} native \
             assistant turn(s), matching the source tool's message grouping.",
            ies = if merged_entries == 1 { "y" } else { "ies" }
        ));
    }

    // 3. Reattach each tool result to the assistant turn that issued its call.
    let mut owners: HashMap<String, usize> = HashMap::new();
    for (position, message) in session.messages.iter().enumerate() {
        if message.role == MessageRole::Assistant {
            for call in &message.tool_calls {
                if let Some(id) = &call.id {
                    owners.insert(id.clone(), position);
                }
            }
        }
    }
    let mut moved = 0usize;
    for position in 0..session.messages.len() {
        let results = std::mem::take(&mut session.messages[position].tool_results);
        let mut kept = Vec::new();
        for result in results {
            match result
                .call_id
                .as_deref()
                .and_then(|id| owners.get(id).copied())
            {
                Some(owner) if owner != position => {
                    session.messages[owner].tool_results.push(result);
                    moved += 1;
                }
                _ => kept.push(result),
            }
        }
        session.messages[position].tool_results = kept;
    }
    if moved > 0 {
        warnings.push(format!(
            "Attached {moved} tool result(s) to the assistant turn(s) that issued the call, the \
             native OpenCode message shape."
        ));
    }

    // 4. Drop turns with nothing left to show.
    let before = session.messages.len();
    session.messages.retain(|message| {
        !(message.content.trim().is_empty()
            && message.tool_calls.is_empty()
            && message.tool_results.is_empty())
    });
    let dropped = before - session.messages.len();
    if dropped > 0 {
        warnings.push(format!(
            "Dropped {dropped} source-side turn(s) that carried only tool results now attached \
             to their assistant turns."
        ));
    }

    // 5. Materialize the readers' derived content for result-only turns.
    let mut materialized = 0usize;
    for message in &mut session.messages {
        if message.content.trim().is_empty() && !message.tool_results.is_empty() {
            let derived = results_join(&message.tool_results);
            if !derived.trim().is_empty() {
                message.content = derived;
                materialized += 1;
            }
        }
    }
    if materialized > 0 {
        warnings.push(format!(
            "Materialized display content for {materialized} tool-only assistant turn(s) from \
             their results, mirroring the native readers."
        ));
    }

    reindex_messages(&mut session.messages);
    warnings
}

/// The source provider's logical message id, when the entry carries one
/// (Claude Code keeps the Anthropic response id in `message.id`).
fn claude_source_message_id(message: &CanonicalMessage) -> Option<String> {
    message
        .extra
        .pointer("/message/id")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(ToString::to_string)
}

/// Build the OpenCode 1.x transfer document for `opencode import`.
///
/// Shapes follow upstream `packages/schema/src/v1/session.ts`: flat parts,
/// one `tool` part per call carrying its own settled state, and assistant
/// rows that reference their parent user message.
pub(super) fn build_v1_transfer(
    session: &CanonicalSession,
    parent_session_id: Option<&str>,
) -> serde_json::Value {
    let ses_id = new_id("ses");
    let workspace = effective_workspace(session);
    let directory = workspace.display().to_string();
    let model_source = session.model_name.clone().or_else(|| {
        session
            .messages
            .iter()
            .find(|message| message.role == MessageRole::Assistant)
            .and_then(|message| message.author.clone())
    });
    let (provider_id, model_id) = split_model(model_source.as_deref());
    let title = native_name_from_metadata(&session.metadata)
        .or_else(|| session.title.clone())
        .unwrap_or_else(|| "Converted session".to_string());

    let mut info = serde_json::json!({
        "id": ses_id,
        "slug": ses_id,
        // Overridden by the importer from its own invocation context.
        "projectID": "casr-import-placeholder",
        "directory": directory,
        "title": title,
        "version": env!("CARGO_PKG_VERSION"),
        "time": {
            "created": session.started_at.unwrap_or(0),
            "updated": session.ended_at.unwrap_or_else(|| session.started_at.unwrap_or(0)),
        },
        "agent": "build",
    });
    if let Some(parent) = parent_session_id {
        info["parentID"] = serde_json::json!(parent);
    }

    let zero_tokens = serde_json::json!({
        "input": 0, "output": 0, "reasoning": 0,
        "cache": { "read": 0, "write": 0 },
    });

    let mut messages = Vec::with_capacity(session.messages.len());
    let mut ids = IdGen(0);
    // The anchor user turn (message 0) must sort before every other row even
    // when timestamps tie: OpenCode orders by `(time_created, id)`, and a
    // resumable transcript has to open with the user turn its assistant rows
    // reference as `parentID`.
    let anchor_user_id = format!("msg_000000000000_anchor_{}", uuid::Uuid::new_v4().simple());
    let mut last_user_id: Option<String> = None;
    for (position, message) in session.messages.iter().enumerate() {
        let msg_id = if position == 0 {
            anchor_user_id.clone()
        } else {
            ids.next("msg")
        };
        let timestamp = message.timestamp.unwrap_or(0);

        if message.role == MessageRole::Assistant {
            let mut parts = Vec::new();
            if should_emit_text_part(message) {
                parts.push(serde_json::json!({
                    "id": ids.next("prt"), "sessionID": ses_id, "messageID": msg_id,
                    "type": "text", "text": message.content,
                }));
            }
            let mut has_tool = false;
            for call in &message.tool_calls {
                has_tool = true;
                let result = message
                    .tool_results
                    .iter()
                    .find(|result| result.call_id.as_deref() == call.id.as_deref());
                let mut state = serde_json::json!({
                    "input": arguments_object(&call.arguments),
                    "metadata": {},
                    "time": { "start": timestamp, "end": timestamp },
                });
                match result {
                    Some(result) if result.is_error => {
                        state["status"] = serde_json::json!("error");
                        state["error"] = serde_json::json!(result.content);
                    }
                    Some(result) => {
                        state["status"] = serde_json::json!("completed");
                        state["output"] = serde_json::json!(result.content);
                        state["title"] = serde_json::json!(call.name);
                    }
                    None => {
                        // A call with no settled result: surface it as an
                        // interrupted execution rather than a dangling call.
                        state["status"] = serde_json::json!("error");
                        state["error"] = serde_json::json!("[Tool execution was interrupted]");
                    }
                }
                parts.push(serde_json::json!({
                    "id": ids.next("prt"), "sessionID": ses_id, "messageID": msg_id,
                    "type": "tool",
                    "callID": call.id.clone().unwrap_or_else(|| new_id("call")),
                    "tool": call.name,
                    "state": state,
                }));
            }

            // 1.x requires a parent user message; `prepare_native_session`
            // anchors assistant-first transcripts, so this is always bound.
            let parent = last_user_id.clone().unwrap_or_else(|| new_id("msg"));
            messages.push(serde_json::json!({
                "info": {
                    "id": msg_id, "sessionID": ses_id, "role": "assistant",
                    "time": { "created": timestamp, "completed": timestamp },
                    "parentID": parent,
                    "modelID": model_id, "providerID": provider_id,
                    "mode": "build", "agent": "build",
                    "path": { "cwd": directory, "root": directory },
                    "cost": 0.0,
                    "tokens": zero_tokens,
                    "finish": if has_tool { "tool-calls" } else { "stop" },
                },
                "parts": parts,
            }));
        } else {
            // User, System, Tool and other side turns all travel as user
            // rows; the 1.x schema has no other replayable role.
            let mut parts = Vec::new();
            if should_emit_text_part(message) {
                parts.push(serde_json::json!({
                    "id": ids.next("prt"), "sessionID": ses_id, "messageID": msg_id,
                    "type": "text", "text": message.content,
                }));
            }
            messages.push(serde_json::json!({
                "info": {
                    "id": msg_id, "sessionID": ses_id, "role": "user",
                    "time": { "created": timestamp },
                    "agent": "build",
                    "model": { "providerID": provider_id, "modelID": model_id },
                },
                "parts": parts,
            }));
            last_user_id = Some(msg_id);
        }
    }

    serde_json::json!({ "info": info, "messages": messages })
}

/// Build the OpenCode 2.x transfer document for `opencode2 import`.
///
/// Shapes follow the live `/openapi.json` schemas: `Session.Info` plus flat
/// `Session.Message.Info` entries with `content[]` blocks on assistants.
pub(super) fn build_v2_transfer(
    session: &CanonicalSession,
    parent_session_id: Option<&str>,
    project_id: &str,
) -> serde_json::Value {
    let ses_id = new_id("ses");
    let workspace = effective_workspace(session);
    let directory = workspace.display().to_string();
    let model_source = session.model_name.clone().or_else(|| {
        session
            .messages
            .iter()
            .find(|message| message.role == MessageRole::Assistant)
            .and_then(|message| message.author.clone())
    });
    let (provider_id, model_id) = split_model(model_source.as_deref());
    let title = native_name_from_metadata(&session.metadata)
        .or_else(|| session.title.clone())
        .unwrap_or_else(|| "Converted session".to_string());

    let mut info = serde_json::json!({
        "id": ses_id,
        "projectID": project_id,
        "title": title,
        "agent": "build",
        "model": { "id": model_id, "providerID": provider_id },
        "cost": 0.0,
        "tokens": {
            "input": 0, "output": 0, "reasoning": 0,
            "cache": { "read": 0, "write": 0 },
        },
        "time": {
            "created": session.started_at.unwrap_or(0),
            "updated": session.ended_at.unwrap_or_else(|| session.started_at.unwrap_or(0)),
        },
        "location": { "directory": directory },
    });
    if let Some(parent) = parent_session_id {
        info["parentID"] = serde_json::json!(parent);
    }

    let mut messages = Vec::with_capacity(session.messages.len());
    let mut ids = IdGen(0);
    for message in &session.messages {
        let msg_id = ids.next("msg");
        let timestamp = message.timestamp.unwrap_or(0);

        if message.role == MessageRole::Assistant {
            let mut content = Vec::new();
            if should_emit_text_part(message) {
                content.push(serde_json::json!({ "type": "text", "text": message.content }));
            }
            let mut has_tool = false;
            for call in &message.tool_calls {
                has_tool = true;
                let result = message
                    .tool_results
                    .iter()
                    .find(|result| result.call_id.as_deref() == call.id.as_deref());
                let mut state = serde_json::json!({
                    "input": arguments_object(&call.arguments),
                    "metadata": {},
                });
                match result {
                    Some(result) if result.is_error => {
                        state["status"] = serde_json::json!("error");
                        // The 2.x readers render errors as `{type}: {message}`;
                        // an empty type keeps the read-back byte-identical to
                        // the canonical error text.
                        state["error"] = serde_json::json!({
                            "type": "",
                            "message": result.content,
                        });
                    }
                    Some(result) => {
                        state["status"] = serde_json::json!("completed");
                        state["content"] =
                            serde_json::json!([{ "type": "text", "text": result.content }]);
                    }
                    None => {
                        state["status"] = serde_json::json!("error");
                        state["error"] = serde_json::json!({
                            "type": "tool_error",
                            "message": "[Tool execution was interrupted]",
                        });
                    }
                }
                content.push(serde_json::json!({
                    "type": "tool",
                    "id": call.id.clone().unwrap_or_else(|| new_id("call")),
                    "name": call.name,
                    "executed": result.is_some(),
                    "state": state,
                    "time": {
                        "created": timestamp,
                        "ran": timestamp,
                        "completed": timestamp,
                    },
                }));
            }
            messages.push(serde_json::json!({
                "id": msg_id,
                "time": { "created": timestamp, "completed": timestamp },
                "agent": "build",
                "model": { "id": model_id, "providerID": provider_id },
                "content": content,
                "finish": if has_tool { "tool-calls" } else { "stop" },
                "type": "assistant",
            }));
        } else {
            messages.push(serde_json::json!({
                "id": msg_id,
                "time": { "created": timestamp },
                "text": message.content,
                "type": "user",
            }));
        }
    }

    serde_json::json!({ "info": info, "messages": messages })
}

/// Resolve the OpenCode 2.x project id for a workspace with a read-only
/// lookup, so `session_v2.project_id` lands on the real row when the
/// workspace is already known to OpenCode.
fn v2_project_id(workspace: &Path) -> Option<String> {
    for db in OpenCode::find_db_files() {
        let Ok(conn) = OpenCode::open_db(&db) else {
            continue;
        };
        if let Ok(id) = conn.query_row(
            "SELECT id FROM project WHERE worktree = ?1 LIMIT 1",
            rusqlite::params![workspace.display().to_string()],
            |row| row.get::<_, String>(0),
        ) {
            return Some(id);
        }
    }
    None
}

impl Provider for OpenCodeNative {
    fn name(&self) -> &str {
        self.variant.label()
    }

    fn slug(&self) -> &str {
        match self.variant {
            NativeVariant::V1 => "opencode-v1",
            NativeVariant::V2 => "opencode-v2",
        }
    }

    fn cli_alias(&self) -> &str {
        match self.variant {
            NativeVariant::V1 => "oc1",
            NativeVariant::V2 => "oc2",
        }
    }

    fn detect(&self) -> DetectionResult {
        let mut installed = false;
        let mut evidence = Vec::new();
        let program = self.variant.cli_program();
        if which::which(program).is_ok() {
            installed = true;
            evidence.push(format!("{program} binary found in PATH"));
        }
        let dbs = OpenCode::find_db_files();
        if !dbs.is_empty() {
            installed = true;
            evidence.push(format!("found {} opencode.db database(s)", dbs.len()));
        }
        DetectionResult {
            installed,
            version: None,
            evidence,
        }
    }

    fn session_roots(&self) -> Vec<PathBuf> {
        OpenCode.session_roots()
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        OpenCode.owns_session(session_id)
    }

    fn list_sessions(&self) -> Option<Vec<(String, PathBuf)>> {
        OpenCode.list_sessions()
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        OpenCode.read_session(path)
    }

    fn prepare_session(&self, session: &mut CanonicalSession) -> anyhow::Result<Vec<String>> {
        Ok(prepare_native_session(session))
    }

    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        self.write_session_with_parent(session, opts, None)
    }

    fn write_session_with_parent(
        &self,
        session: &CanonicalSession,
        _opts: &WriteOptions,
        parent_session_id: Option<&str>,
    ) -> anyhow::Result<WrittenSession> {
        let variant = self.variant;
        let program = variant.cli_program();
        anyhow::ensure!(
            which::which(program).is_ok(),
            "the `{program}` CLI was not found in PATH; native {variant_label} imports run \
             through it so OpenCode itself owns every row it writes",
            variant_label = variant.label()
        );

        let workspace = effective_workspace(session);
        let directory = workspace.display().to_string();

        let project_id = match variant {
            NativeVariant::V2 => {
                v2_project_id(&workspace).unwrap_or_else(|| "casr-import-placeholder".to_string())
            }
            NativeVariant::V1 => "casr-import-placeholder".to_string(),
        };

        let transfer = match variant {
            NativeVariant::V1 => build_v1_transfer(session, parent_session_id),
            NativeVariant::V2 => build_v2_transfer(session, parent_session_id, &project_id),
        };
        let target_id = transfer["info"]["id"]
            .as_str()
            .context("transfer info.id missing")?
            .to_string();

        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join(format!("casr-{program}-import-{}.json", target_id));
        {
            use std::io::Write as _;
            let mut file = std::fs::File::create(&temp_path)
                .with_context(|| format!("failed to create {}", temp_path.display()))?;
            let payload =
                serde_json::to_string(&transfer).context("failed to serialize transfer")?;
            file.write_all(payload.as_bytes())
                .with_context(|| format!("failed to write {}", temp_path.display()))?;
        }

        info!(
            program,
            session_id = %target_id,
            directory = %directory,
            messages = session.messages.len(),
            "importing through native OpenCode CLI"
        );

        let mut command = std::process::Command::new(program);
        command.arg("import").arg(&temp_path);
        if variant == NativeVariant::V2 {
            command.arg("--directory").arg(&directory);
        }
        command.current_dir(&workspace);
        let output = command
            .output()
            .with_context(|| format!("failed to run `{program} import`"))?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        anyhow::ensure!(
            output.status.success(),
            "`{program} import` failed ({}): {}",
            output.status,
            if stderr.trim().is_empty() {
                &stdout
            } else {
                &stderr
            }
        );
        if !stdout.contains(&target_id) {
            warn!(
                program,
                session_id = %target_id,
                stdout = %stdout.trim(),
                "importer did not echo the session id; relying on database discovery"
            );
        }

        // The rows now live wherever the CLI put them; find the database that
        // carries the new session so read-back verification can reach it.
        let db_path = OpenCode::find_db_files()
            .into_iter()
            .find(|db| {
                OpenCode::open_db(db)
                    .map(|conn| OpenCode::session_exists_any_schema(&conn, &target_id))
                    .unwrap_or(false)
            })
            .with_context(|| {
                format!(
                    "`{program} import` reported success but session `{target_id}` is not \
                     present in any discovered opencode.db; the import did not land where \
                     casr looks. Inspect {temp} for the payload that was sent.",
                    temp = temp_path.display()
                )
            })?;
        let _ = std::fs::remove_file(&temp_path);

        let resume_command = self.resume_command(&target_id);
        let mut warnings = vec![format!(
            "Imported natively through `{program}`. Resume with `{resume_command}` run from \
             {directory}."
        )];
        if project_id == "casr-import-placeholder" && variant == NativeVariant::V2 {
            warnings.push(format!(
                "No OpenCode project row exists for {directory}; the importer was left to \
                 resolve the project itself."
            ));
        }

        debug!(
            session_id = %target_id,
            db = %db_path.display(),
            "native OpenCode import verified present"
        );

        Ok(WrittenSession {
            paths: vec![OpenCode::virtual_session_path(&db_path, &target_id)],
            session_id: target_id,
            resume_command,
            backup_path: None,
            warnings,
        })
    }

    fn resume_command(&self, session_id: &str) -> String {
        match self.variant {
            NativeVariant::V1 => format!("opencode -s {session_id}"),
            NativeVariant::V2 => format!("opencode2 -s {session_id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ToolCall;

    fn message(
        role: MessageRole,
        content: &str,
        author: Option<&str>,
        extra: serde_json::Value,
    ) -> CanonicalMessage {
        CanonicalMessage {
            idx: 0,
            role,
            content: content.to_string(),
            timestamp: Some(1_700_000_000_000),
            author: author.map(ToString::to_string),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            extra,
        }
    }

    fn sample_session() -> CanonicalSession {
        let mut user = message(
            MessageRole::User,
            "Please inspect src/main.rs",
            None,
            serde_json::Value::Null,
        );
        user.timestamp = Some(1_700_000_000_000);

        let mut tool_turn = message(
            MessageRole::Assistant,
            "",
            Some("claude-sonnet-4-5"),
            serde_json::json!({ "message": { "id": "msg_src_1" } }),
        );
        tool_turn.tool_calls.push(ToolCall {
            id: Some("toolu_01".to_string()),
            name: "Read".to_string(),
            arguments: serde_json::json!({ "file_path": "src/main.rs" }),
        });

        let mut result_turn = message(MessageRole::User, "", None, serde_json::Value::Null);
        result_turn.tool_results.push(ToolResult {
            call_id: Some("toolu_01".to_string()),
            content: "fn main() {}".to_string(),
            is_error: false,
        });

        let text_turn = message(
            MessageRole::Assistant,
            "main is empty.",
            Some("claude-sonnet-4-5"),
            serde_json::json!({ "message": { "id": "msg_src_2" } }),
        );

        CanonicalSession {
            session_id: "source-1".to_string(),
            provider_slug: "claude-code".to_string(),
            workspace: Some(PathBuf::from("/tmp/proj")),
            title: Some("Fix the adapter".to_string()),
            started_at: Some(1_700_000_000_000),
            ended_at: Some(1_700_000_010_000),
            messages: vec![user, tool_turn, result_turn, text_turn],
            metadata: serde_json::json!({
                crate::model::NATIVE_NAME_META_KEY: "CBenchNew",
            }),
            source_path: PathBuf::from("/tmp/source.jsonl"),
            model_name: Some("claude-sonnet-4-5".to_string()),
        }
    }

    #[test]
    fn prepare_moves_results_and_materializes_content() {
        let mut session = sample_session();
        let warnings = prepare_native_session(&mut session);

        // user, assistant(tool+result), assistant(text)
        assert_eq!(session.messages.len(), 3);
        assert_eq!(session.messages[0].role, MessageRole::User);
        assert_eq!(session.messages[1].role, MessageRole::Assistant);
        assert_eq!(session.messages[1].tool_results.len(), 1);
        assert_eq!(session.messages[1].content, "fn main() {}");
        assert_eq!(session.messages[2].content, "main is empty.");
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("tool result"))
        );
    }

    #[test]
    fn prepare_merges_streamed_assistant_entries() {
        let mut session = sample_session();
        // The text turn directly follows the tool turn within one logical
        // source message: contiguous entries with the same source id merge,
        // and the user-side result reattaches to the merged turn.
        session.messages[3].extra["message"]["id"] = serde_json::json!("msg_src_1");
        session.messages.swap(2, 3); // [user, tool, text, result-holder]
        prepare_native_session(&mut session);

        // user + one merged assistant turn carrying call, result and text.
        assert_eq!(session.messages.len(), 2);
        let merged = &session.messages[1];
        assert_eq!(merged.tool_calls.len(), 1);
        assert_eq!(merged.tool_results.len(), 1);
        assert_eq!(merged.content, "main is empty.");
    }

    #[test]
    fn prepare_anchors_assistant_first_transcripts() {
        let mut session = sample_session();
        session.messages.remove(0);
        session.messages.remove(0);
        session.messages.remove(0);
        let warnings = prepare_native_session(&mut session);
        assert_eq!(session.messages[0].role, MessageRole::User);
        assert!(session.messages[0].content.contains("[casr]"));
        assert!(warnings.iter().any(|warning| warning.contains("synthetic")));
    }

    #[test]
    fn v1_transfer_shapes_match_native_schema() {
        let mut session = sample_session();
        prepare_native_session(&mut session);
        let transfer = build_v1_transfer(&session, None);

        let info = &transfer["info"];
        assert!(info["id"].as_str().unwrap().starts_with("ses_"));
        assert_eq!(info["title"], "CBenchNew");
        assert_eq!(info["directory"], "/tmp/proj");
        assert!(info.get("parentID").is_none());

        let messages = transfer["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);

        let user = &messages[0];
        assert_eq!(user["info"]["role"], "user");
        assert!(user["info"]["id"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(user["info"]["model"]["modelID"], "claude-sonnet-4-5");
        let user_id = user["info"]["id"].as_str().unwrap();

        let tool_turn = &messages[1];
        assert_eq!(tool_turn["info"]["role"], "assistant");
        assert_eq!(tool_turn["info"]["parentID"], user_id);
        assert_eq!(tool_turn["info"]["finish"], "tool-calls");
        assert_eq!(tool_turn["info"]["providerID"], "anthropic");
        assert_eq!(tool_turn["info"]["tokens"]["cache"]["read"], 0);

        // No text part: the content is derived from the tool result.
        let parts = tool_turn["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], "tool");
        assert!(parts[0]["id"].as_str().unwrap().starts_with("prt_"));
        assert_eq!(parts[0]["callID"], "toolu_01");
        assert_eq!(parts[0]["state"]["status"], "completed");
        assert_eq!(parts[0]["state"]["output"], "fn main() {}");
        assert_eq!(parts[0]["state"]["input"]["file_path"], "src/main.rs");

        let text_turn = &messages[2];
        assert_eq!(text_turn["info"]["finish"], "stop");
        let text_parts = text_turn["parts"].as_array().unwrap();
        assert_eq!(text_parts.len(), 1);
        assert_eq!(text_parts[0]["text"], "main is empty.");
    }

    #[test]
    fn v2_transfer_shapes_match_native_schema() {
        let mut session = sample_session();
        prepare_native_session(&mut session);
        let transfer = build_v2_transfer(&session, Some("ses_parent"), "prj_1");

        let info = &transfer["info"];
        assert!(info["id"].as_str().unwrap().starts_with("ses_"));
        assert_eq!(info["parentID"], "ses_parent");
        assert_eq!(info["projectID"], "prj_1");
        assert_eq!(info["location"]["directory"], "/tmp/proj");
        assert_eq!(info["model"]["providerID"], "anthropic");
        assert_eq!(info["tokens"]["cache"]["write"], 0);

        let messages = transfer["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["type"], "user");
        assert_eq!(messages[0]["text"], "Please inspect src/main.rs");

        let tool_turn = &messages[1];
        assert_eq!(tool_turn["type"], "assistant");
        assert_eq!(tool_turn["finish"], "tool-calls");
        let content = tool_turn["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool");
        assert_eq!(content[0]["id"], "toolu_01");
        assert_eq!(content[0]["executed"], true);
        assert_eq!(content[0]["state"]["status"], "completed");
        assert_eq!(content[0]["state"]["content"][0]["text"], "fn main() {}");

        assert_eq!(messages[2]["type"], "assistant");
        assert_eq!(messages[2]["content"][0]["type"], "text");
    }

    #[test]
    fn error_results_map_to_native_error_states() {
        let mut session = sample_session();
        session.messages[2].tool_results[0].is_error = true;
        session.messages[2].tool_results[0].content = "boom".to_string();
        prepare_native_session(&mut session);

        let v1 = build_v1_transfer(&session, None);
        let tool = &v1["messages"][1]["parts"][0];
        assert_eq!(tool["state"]["status"], "error");
        assert_eq!(tool["state"]["error"], "boom");

        let v2 = build_v2_transfer(&session, None, "prj");
        let tool = &v2["messages"][1]["content"][0];
        assert_eq!(tool["state"]["status"], "error");
        assert_eq!(tool["state"]["error"]["type"], "");
        assert_eq!(tool["state"]["error"]["message"], "boom");
    }

    #[test]
    fn generated_ids_sort_in_write_order() {
        let mut ids = IdGen(0);
        let a = ids.next("msg");
        let b = ids.next("msg");
        let c = ids.next("prt");
        assert!(a.starts_with("msg_") && b.starts_with("msg_") && c.starts_with("prt_"));
        assert!(a < b, "same-prefix ids must sort in generation order");
    }

    #[test]
    fn split_model_covers_common_families() {
        assert_eq!(
            split_model(Some("claude-sonnet-4-5")),
            ("anthropic".to_string(), "claude-sonnet-4-5".to_string())
        );
        assert_eq!(
            split_model(Some("gpt-5.2")),
            ("openai".to_string(), "gpt-5.2".to_string())
        );
        assert_eq!(
            split_model(Some("gemini-3.1-pro")),
            ("google".to_string(), "gemini-3.1-pro".to_string())
        );
        assert_eq!(split_model(None).0, "casr");
    }
}
