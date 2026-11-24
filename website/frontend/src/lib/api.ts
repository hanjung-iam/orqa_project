// 與 FastAPI 後端溝通。API Key 只存在後端，前端永遠不會接觸到。

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000").replace(/\/$/, "");

export type BranchName = "baseline" | "skill";
export type BranchStatus = "idle" | "loading" | "completed" | "error" | "timeout";

export interface UserTurn {
  role: "user";
  content: string;
}

export interface AssistantTurn {
  role: "assistant";
  content: string;
  thinking: string | null;
  model: string;
  response_time: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
  stop_reason: string | null;
  truncated: boolean;
  fallback_used: boolean;
}

export type Turn = UserTurn | AssistantTurn;

export interface BranchError {
  message: string;
  retryable: boolean;
}

export interface PublicConfig {
  model: string;
  effort: string;
  max_tokens: number;
  show_thinking_summary: boolean;
  max_input_chars: number;
  max_turns_per_branch: number;
  session_ttl_minutes: number;
  skill_name: string;
  api_key_configured: boolean;
  notice: string;
}

export interface Example {
  id: string;
  label: string;
  question: string;
}

export interface Material {
  id: string;
  title: string;
  category: string;
  description: string;
  filename: string;
  available: boolean;
  size_bytes: number | null;
  url: string | null;
}

export interface SkillInfo {
  name: string;
  files: string[];
  sha256: string;
  overview_markdown: string;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function errorFrom(res: Response): Promise<ApiError> {
  let message = `後端回應錯誤（HTTP ${res.status}）`;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") message = body.detail;
    else if (Array.isArray(body?.detail)) message = "請求格式不正確。";
  } catch {
    /* 非 JSON 回應 */
  }
  return new ApiError(message, res.status);
}

const OFFLINE = "無法連線到後端（http://localhost:8000）。請確認 FastAPI 已啟動。";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, init);
  } catch {
    throw new ApiError(OFFLINE, 0);
  }
  if (!res.ok) throw await errorFrom(res);
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string; api_key_configured: boolean }>("/api/health"),
  config: () => request<PublicConfig>("/api/config"),
  examples: () => request<Example[]>("/api/examples"),
  materials: () => request<Material[]>("/api/materials"),
  skill: () => request<SkillInfo>("/api/skill"),
  createSession: () => request<{ session_id: string }>("/api/sessions", { method: "POST" }),
  deleteSession: (id: string) => request<void>(`/api/sessions/${id}`, { method: "DELETE" }),
};

export type StreamEvent =
  | { event: "start"; data: { branch: BranchName; message: string } }
  | { event: "thinking"; data: { text: string } }
  | { event: "delta"; data: { text: string } }
  | { event: "done"; data: { branch: BranchName; turn: AssistantTurn } }
  | { event: "error"; data: { branch: BranchName; status: "error" | "timeout"; message: string; retryable: boolean } };

/**
 * 對單一分支送出訊息（或重試），以 SSE 逐步回傳事件。
 * 後端是 POST + text/event-stream，所以用 fetch 讀取串流（EventSource 只支援 GET）。
 */
export async function streamBranch(
  sessionId: string,
  branch: BranchName,
  message: string | null,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const path =
    message === null
      ? `/api/sessions/${sessionId}/branches/${branch}/retry`
      : `/api/sessions/${sessionId}/branches/${branch}/messages`;
  let res: Response;
  try {
    res = await fetch(API_BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: message === null ? undefined : JSON.stringify({ message }),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(OFFLINE, 0);
  }
  if (!res.ok || !res.body) throw await errorFrom(res);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let finished = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message";
      const dataLines: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (dataLines.length === 0) continue;
      const parsed = { event, data: JSON.parse(dataLines.join("\n")) } as StreamEvent;
      if (parsed.event === "done" || parsed.event === "error") finished = true;
      onEvent(parsed);
    }
  }
  if (!finished) throw new ApiError("與後端的連線中斷，回答未完成。", 0);
}
