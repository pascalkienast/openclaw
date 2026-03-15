import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { describe, expect, it, vi } from "vitest";
import {
  TEST_SESSION_ID,
  loadSanitizeSessionHistoryWithCleanMocks,
  makeMockSessionManager,
} from "./pi-embedded-runner.sanitize-session-history.test-harness.js";
import { castAgentMessage } from "./test-helpers/agent-message-fixtures.js";

vi.mock("./pi-embedded-helpers.js", async () => ({
  ...(await vi.importActual("./pi-embedded-helpers.js")),
  isGoogleModelApi: vi.fn(),
  sanitizeSessionMessagesImages: vi.fn(async (msgs) => msgs),
}));

describe("sanitizeSessionHistory OpenAI full-context replay preservation", () => {
  async function sanitize(messages: AgentMessage[]) {
    const harness = await loadSanitizeSessionHistoryWithCleanMocks();
    return harness.sanitizeSessionHistory({
      messages,
      modelApi: "openai-codex-responses",
      provider: "openai-codex",
      modelId: "gpt-5.4",
      sessionManager: makeMockSessionManager(),
      sessionId: TEST_SESSION_ID,
    });
  }

  it("preserves stored OpenAI reasoning, text signatures, and function_call ids", async () => {
    const messages: AgentMessage[] = [
      castAgentMessage({
        role: "assistant",
        api: "openai-responses",
        content: [
          {
            type: "thinking",
            thinking: "internal reasoning",
            thinkingSignature: JSON.stringify({ id: "rs_123", type: "reasoning" }),
          },
          {
            type: "text",
            text: "Working on it.",
            textSignature: JSON.stringify({ v: 1, id: "msg_123", phase: "commentary" }),
          },
          {
            type: "toolCall",
            id: "call_123|fc_123",
            name: "read",
            arguments: { path: "README.md" },
          },
        ],
      }),
      castAgentMessage({
        role: "toolResult",
        toolCallId: "call_123|fc_123",
        toolName: "read",
        content: [{ type: "text", text: "ok" }],
        isError: false,
      }),
    ];

    const result = await sanitize(messages);

    const assistant = result[0] as {
      content?: Array<{
        type?: string;
        id?: string;
        textSignature?: string;
        thinkingSignature?: string;
      }>;
    };
    expect(assistant.content?.some((block) => block.type === "thinking")).toBe(true);
    expect(assistant.content?.find((block) => block.type === "text")?.textSignature).toBe(
      JSON.stringify({ v: 1, id: "msg_123", phase: "commentary" }),
    );
    expect(assistant.content?.find((block) => block.type === "toolCall")?.id).toBe(
      "call_123|fc_123",
    );

    const toolResult = result[1] as { toolCallId?: string };
    expect(toolResult.toolCallId).toBe("call_123|fc_123");
  });

  it("preserves legacy plain-string textSignature ids in stored history", async () => {
    const messages: AgentMessage[] = [
      castAgentMessage({
        role: "assistant",
        api: "openai-responses",
        content: [
          {
            type: "text",
            text: "First block",
            textSignature: "msg_legacy_a",
          },
          {
            type: "text",
            text: "Second block",
            textSignature: "msg_legacy_b",
          },
        ],
      }),
    ];

    const result = await sanitize(messages);

    const assistant = result[0] as {
      content?: Array<{ type?: string; text?: string; textSignature?: string }>;
    };
    expect(assistant.content).toEqual([
      {
        type: "text",
        text: "First block",
        textSignature: "msg_legacy_a",
      },
      {
        type: "text",
        text: "Second block",
        textSignature: "msg_legacy_b",
      },
    ]);
  });

  it("preserves historical OpenAI reasoning-only turns for full-context fallback", async () => {
    const messages: AgentMessage[] = [
      castAgentMessage({
        role: "assistant",
        api: "openai-responses",
        content: [
          {
            type: "thinking",
            thinking: "internal reasoning",
            thinkingSignature: JSON.stringify({ id: "rs_123", type: "reasoning" }),
          },
        ],
      }),
    ];

    const result = await sanitize(messages);

    const assistant = result[0] as {
      content?: Array<{ type?: string; thinking?: string; thinkingSignature?: string }>;
    };
    expect(assistant.content).toEqual([
      {
        type: "thinking",
        thinking: "internal reasoning",
        thinkingSignature: JSON.stringify({ id: "rs_123", type: "reasoning" }),
      },
    ]);
  });
});
