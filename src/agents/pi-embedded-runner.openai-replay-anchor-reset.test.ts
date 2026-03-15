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

describe("sanitizeSessionHistory OpenAI replay anchor reset", () => {
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

  it("strips historical responses replay anchors but preserves call_id pairing", async () => {
    const messages: AgentMessage[] = [
      castAgentMessage({
        role: "assistant",
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
    expect(assistant.content?.some((block) => block.type === "thinking")).toBe(false);
    expect(assistant.content?.find((block) => block.type === "text")?.textSignature).toBe(
      JSON.stringify({ v: 1, id: "msg_reset_0_1", phase: "commentary" }),
    );
    expect(assistant.content?.find((block) => block.type === "toolCall")?.id).toBe("call_123");

    const toolResult = result[1] as { toolCallId?: string };
    expect(toolResult.toolCallId).toBe("call_123");
  });

  it("leaves already-reset OpenAI replay ids untouched", async () => {
    const messages: AgentMessage[] = [
      castAgentMessage({
        role: "assistant",
        content: [
          {
            type: "text",
            text: "Done.",
          },
          {
            type: "toolCall",
            id: "call_plain",
            name: "read",
            arguments: {},
          },
        ],
      }),
      castAgentMessage({
        role: "toolResult",
        toolCallId: "call_plain",
        toolName: "read",
        content: [{ type: "text", text: "ok" }],
        isError: false,
      }),
    ];

    const result = await sanitize(messages);

    const assistant = result[0] as {
      content?: Array<{ type?: string; id?: string; text?: string }>;
      usage?: unknown;
    };
    expect(assistant.content).toEqual(
      messages[0]?.role === "assistant" ? messages[0].content : undefined,
    );
    expect(assistant.usage).toBeDefined();

    const toolResult = result[1] as { toolCallId?: string; content?: unknown };
    expect(toolResult.toolCallId).toBe("call_plain");
    expect(toolResult.content).toEqual(
      messages[1]?.role === "toolResult" ? messages[1].content : undefined,
    );
  });
});
