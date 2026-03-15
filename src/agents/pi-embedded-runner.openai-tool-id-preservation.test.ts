import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { describe, expect, it } from "vitest";
import {
  makeInMemorySessionManager,
  makeModelSnapshotEntry,
} from "./pi-embedded-runner.sanitize-session-history.test-harness.js";
import { sanitizeSessionHistory } from "./pi-embedded-runner/google.js";
import { castAgentMessage } from "./test-helpers/agent-message-fixtures.js";

describe("sanitizeSessionHistory openai replay tool id preservation", () => {
  const makeSessionManager = () =>
    makeInMemorySessionManager([
      makeModelSnapshotEntry({
        provider: "openai",
        modelApi: "openai-responses",
        modelId: "gpt-5.2-codex",
      }),
    ]);

  const makeMessages = (withReasoning: boolean): AgentMessage[] => [
    castAgentMessage({
      role: "assistant",
      content: [
        ...(withReasoning
          ? [
              {
                type: "thinking",
                thinking: "internal reasoning",
                thinkingSignature: JSON.stringify({ id: "rs_123", type: "reasoning" }),
              },
            ]
          : []),
        { type: "toolCall", id: "call_123|fc_123", name: "noop", arguments: {} },
      ],
    }),
    castAgentMessage({
      role: "toolResult",
      toolCallId: "call_123|fc_123",
      toolName: "noop",
      content: [{ type: "text", text: "ok" }],
      isError: false,
    }),
  ];

  it.each([
    {
      name: "preserves fc ids when replayable reasoning metadata is missing",
      withReasoning: false,
    },
    {
      name: "preserves fc ids when replayable reasoning metadata is present",
      withReasoning: true,
    },
  ])("$name", async ({ withReasoning }) => {
    const result = await sanitizeSessionHistory({
      messages: makeMessages(withReasoning),
      modelApi: "openai-responses",
      provider: "openai",
      modelId: "gpt-5.2-codex",
      sessionManager: makeSessionManager(),
      sessionId: "test-session",
    });

    const assistant = result[0] as { content?: Array<{ type?: string; id?: string }> };
    const toolCall = assistant.content?.find((block) => block.type === "toolCall");
    expect(toolCall?.id).toBe("call_123|fc_123");

    const toolResult = result[1] as { toolCallId?: string };
    expect(toolResult.toolCallId).toBe("call_123|fc_123");
  });
});
