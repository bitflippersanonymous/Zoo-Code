// npx vitest run core/condense/__tests__/stripReasoningAndFit.spec.ts

import { Anthropic } from "@anthropic-ai/sdk"
import type { ModelInfo } from "@roo-code/types"

import type { ApiHandler } from "../../../api"
import { ApiMessage } from "../../task-persistence/apiMessages"
import {
	isCondenseContextSizeError,
	splitMessagesIntoCondenseChunks,
	stripReasoningBlocks,
	summarizeConversation,
} from "../index"

// Reasoning blocks are an internal extension of the Anthropic content-block union
// (see `buildCleanConversationHistory`), not part of the public SDK type — hence the
// documented double assertion in these helpers.
function textReasoning(text: string): Anthropic.Messages.ContentBlockParam {
	return { type: "reasoning", text, summary: [] } as unknown as Anthropic.Messages.ContentBlockParam
}
function encryptedReasoning(id: string): Anthropic.Messages.ContentBlockParam {
	return {
		type: "reasoning",
		summary: [],
		encrypted_content: `enc-${id}`,
		id,
	} as unknown as Anthropic.Messages.ContentBlockParam
}

vi.mock("@roo-code/telemetry", () => ({
	TelemetryService: {
		instance: {
			captureContextCondensed: vi.fn(),
		},
	},
}))

const taskId = "test-task-id"

function createMockHandler(
	tokensPerMessage: number,
	modelInfo: Partial<ModelInfo> = { contextWindow: 100000, maxTokens: 20000 },
): ApiHandler {
	const handler = {
		createMessage: vi.fn().mockReturnValue(
			(async function* () {
				yield { type: "text" as const, text: "A summary" }
				yield { type: "usage" as const, totalCost: 0.01, outputTokens: 10 }
			})(),
		),
		countTokens: vi.fn().mockResolvedValue(tokensPerMessage),
		getModel: vi.fn().mockReturnValue({ id: "test-model", info: modelInfo }),
	} as unknown as ApiHandler
	return handler
}

describe("stripReasoningBlocks", () => {
	it("should strip plain-text reasoning blocks from assistant messages", () => {
		const messages: ApiMessage[] = [
			{ role: "user", content: "Hello", ts: 1 },
			{
				role: "assistant",
				content: [textReasoning("internal thinking"), { type: "text", text: "Answer" }],
				ts: 2,
			},
			{ role: "user", content: "Thanks", ts: 3 },
		]

		const result = stripReasoningBlocks(messages)

		expect(result).toHaveLength(3)
		// The single remaining text block is collapsed to a plain string (no reasoning)
		expect(typeof result[1].content).toBe("string")
		expect(result[1].content).toBe("Answer")
	})

	it("should strip encrypted reasoning blocks and keep remaining blocks as an array", () => {
		const messages: ApiMessage[] = [
			{
				role: "assistant",
				content: [
					encryptedReasoning("reasoning-1"),
					{ type: "text", text: "Step one" },
					{
						type: "tool_use",
						id: "call-1",
						name: "read_file",
						input: { path: "file.txt" },
					},
				],
				ts: 2,
			},
		]

		const result = stripReasoningBlocks(messages)

		const content = result[0].content as Anthropic.Messages.ContentBlockParam[]
		expect(content.some((b) => (b.type as string) === "reasoning")).toBe(false)
		expect(content).toHaveLength(2)
		expect(content[0].type).toBe("text")
		expect(content[1].type).toBe("tool_use")
	})

	it("should drop standalone reasoning items", () => {
		const messages: ApiMessage[] = [
			{
				role: "user",
				content: "",
				type: "reasoning",
				encrypted_content: "encrypted",
				id: "r-1",
				ts: 1,
			} as ApiMessage,
			{ role: "assistant", content: "Answer", ts: 2 },
		]

		const result = stripReasoningBlocks(messages)

		expect(result).toHaveLength(1)
		expect(result[0].role).toBe("assistant")
	})

	it("should return messages unchanged when no reasoning is present", () => {
		const messages: ApiMessage[] = [
			{ role: "user", content: "Hello", ts: 1 },
			{ role: "assistant", content: "Hi", ts: 2 },
		]

		const result = stripReasoningBlocks(messages)

		expect(result).toEqual(messages)
	})
})

describe("isCondenseContextSizeError", () => {
	it("should match the observed llama.cpp rejection wording", () => {
		expect(
			isCondenseContextSizeError(
				new Error("request (54390 tokens) exceeds the available context size (50176 tokens)"),
			),
		).toBe(true)
	})

	it("should match common OpenAI-compatible variants (case-insensitive)", () => {
		expect(isCondenseContextSizeError(new Error("Maximum context length is 32768 tokens"))).toBe(true)
		expect(isCondenseContextSizeError(new Error("This model's maximum context length is 32768 tokens"))).toBe(true)
		expect(isCondenseContextSizeError(new Error("You exceeded the current context window"))).toBe(true)
		expect(isCondenseContextSizeError(new Error("Too many tokens in request"))).toBe(true)
		expect(isCondenseContextSizeError(new Error("prompt is too long: 60000 tokens > 50000 maximum"))).toBe(true)
		expect(
			isCondenseContextSizeError(
				new Error("This request (90000 tokens) exceeds the context window limit (50000)"),
			),
		).toBe(true)
	})

	it("should detect a context-size error carried in an attached response/body", () => {
		const error = new Error("400 Bad Request")
		;(error as unknown as Record<string, unknown>).response = {
			error: { message: "request exceeds the available context size" },
		}
		expect(isCondenseContextSizeError(error)).toBe(true)
	})

	it("should not match non-context errors", () => {
		expect(isCondenseContextSizeError(new Error("Connection refused"))).toBe(false)
		expect(isCondenseContextSizeError(new Error("401 unauthorized"))).toBe(false)
		expect(isCondenseContextSizeError("not an error object")).toBe(false)
		expect(isCondenseContextSizeError(null)).toBe(false)
	})
})

describe("splitMessagesIntoCondenseChunks", () => {
	function createSplittingHandler(tokensPerMessage: number): ApiHandler {
		return {
			countTokens: vi.fn().mockResolvedValue(tokensPerMessage),
			getModel: vi.fn().mockReturnValue({
				id: "test-model",
				info: { contextWindow: 100000, maxTokens: 20000 },
			}),
		} as unknown as ApiHandler
	}

	const sevenMessages: ApiMessage[] = [
		{ role: "user", content: "m1", ts: 1 },
		{ role: "assistant", content: "m2", ts: 2 },
		{ role: "user", content: "m3", ts: 3 },
		{ role: "assistant", content: "m4", ts: 4 },
		{ role: "user", content: "m5", ts: 5 },
		{ role: "assistant", content: "m6", ts: 6 },
		{ role: "user", content: "m7", ts: 7 },
	]

	it("should return a single chunk when all messages fit", async () => {
		// 7 × 100 = 700 ≤ 2000 budget
		const handler = createSplittingHandler(100)

		const { chunks, allChunksFit } = await splitMessagesIntoCondenseChunks(sevenMessages, handler, 2000)

		expect(chunks).toHaveLength(1)
		expect(chunks[0]).toHaveLength(7)
		expect(allChunksFit).toBe(true)
	})

	it("should split messages into multiple chunks at the budget boundary", async () => {
		// 7 × 1000 = 7000; budget 2000 → 2 per chunk → [2,2,2,1] = 4 chunks
		const handler = createSplittingHandler(1000)

		const { chunks, allChunksFit } = await splitMessagesIntoCondenseChunks(sevenMessages, handler, 2000)

		expect(allChunksFit).toBe(true)
		const sizes = chunks.map((c) => c.length)
		expect(sizes).toEqual([2, 2, 2, 1])
		// Chunks partition the messages in order
		expect(chunks.flat()).toHaveLength(7)
		expect(chunks[0][0].content).toBe("m1")
		expect(chunks[3][0].content).toBe("m7")
	})

	it("should keep a tool_use message with its tool_result (not end a chunk on tool_use)", async () => {
		// assistant with tool_use at index 1, its tool_result at index 2.
		const messages: ApiMessage[] = [
			{ role: "user", content: "start", ts: 1 },
			{
				role: "assistant",
				content: [{ type: "tool_use", id: "t1", name: "read_file", input: {} }],
				ts: 2,
			},
			{
				role: "user",
				content: [{ type: "tool_result", tool_use_id: "t1", content: "file contents" }],
				ts: 3,
			},
			{ role: "assistant", content: "done", ts: 4 },
		]
		// 1000 each; budget 2000 would pack [user, assistant(tool_use)] then [tool_result, done],
		// splitting the tool pair. The boundary rule must move the tool_use into chunk 2.
		const handler = createSplittingHandler(1000)

		const { chunks } = await splitMessagesIntoCondenseChunks(messages, handler, 2000)

		// The assistant tool_use must not be the last message of its chunk.
		for (const chunk of chunks) {
			const last = chunk[chunk.length - 1]
			const hasToolUse =
				Array.isArray(last.content) && last.content.some((b) => (b as { type: string }).type === "tool_use")
			expect(hasToolUse).toBe(false)
		}
		// The tool_use and its tool_result must be in the same chunk.
		const toolUseChunk = chunks.findIndex((c) =>
			c.some(
				(m) => Array.isArray(m.content) && m.content.some((b) => (b as { type: string }).type === "tool_use"),
			),
		)
		const toolResultChunk = chunks.findIndex((c) =>
			c.some(
				(m) =>
					Array.isArray(m.content) && m.content.some((b) => (b as { type: string }).type === "tool_result"),
			),
		)
		expect(toolUseChunk).toBe(toolResultChunk)
	})

	it("should flag allChunksFit=false when a single message exceeds the budget", async () => {
		// One message at 5000 > 2000 budget
		const handler = createSplittingHandler(5000)

		const { chunks, allChunksFit } = await splitMessagesIntoCondenseChunks(
			[
				{ role: "user", content: "big", ts: 1 },
				{ role: "assistant", content: "small", ts: 2 },
			],
			handler,
			2000,
		)

		expect(allChunksFit).toBe(false)
		// The oversized message gets its own chunk
		expect(chunks).toHaveLength(2)
		expect(chunks[0]).toHaveLength(1)
	})
})

describe("summarizeConversation chunked condense", () => {
	/**
	 * A handler whose FULL condense attempt (the first createMessage call) may be rejected
	 * by the server with `failFirstWith`, after which each subsequent call (the chunk
	 * requests) succeeds with a distinct summary. Without `failFirstWith`, every call
	 * succeeds — used to verify the single path is taken when the server accepts it.
	 */
	function createChunkedHandler(
		tokensPerMessage: number,
		modelInfo: { contextWindow: number; maxTokens: number },
		failFirstWith?: string,
	) {
		let callCount = 0
		const handler = {
			createMessage: vi.fn().mockImplementation(() => {
				callCount++
				const n = callCount
				if (n === 1 && failFirstWith) {
					throw new Error(failFirstWith)
				}
				return (async function* () {
					yield { type: "text" as const, text: `summary-${n}` }
					yield { type: "usage" as const, totalCost: 0.01, outputTokens: 10 }
				})()
			}),
			countTokens: vi.fn().mockResolvedValue(tokensPerMessage),
			getModel: vi.fn().mockReturnValue({ id: "test-model", info: modelInfo }),
		} as unknown as ApiHandler
		return handler
	}

	// 8 messages so that the single-request fit check fails but chunks fit.
	const largeMessages: ApiMessage[] = [
		{ role: "user", content: "a", ts: 1 },
		{ role: "assistant", content: "b", ts: 2 },
		{ role: "user", content: "c", ts: 3 },
		{ role: "assistant", content: "d", ts: 4 },
		{ role: "user", content: "e", ts: 5 },
		{ role: "assistant", content: "f", ts: 6 },
		{ role: "user", content: "g", ts: 7 },
		{ role: "assistant", content: "h", ts: 8 },
	]

	it("should use the single (full-context) condense when the server accepts it", async () => {
		// First call succeeds → no chunking, exactly one API call.
		const handler = createChunkedHandler(10000, { contextWindow: 100000, maxTokens: 20000 })

		const result = await summarizeConversation({
			messages: largeMessages,
			apiHandler: handler,
			systemPrompt: "System prompt",
			taskId,
		})

		expect(result.error).toBeUndefined()
		expect(result.summary).toBeTruthy()
		expect(handler.createMessage).toHaveBeenCalledTimes(1)
		expect(result.messages.at(-1)?.isSummary).toBe(true)
	})

	it("should fall back to chunked condense when the server rejects the full request for context size", async () => {
		// First call (full) throws a context-size error → chunked path.
		// Chunk overhead = SUMMARY_PROMPT (10000) + instructions (10000) = 20000.
		// Chunk budget = 75000 (0.75 × window) − 20000 (maxTokens) − 20000 = 35000 →
		// 3 messages per chunk (3 × 10000 = 30000 ≤ 35000) → 3 chunks.
		const handler = createChunkedHandler(
			10000,
			{ contextWindow: 100000, maxTokens: 20000 },
			"request (120000 tokens) exceeds the available context size (80000 tokens)",
		)

		const result = await summarizeConversation({
			messages: largeMessages,
			apiHandler: handler,
			systemPrompt: "System prompt",
			taskId,
		})

		expect(result.error).toBeUndefined()
		expect(result.summary).toBeTruthy()
		// 1 failed full attempt + 3 chunk requests
		expect(handler.createMessage).toHaveBeenCalledTimes(4)
		// The final (chunk) call is the merge: its instruction includes "merging".
		const lastCallArgs = (handler.createMessage as ReturnType<typeof vi.fn>).mock.calls.at(-1)
		expect(lastCallArgs).toBeDefined()
		const lastRequestMessages = lastCallArgs?.[1]
		const instructionText = JSON.stringify(lastRequestMessages?.at(-1))
		expect(instructionText).toContain("merging")
		// A summary message is produced and originals are tagged (non-destructive)
		expect(result.messages.at(-1)?.isSummary).toBe(true)
	})

	it("should NOT chunk when the server rejects the full request for a non-context reason", async () => {
		// First call throws a non-context-size error → surfaced as-is (no chunking).
		const handler = createChunkedHandler(10000, { contextWindow: 100000, maxTokens: 20000 }, "401 unauthorized")

		const result = await summarizeConversation({
			messages: largeMessages,
			apiHandler: handler,
			systemPrompt: "System prompt",
			taskId,
		})

		expect(result.error).toBeTruthy()
		// Only the single failed attempt; no chunked retry.
		expect(handler.createMessage).toHaveBeenCalledTimes(1)
		expect(result.messages).toEqual(largeMessages)
	})

	it("should error (for truncation fallback) when a single message alone exceeds the window", async () => {
		// Full attempt is rejected for context size; chunking then detects a single message
		// (90000) exceeding the chunk budget → allChunksFit=false → error before any chunk
		// request is made. So exactly one API call (the failed full attempt).
		const handler = createChunkedHandler(
			90000,
			{ contextWindow: 100000, maxTokens: 20000 },
			"request exceeds the available context size",
		)

		const result = await summarizeConversation({
			messages: largeMessages,
			apiHandler: handler,
			systemPrompt: "System prompt",
			taskId,
		})

		expect(result.error).toBeTruthy()
		expect(handler.createMessage).toHaveBeenCalledTimes(1)
		expect(result.messages).toEqual(largeMessages)
	})
})
