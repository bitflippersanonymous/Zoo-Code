import Anthropic from "@anthropic-ai/sdk"
import crypto from "crypto"

import { TelemetryService } from "@roo-code/telemetry"
import { ANTHROPIC_DEFAULT_MAX_TOKENS } from "@roo-code/types"

import { t } from "../../i18n"
import { ApiHandler, ApiHandlerCreateMessageMetadata } from "../../api"
import { ApiMessage } from "../task-persistence/apiMessages"
import { findLast } from "../../shared/array"
import { supportPrompt } from "../../shared/support-prompt"
import { RooIgnoreController } from "../ignore/RooIgnoreController"
import { generateFoldedFileContext } from "./foldedFileContext"

export type { FoldedFileContextResult, FoldedFileContextOptions } from "./foldedFileContext"

/**
 * Converts a tool_use block to a text representation.
 * This allows the conversation to be summarized without requiring the tools parameter.
 */
export function toolUseToText(block: Anthropic.Messages.ToolUseBlockParam): string {
	let input: string
	if (typeof block.input === "object" && block.input !== null) {
		input = Object.entries(block.input)
			.map(([key, value]) => {
				const formattedValue =
					typeof value === "object" && value !== null ? JSON.stringify(value, null, 2) : String(value)
				return `${key}: ${formattedValue}`
			})
			.join("\n")
	} else {
		input = String(block.input)
	}
	return `[Tool Use: ${block.name}]\n${input}`
}

/**
 * Converts a tool_result block to a text representation.
 * This allows the conversation to be summarized without requiring the tools parameter.
 */
export function toolResultToText(block: Anthropic.Messages.ToolResultBlockParam): string {
	const errorSuffix = block.is_error ? " (Error)" : ""
	if (typeof block.content === "string") {
		return `[Tool Result${errorSuffix}]\n${block.content}`
	} else if (Array.isArray(block.content)) {
		const contentText = block.content
			.map((contentBlock) => {
				if (contentBlock.type === "text") {
					return contentBlock.text
				}
				if (contentBlock.type === "image") {
					return "[Image]"
				}
				// Handle any other content block types
				return `[${(contentBlock as { type: string }).type}]`
			})
			.join("\n")
		return `[Tool Result${errorSuffix}]\n${contentText}`
	}
	return `[Tool Result${errorSuffix}]`
}

/**
 * Converts all tool_use and tool_result blocks in a message's content to text representations.
 * This is necessary for providers like Bedrock that require the tools parameter when tool blocks are present.
 * By converting to text, we can send the conversation for summarization without the tools parameter.
 *
 * @param content - The message content (string or array of content blocks)
 * @returns The transformed content with tool blocks converted to text blocks
 */
export function convertToolBlocksToText(
	content: string | Anthropic.Messages.ContentBlockParam[],
): string | Anthropic.Messages.ContentBlockParam[] {
	if (typeof content === "string") {
		return content
	}

	return content.map((block) => {
		if (block.type === "tool_use") {
			return {
				type: "text" as const,
				text: toolUseToText(block),
			}
		}
		if (block.type === "tool_result") {
			return {
				type: "text" as const,
				text: toolResultToText(block),
			}
		}
		return block
	})
}

/**
 * Transforms all messages by converting tool_use and tool_result blocks to text representations.
 * This ensures the conversation can be sent for summarization without requiring the tools parameter.
 *
 * @param messages - The messages to transform
 * @returns The transformed messages with tool blocks converted to text
 */
export function transformMessagesForCondensing<
	T extends { role: string; content: string | Anthropic.Messages.ContentBlockParam[] },
>(messages: T[]): T[] {
	return messages.map((msg) => ({
		...msg,
		content: convertToolBlocksToText(msg.content),
	}))
}

/**
 * Removes reasoning content from messages so the condense request matches the size of a
 * normal request.
 *
 * Normal requests run `buildCleanConversationHistory()`, which strips plain-text
 * `{type:"reasoning"}` blocks when the model's `preserveReasoning` flag is not true. The
 * condense path did not do this, so it shipped every turn's reasoning on top of the
 * conversation — inflating the outgoing payload past the context window and causing
 * guaranteed condense failures (the payload was a superset of the last normal request).
 * Reasoning is irrelevant to summarization, so we drop standalone reasoning items and strip
 * all `{type:"reasoning"}` content blocks (plain-text and encrypted) here.
 *
 * @param messages - The conversation messages to clean.
 * @returns The messages with reasoning content removed.
 */
export function stripReasoningBlocks(messages: ApiMessage[]): ApiMessage[] {
	return messages
		.filter((msg) => msg.type !== "reasoning")
		.map((msg) => {
			if (!Array.isArray(msg.content)) {
				return msg
			}
			const blocks = msg.content as Array<{ type: string } & object>
			const kept = blocks.filter((block) => block.type !== "reasoning")
			if (kept.length === blocks.length) {
				return msg
			}
			if (kept.length === 1 && kept[0].type === "text") {
				return { ...msg, content: (kept[0] as Anthropic.Messages.TextBlockParam).text }
			}
			return { ...msg, content: kept as Anthropic.Messages.ContentBlockParam[] }
		})
}

export const MIN_CONDENSE_THRESHOLD = 5 // Minimum percentage of context window to trigger condensing
export const MAX_CONDENSE_THRESHOLD = 100 // Maximum percentage of context window to trigger condensing

const SUMMARY_PROMPT = `You are a helpful AI assistant tasked with summarizing conversations.

CRITICAL: This is a summarization-only request. DO NOT call any tools or functions.
Your ONLY task is to analyze the conversation and produce a text summary.
Respond with text only - no tool calls will be processed.

CRITICAL: This summarization request is a SYSTEM OPERATION, not a user message.
When analyzing "user requests" and "user intent", completely EXCLUDE this summarization message.
The "most recent user request" and "next step" must be based on what the user was doing BEFORE this system message appeared.
The goal is for work to continue seamlessly after condensation - as if it never happened.`

/**
 * Injects synthetic tool_results for orphan tool_calls that don't have matching results.
 * This is necessary because OpenAI's Responses API rejects conversations with orphan tool_calls.
 * This can happen when the user triggers condense after receiving a tool_call (like attempt_completion)
 * but before responding to it.
 *
 * @param messages - The conversation messages to process
 * @returns The messages with synthetic tool_results appended if needed
 */
export function injectSyntheticToolResults(messages: ApiMessage[]): ApiMessage[] {
	// Find all tool_call IDs in assistant messages
	const toolCallIds = new Set<string>()
	// Find all tool_result IDs in user messages
	const toolResultIds = new Set<string>()

	for (const msg of messages) {
		if (msg.role === "assistant" && Array.isArray(msg.content)) {
			for (const block of msg.content) {
				if (block.type === "tool_use") {
					toolCallIds.add(block.id)
				}
			}
		}
		if (msg.role === "user" && Array.isArray(msg.content)) {
			for (const block of msg.content) {
				if (block.type === "tool_result") {
					toolResultIds.add(block.tool_use_id)
				}
			}
		}
	}

	// Find orphans (tool_calls without matching tool_results)
	const orphanIds = [...toolCallIds].filter((id) => !toolResultIds.has(id))

	if (orphanIds.length === 0) {
		return messages
	}

	// Inject synthetic tool_results as a new user message
	const syntheticResults: Anthropic.Messages.ToolResultBlockParam[] = orphanIds.map((id) => ({
		type: "tool_result" as const,
		tool_use_id: id,
		content: "Context condensation triggered. Tool execution deferred.",
	}))

	const syntheticMessage: ApiMessage = {
		role: "user",
		content: syntheticResults,
		ts: Date.now(),
	}

	return [...messages, syntheticMessage]
}

/**
 * Removes `tool_result` blocks whose referenced `tool_use` blocks are not present in the
 * given message list (orphaned by a trim/condense boundary), dropping any user messages
 * that end up with no content.
 */
export function filterOrphanedToolResults(messages: ApiMessage[]): ApiMessage[] {
	const toolUseIds = new Set<string>()
	for (const msg of messages) {
		if (msg.role === "assistant" && Array.isArray(msg.content)) {
			for (const block of msg.content) {
				if (block.type === "tool_use" && block.id) {
					toolUseIds.add(block.id)
				}
			}
		}
	}

	return messages
		.map((msg) => {
			if (msg.role === "user" && Array.isArray(msg.content)) {
				const filteredContent = msg.content.filter((block) =>
					block.type === "tool_result" ? toolUseIds.has(block.tool_use_id) : true,
				)
				if (filteredContent.length === 0) {
					return null
				}
				if (filteredContent.length !== msg.content.length) {
					return { ...msg, content: filteredContent }
				}
			}
			return msg
		})
		.filter((msg): msg is ApiMessage => msg !== null)
}

/**
 * Unconditionally replaces image blocks with a "[Image]" text placeholder, both at the
 * top level of message content and inside tool_result content arrays.
 *
 * Condensing only needs the conversation text; multimodal servers expand image blocks
 * into large numbers of vision tokens, so raw image data in the condense input can push
 * the request past the model's enforced input limit.
 */
export function removeImageBlocks(messages: ApiMessage[]): ApiMessage[] {
	return messages.map((msg) => {
		if (!Array.isArray(msg.content)) {
			return msg
		}
		let changed = false
		const content = msg.content.map((block) => {
			if (block.type === "image") {
				changed = true
				return { type: "text" as const, text: "[Image]" }
			}
			if (block.type === "tool_result" && Array.isArray(block.content)) {
				let toolChanged = false
				const toolContent = block.content.map((item) => {
					if (item.type === "image") {
						toolChanged = true
						return { type: "text" as const, text: "[Image]" }
					}
					return item
				})
				if (toolChanged) {
					changed = true
					return { ...block, content: toolContent }
				}
			}
			return block
		})
		return changed ? { ...msg, content } : msg
	})
}

/**
 * Computes the maximum number of tokens the condense API call may send as input.
 *
 * Uses the same convention as `manageContext`: the model's input budget is the context
 * window minus the reserved output (the model's maxTokens, or the Anthropic default when
 * unavailable). Returns 0 when no usable budget can be derived (no context window known),
 * in which case callers should skip token-counting and trimming.
 */
export function getCondenseInputBudget(apiHandler: ApiHandler): number {
	const modelInfo = apiHandler.getModel().info
	const contextWindow = apiHandler.getCondenseContextWindow?.() ?? modelInfo.contextWindow
	if (!contextWindow || contextWindow <= 0) {
		return 0
	}
	const reservedOutput =
		modelInfo.maxTokens && modelInfo.maxTokens > 0 ? modelInfo.maxTokens : ANTHROPIC_DEFAULT_MAX_TOKENS
	const budget = contextWindow - reservedOutput
	return budget > 0 ? budget : 0
}

/**
 * Trims the oldest messages from a condense request body so that the total input
 * (body + instructions + summarizer system prompt) fits within `maxInputTokens`.
 * Returns the messages unchanged when they already fit. After trimming, tool_use /
 * tool_result integrity across the trim boundary is repaired: orphan `tool_result`
 * blocks are dropped and any trailing orphan `tool_use` blocks are given synthetic results.
 */
async function trimCondenseInputToFit(
	body: ApiMessage[],
	instructions: Anthropic.MessageParam,
	apiHandler: ApiHandler,
	maxInputTokens: number,
): Promise<ApiMessage[]> {
	if (body.length <= 1) {
		return body
	}

	const toBlocks = (content: ApiMessage["content"]): Anthropic.Messages.ContentBlockParam[] =>
		typeof content === "string" ? [{ type: "text", text: content }] : content

	const [bodyTokens, instructionsTokens, systemTokens] = await Promise.all([
		Promise.all(body.map((msg) => apiHandler.countTokens(toBlocks(msg.content)))),
		apiHandler.countTokens(toBlocks(instructions.content)),
		apiHandler.countTokens([{ type: "text", text: SUMMARY_PROMPT }]),
	])

	const budget = maxInputTokens - instructionsTokens - systemTokens
	const total = bodyTokens.reduce((sum, n) => sum + n, 0)
	if (budget <= 0 || total <= budget) {
		return body
	}

	let start = 0
	let remaining = total
	while (start < body.length - 1 && remaining - bodyTokens[start] > budget) {
		remaining -= bodyTokens[start]
		start++
	}

	const trimmed = body.slice(start)
	return injectSyntheticToolResults(filterOrphanedToolResults(trimmed))
}

/**
 * Extracts <command> blocks from a message's content.
 * These blocks represent active workflows that must be preserved across condensings.
 *
 * @param message - The message to extract command blocks from
 * @returns A string containing all command blocks found, or empty string if none
 */
export function extractCommandBlocks(message: ApiMessage): string {
	const content = message.content
	let text: string

	if (typeof content === "string") {
		text = content
	} else if (Array.isArray(content)) {
		// Concatenate all text blocks
		text = content
			.filter((block): block is Anthropic.Messages.TextBlockParam => block.type === "text")
			.map((block) => block.text)
			.join("\n")
	} else {
		return ""
	}

	// Match all <command> blocks including their content
	const commandRegex = /<command[^>]*>[\s\S]*?<\/command>/g
	const matches = text.match(commandRegex)

	if (!matches || matches.length === 0) {
		return ""
	}

	return matches.join("\n")
}

export type SummarizeResponse = {
	messages: ApiMessage[] // The messages after summarization
	summary: string // The summary text; empty string for no summary
	cost: number // The cost of the summarization operation
	newContextTokens?: number // The number of tokens in the context for the next API request
	error?: string // Populated iff the operation fails: error message shown to the user on failure (see Task.ts)
	errorDetails?: string // Detailed error information including stack trace and API error info
	condenseId?: string // The unique ID of the created Summary message, for linking to condense_context clineMessage
}

export type SummarizeConversationOptions = {
	messages: ApiMessage[]
	apiHandler: ApiHandler
	systemPrompt: string
	taskId: string
	isAutomaticTrigger?: boolean
	customCondensingPrompt?: string
	metadata?: ApiHandlerCreateMessageMetadata
	environmentDetails?: string
	filesReadByRoo?: string[]
	cwd?: string
	rooIgnoreController?: RooIgnoreController
}

/**
 * Returns true when an API error indicates the request was rejected because the payload
 * exceeded the model's context window (as opposed to, e.g., a rate limit or auth error).
 *
 * The full-context condense is always attempted first and only the server's rejection for
 * context size engages the chunked fallback, because the local tiktoken estimator is not
 * the model's tokenizer and is inaccurate for foreign tokenizers (e.g. Qwen) in both
 * directions — a local pre-flight estimate cannot reliably decide single-vs-chunked.
 *
 * Matches the observed rejection wording across OpenAI-compatible servers, e.g. llama.cpp's
 * `request (54390 tokens) exceeds the available context size (50176 tokens)`, plus common
 * variants from other OpenAI-compatible providers.
 */
export function isCondenseContextSizeError(error: unknown): boolean {
	if (!(error instanceof Error)) {
		return false
	}
	const message = (error.message ?? "").toLowerCase()

	// Phrasings that are unambiguously about context/payload size (no "context"
	// keyword required):
	// - "maximum context length is N tokens" (OpenAI)
	// - "too many tokens in request"
	// - "prompt is too long: N tokens > M maximum"
	// - "request (N tokens) exceeds the available context size (M tokens)" (llama.cpp)
	if (
		/maximum context length/.test(message) ||
		/too many tokens/.test(message) ||
		/prompt is too long/.test(message) ||
		/exceeds the available context size/.test(message)
	) {
		return true
	}

	// Other phrasings require the word "context" nearby to avoid false positives
	// (e.g. rate-limit or "request too large" errors that happen to mention context).
	const hasContext = /context/.test(message)
	if (
		hasContext &&
		// "exceeds the context window", "exceeded the current context window", etc.
		(/exceed\w*[^.!?]*context/.test(message) ||
			// "prompt is too long: N tokens > M maximum" (variant without "prompt")
			/too long[^.!?]*tokens/.test(message) ||
			// "context window is too small"
			/(context( window)? (is |is too )?too small)/.test(message) ||
			// "request does not fit within the context"
			/does not fit within/.test(message))
	) {
		return true
	}

	// Some SDKs attach the provider's error body separately (error.response / error.body).
	// Check those too, in case the top-level message is just "400 Bad Request".
	const anyError = error as unknown as Record<string, unknown>
	for (const key of ["response", "body"]) {
		const value = anyError[key]
		if (value && typeof value === "object") {
			try {
				const text = JSON.stringify(value).toLowerCase()
				if (/context/.test(text) && /exceed|maximum context length|too many tokens|too long/.test(text)) {
					return true
				}
			} catch {
				// Ignore serialization failures.
			}
		}
	}
	return false
}

/**
 * Fraction of the context window that a single chunk of a chunked (rolling) condense may
 * use. Each chunk request must also carry the running summary, the summary prompt, and the
 * condense instructions, so chunks are packed to a value below the full budget to leave
 * headroom.
 *
 * This is intentionally conservative (75%, not 90%): the per-message token counts that drive
 * the packing come from the local tiktoken estimator, which is not the model's own tokenizer
 * and can under- or over-count by a wide margin for foreign tokenizers. Packing to 75% leaves
 * a 25% margin so a chunk that the local estimator thinks fits is very likely to fit at the
 * server; the only cost of being conservative is one extra chunk in the worst case.
 */
const CONDENSE_CHUNK_CONTEXT_PERCENT = 0.75

/**
 * Splits a conversation into consecutive chunks, each within `chunkBudgetTokens`, for
 * chunked (rolling) condensing.
 *
 * Messages are packed greedily from the oldest. Two constraints shape the boundaries:
 *
 * - A chunk never ends on an assistant message that contains `tool_use` blocks (when one
 *   would otherwise be stranded at the boundary, it is moved to the start of the next
 *   chunk so it stays with its `tool_result`).
 * - A single message that alone exceeds the budget gets its own chunk and is reported
 *   via `allChunksFit: false` — it cannot be split, so chunking cannot help.
 *
 * @param messages - The (reasoning-stripped, tool-injected) messages to split.
 * @param apiHandler - The API handler used for local token counting.
 * @param chunkBudgetTokens - Maximum tokens for a single chunk.
 */
export async function splitMessagesIntoCondenseChunks(
	messages: ApiMessage[],
	apiHandler: ApiHandler,
	chunkBudgetTokens: number,
): Promise<{ chunks: ApiMessage[][]; allChunksFit: boolean }> {
	const flattenContent = (content: string | Anthropic.Messages.ContentBlockParam[]) =>
		typeof content === "string" ? [{ type: "text" as const, text: content }] : content

	const tokenCounts: number[] = []
	for (const msg of messages) {
		tokenCounts.push(await apiHandler.countTokens(flattenContent(msg.content)))
	}

	const endsWithPendingToolUse = (msg: ApiMessage): boolean => {
		if (msg.role !== "assistant" || !Array.isArray(msg.content)) {
			return false
		}
		return msg.content.some((block) => block.type === "tool_use")
	}

	const chunks: ApiMessage[][] = []
	let allChunksFit = true
	let i = 0
	while (i < messages.length) {
		let j = i
		let total = 0
		while (j < messages.length) {
			const candidate = total + tokenCounts[j]
			if (candidate > chunkBudgetTokens && j > i) {
				break
			}
			total = candidate
			j++
		}
		// Keep tool_use with its tool_result: don't end the chunk on a tool_use message.
		let end = j
		if (end > i && end < messages.length && end - i > 1 && endsWithPendingToolUse(messages[end - 1])) {
			end -= 1
		}
		const chunk = messages.slice(i, end)
		chunks.push(chunk)
		// A single-message chunk over budget means one message exceeds the whole window.
		if (chunk.length === 1 && tokenCounts[i] > chunkBudgetTokens) {
			allChunksFit = false
		}
		i = end
	}
	return { chunks, allChunksFit }
}

/**
 * Runs a single condense (summarization) request, draining the stream and accumulating
 * the summary text and cost. Returns an error (with details) instead of throwing. The
 * original `rawError` is also returned so callers can classify the failure (e.g. detect a
 * server context-size rejection via `isCondenseContextSizeError`).
 */
async function runCondenseRequest(
	apiHandler: ApiHandler,
	systemPrompt: string,
	requestMessages: Anthropic.Messages.MessageParam[],
	metadata?: ApiHandlerCreateMessageMetadata,
): Promise<{ summary: string; cost: number; error?: string; errorDetails?: string; rawError?: unknown }> {
	let summary = ""
	let cost = 0
	try {
		const stream = apiHandler.createMessage(systemPrompt, requestMessages, metadata)

		for await (const chunk of stream) {
			if (chunk.type === "text") {
				summary += chunk.text
			} else if (chunk.type === "usage") {
				// Record final usage chunk only
				cost = chunk.totalCost ?? 0
			}
		}
		return { summary, cost }
	} catch (error) {
		console.error("Error during condensing API call:", error)
		const errorMessage = error instanceof Error ? error.message : String(error)

		// Capture detailed error information for debugging
		let errorDetails = ""
		if (error instanceof Error) {
			errorDetails = `Error: ${error.message}`
			// Capture any additional API error properties
			const anyError = error as unknown as Record<string, unknown>
			if (anyError.status) {
				errorDetails += `\n\nHTTP Status: ${anyError.status}`
			}
			if (anyError.code) {
				errorDetails += `\nError Code: ${anyError.code}`
			}
			if (anyError.response) {
				try {
					errorDetails += `\n\nAPI Response:\n${JSON.stringify(anyError.response, null, 2)}`
				} catch {
					errorDetails += `\n\nAPI Response: [Unable to serialize]`
				}
			}
			if (anyError.body) {
				try {
					errorDetails += `\n\nResponse Body:\n${JSON.stringify(anyError.body, null, 2)}`
				} catch {
					errorDetails += `\n\nResponse Body: [Unable to serialize]`
				}
			}
		} else {
			errorDetails = String(error)
		}

		return {
			summary,
			cost,
			error: t("common:errors.condense_api_failed", { message: errorMessage }),
			errorDetails,
			rawError: error,
		}
	}
}

/**
 * Condenses a conversation that is too large for a single request by summarizing it in
 * chunks (rolling summary): each chunk request carries the previous chunks' summary plus
 * the next chunk of messages. The final chunk's summary is a global summary of the whole
 * conversation, so no messages are lost — the non-destructive end state (one summary
 * message + originals tagged `condenseParent`) is identical to a single condense.
 *
 * Returns an error (for the caller to fall back to truncation) when a single message
 * alone exceeds the window (cannot be split) or a chunk request fails.
 */
async function runChunkedCondense({
	messages,
	condenseInstructions,
	apiHandler,
	metadata,
}: {
	messages: ApiMessage[]
	condenseInstructions: string
	apiHandler: ApiHandler
	metadata?: ApiHandlerCreateMessageMetadata
}): Promise<{ summary?: string; cost: number; error?: string; errorDetails?: string }> {
	const modelInfo = apiHandler.getModel().info
	const contextWindow = apiHandler.getCondenseContextWindow?.() ?? modelInfo.contextWindow
	const reservedTokens =
		modelInfo.maxTokens && modelInfo.maxTokens > 0 ? modelInfo.maxTokens : ANTHROPIC_DEFAULT_MAX_TOKENS

	// Fixed overhead of each chunk request (system prompt + tool definitions + the
	// condense instructions), so chunks are packed to leave room for it plus the
	// running summary (covered by the CONDENSE_CHUNK_CONTEXT_PERCENT headroom).
	const overheadTokens =
		(await apiHandler.countTokens([{ type: "text", text: SUMMARY_PROMPT }])) +
		(metadata?.tools && metadata.tools.length > 0
			? await apiHandler.countTokens([{ type: "text", text: JSON.stringify(metadata.tools) }])
			: 0) +
		(await apiHandler.countTokens([{ type: "text", text: condenseInstructions }]))

	const chunkBudgetTokens =
		Math.floor(contextWindow * CONDENSE_CHUNK_CONTEXT_PERCENT) - reservedTokens - overheadTokens

	// Not even one message's worth of headroom: chunking cannot help.
	if (chunkBudgetTokens <= 0) {
		return {
			cost: 0,
			error: t("common:errors.condense_truncation_still_exceeds", { window: contextWindow }),
		}
	}

	const { chunks, allChunksFit } = await splitMessagesIntoCondenseChunks(messages, apiHandler, chunkBudgetTokens)

	// A single message alone exceeds the window — it cannot be split; chunking is
	// impossible, so the caller falls back to truncation with an actionable error.
	if (!allChunksFit) {
		return {
			cost: 0,
			error: t("common:errors.condense_truncation_still_exceeds", { window: contextWindow }),
		}
	}

	// Build the trailing instruction message for a chunk. For the last chunk, tell the
	// model to merge the earlier partial summary into one complete summary.
	const buildInstructionContent = (
		runningSummary: string,
		isLast: boolean,
	): Anthropic.Messages.ContentBlockParam[] => {
		const blocks: Anthropic.Messages.ContentBlockParam[] = []
		if (runningSummary) {
			blocks.push({
				type: "text",
				text: `## Summary of the earlier part of this conversation (from a previous chunk)\n${runningSummary}`,
			})
		}
		const instructionText = isLast
			? `${condenseInstructions}\n\nNOTE: This conversation is longer than the model's context window, so it is being summarized in chunks. The part of the conversation before the messages above was already summarized (provided in the block above, if present). Produce ONE complete summary of the ENTIRE conversation by merging that earlier summary with the messages above. Preserve all important details from both.`
			: `${condenseInstructions}\n\nNOTE: This conversation is longer than the model's context window, so it is being summarized in chunks. You are only seeing a part of the conversation here; more chunks will follow. Produce a detailed, complete summary of this part (including the earlier part, if a summary of it is provided in a block above). Do not conclude or wrap up the conversation; it continues in later chunks.`
		blocks.push({ type: "text", text: instructionText })
		return blocks
	}

	let runningSummary = ""
	let totalCost = 0

	for (let i = 0; i < chunks.length; i++) {
		const isLast = i === chunks.length - 1
		const chunkMessages: ApiMessage[] = [
			...chunks[i],
			{ role: "user", content: buildInstructionContent(runningSummary, isLast), ts: Date.now() },
		]

		// Strip image blocks so multimodal payloads cannot push a chunk over the limit.
		const requestMessages = transformMessagesForCondensing(removeImageBlocks(chunkMessages)).map(
			({ role, content }) => ({ role, content }),
		)

		const result = await runCondenseRequest(apiHandler, SUMMARY_PROMPT, requestMessages, metadata)
		totalCost += result.cost
		if (result.error || !result.summary?.trim()) {
			// A chunk failed; the conversation can't be fully summarized in chunks, so
			// let the caller fall back to truncation.
			return {
				cost: totalCost,
				error: result.error ?? t("common:errors.condense_failed"),
				errorDetails: result.errorDetails,
			}
		}
		runningSummary = result.summary.trim()
	}

	return { summary: runningSummary, cost: totalCost }
}

/**
 * Summarizes the conversation messages using an LLM call.
 *
 * This implements the "fresh start" model where:
 * - The summary becomes a user message (not assistant)
 * - Post-condense, the model sees only the summary (true fresh start)
 * - All messages are still stored but tagged with condenseParent
 * - <command> blocks from the original task are preserved across condensings
 * - File context (folded code definitions) can be preserved for continuity
 *
 * Environment details handling:
 * - For AUTOMATIC condensing (isAutomaticTrigger=true): Environment details are included
 *   in the summary because the API request is already in progress and the next user
 *   message won't have fresh environment details injected.
 * - For MANUAL condensing (isAutomaticTrigger=false): Environment details are NOT included
 *   because fresh environment details will be injected on the very next turn via
 *   getEnvironmentDetails() in recursivelyMakeClineRequests().
 */
export async function summarizeConversation(options: SummarizeConversationOptions): Promise<SummarizeResponse> {
	const {
		messages,
		apiHandler,
		systemPrompt,
		taskId,
		isAutomaticTrigger,
		customCondensingPrompt,
		metadata,
		environmentDetails,
		filesReadByRoo,
		cwd,
		rooIgnoreController,
	} = options
	TelemetryService.instance.captureContextCondensed(
		taskId,
		isAutomaticTrigger ?? false,
		!!customCondensingPrompt?.trim(),
	)

	const response: SummarizeResponse = { messages, cost: 0, summary: "" }

	// Validate that the API handler supports message creation before doing any work.
	if (!apiHandler || typeof apiHandler.createMessage !== "function") {
		console.error("API handler is invalid for condensing. Cannot proceed.")
		const error = t("common:errors.condense_handler_invalid")
		return { ...response, error }
	}

	// Get messages to summarize (all ACTIVE messages since the last summary, if any).
	// Base this on the effective history so that messages already hidden by a
	// sliding-window truncation marker (truncationParent) or an existing summary are
	// NOT re-sent in the condense request. Without this the condense call can include
	// far more than the active context and exceed the model's max input length.
	const messagesToSummarize = getMessagesSinceLastSummary(getEffectiveApiHistory(messages))

	if (messagesToSummarize.length <= 1) {
		const error =
			messages.length <= 1
				? t("common:errors.condense_not_enough_messages")
				: t("common:errors.condensed_recently")
		return { ...response, error }
	}

	// Check if there's a recent summary in the messages (edge case)
	const recentSummaryExists = messagesToSummarize.some((message: ApiMessage) => message.isSummary)

	if (recentSummaryExists && messagesToSummarize.length <= 2) {
		const error = t("common:errors.condensed_recently")
		return { ...response, error }
	}

	// Use custom prompt if provided and non-empty, otherwise use the default CONDENSE prompt
	// This respects user's custom condensing prompt setting
	const condenseInstructions = customCondensingPrompt?.trim() || supportPrompt.default.CONDENSE

	const finalRequestMessage: Anthropic.MessageParam = {
		role: "user",
		content: condenseInstructions,
	}

	// Strip reasoning so the condense payload matches a normal request (see stripReasoningBlocks):
	// normal requests clean reasoning blocks out of the history, so without this the condense
	// call ships every turn's reasoning on top of the conversation — inflating the payload past
	// the context window and causing guaranteed failures (issue #1342).
	// Then inject synthetic tool_results for orphan tool_calls to prevent API rejections
	// (e.g., when user triggers condense after receiving attempt_completion but before responding)
	let condenseBody = injectSyntheticToolResults(stripReasoningBlocks(messagesToSummarize))

	// Unconditionally strip image blocks from the condense input. Summarization only
	// needs the conversation text, and multimodal servers expand image blocks into
	// thousands of vision tokens — raw image data in the condense request can push it
	// past the model's enforced input limit even when the text alone would fit.
	condenseBody = removeImageBlocks(condenseBody)

	// Defensive guard: if the messages to summarize would still exceed the model's
	// input budget (countTokens is only an estimate), trim the oldest messages so the
	// condense call itself cannot exceed the max input length.
	const maxInputTokens = getCondenseInputBudget(apiHandler)
	if (maxInputTokens > 0) {
		condenseBody = await trimCondenseInputToFit(condenseBody, finalRequestMessage, apiHandler, maxInputTokens)
	}

	// Transform tool_use and tool_result blocks to text representations.
	// This is necessary because some providers (like Bedrock via LiteLLM) require the `tools` parameter
	// when tool blocks are present. By converting them to text, we can send the conversation for
	// summarization without needing to pass the tools parameter.
	const messagesWithTextToolBlocks = transformMessagesForCondensing([...condenseBody, finalRequestMessage])

	const requestMessages = messagesWithTextToolBlocks.map(({ role, content }) => ({ role, content }))

	let summary = ""
	let cost = 0

	// The condense request must NOT include tool definitions: tool blocks are
	// already converted to text (transformMessagesForCondensing) and SUMMARY_PROMPT
	// instructs the model not to call tools. Sending tool definitions inflates the
	// condense payload and can push the request over the model's context limit
	// (issue #11998). Strip tool-related fields, but keep tracking/abort fields.
	// The original `metadata` is retained below for the newContextTokens estimate.
	const condenseMetadata: ApiHandlerCreateMessageMetadata | undefined = metadata
		? {
				taskId: metadata.taskId,
				mode: metadata.mode,
				suppressPreviousResponseId: metadata.suppressPreviousResponseId,
				store: metadata.store,
				abortSignal: metadata.abortSignal,
			}
		: undefined

	// The full-context condense is always attempted first. This preserves the existing
	// behavior for providers/models that already condense the whole conversation in one
	// request; the chunked (rolling) fallback below is only engaged when the SERVER
	// rejects the full request for context size — the authoritative signal that it cannot
	// fit. The local token estimate (pre-flight trim above) is not the model's tokenizer
	// and is unreliable for foreign tokenizers, so a server rejection is the only reliable
	// trigger for chunking.
	const single = await runCondenseRequest(apiHandler, SUMMARY_PROMPT, requestMessages, condenseMetadata)

	if (single.error && isCondenseContextSizeError(single.rawError)) {
		// The server rejected the full-context condense because the payload exceeded the
		// context window. Condense in chunks (rolling summary) so no messages are lost.
		console.info("[condense] Server rejected the full condense (context too large); condensing in chunks.")
		const chunkResult = await runChunkedCondense({
			messages: condenseBody,
			condenseInstructions,
			apiHandler,
			metadata: condenseMetadata,
		})
		if (chunkResult.error || !chunkResult.summary?.trim()) {
			// Chunking could not produce a summary (e.g. a single message alone exceeds
			// the window, or a chunk request failed); fall back to truncation with a
			// clear, actionable message.
			console.warn("[condense] Chunked condense failed; falling back to truncation.")
			return {
				...response,
				cost: chunkResult.cost,
				error: chunkResult.error ?? t("common:errors.condense_failed"),
				errorDetails: chunkResult.errorDetails,
			}
		}
		summary = chunkResult.summary
		cost = chunkResult.cost
	} else if (single.error) {
		// Failed for a non-context-size reason — surface the original error and let
		// callers fall back to truncation.
		return {
			...response,
			cost: single.cost,
			error: single.error,
			errorDetails: single.errorDetails,
		}
	} else {
		summary = single.summary
		cost = single.cost
	}

	summary = summary.trim()

	if (summary.length === 0) {
		const error = t("common:errors.condense_failed")
		return { ...response, cost, error }
	}

	// Extract command blocks from the first message (original task)
	// These represent active workflows that must persist across condensings
	const firstMessage = messages[0]
	const commandBlocks = firstMessage ? extractCommandBlocks(firstMessage) : ""

	// Build the summary content as separate text blocks
	const summaryContent: Anthropic.Messages.ContentBlockParam[] = [
		{ type: "text", text: `## Conversation Summary\n${summary}` },
	]

	// Add command blocks (active workflows) in their own system-reminder block if present
	if (commandBlocks) {
		summaryContent.push({
			type: "text",
			text: `<system-reminder>
## Active Workflows
The following directives must be maintained across all future condensings:
${commandBlocks}
</system-reminder>`,
		})
	}

	// Generate and add folded file context (smart code folding) if file paths are provided
	// Each file gets its own <system-reminder> block as a separate content block
	if (filesReadByRoo && filesReadByRoo.length > 0 && cwd) {
		try {
			const foldedResult = await generateFoldedFileContext(filesReadByRoo, {
				cwd,
				rooIgnoreController,
			})
			if (foldedResult.sections.length > 0) {
				for (const section of foldedResult.sections) {
					if (section.trim()) {
						summaryContent.push({
							type: "text",
							text: section,
						})
					}
				}
			}
		} catch (error) {
			console.error("[summarizeConversation] Failed to generate folded file context:", error)
			// Continue without folded context - non-critical failure
		}
	}

	// Add environment details as a separate text block if provided AND this is an automatic trigger.
	// For manual condensing, fresh environment details will be injected on the next turn.
	// For automatic condensing, the API request is already in progress so we need them in the summary.
	if (isAutomaticTrigger && environmentDetails?.trim()) {
		summaryContent.push({
			type: "text",
			text: environmentDetails,
		})
	}

	// Generate a unique condenseId for this summary
	const condenseId = crypto.randomUUID()

	// Use the last message's timestamp + 1 to ensure unique timestamp for summary.
	// The summary goes at the end of all messages.
	const lastMsgTs = messages[messages.length - 1]?.ts ?? Date.now()

	const summaryMessage: ApiMessage = {
		role: "user", // Fresh start model: summary is a user message
		content: summaryContent,
		ts: lastMsgTs + 1, // Unique timestamp after last message
		isSummary: true,
		condenseId, // Unique ID for this summary, used to track which messages it replaces
	}

	// NON-DESTRUCTIVE CONDENSE:
	// Tag ALL existing messages with condenseParent so they are filtered out when
	// the effective history is computed. The summary message is the only message
	// that will be visible to the API after condensing (fresh start model).
	//
	// Storage structure after condense:
	// [msg1(parent=X), msg2(parent=X), ..., msgN(parent=X), summary(id=X)]
	//
	// Effective for API (filtered by getEffectiveApiHistory):
	// [summary]  ← Fresh start!

	// Tag ALL messages with condenseParent
	const newMessages = messages.map((msg) => {
		// If message already has a condenseParent, we leave it - nested condense is handled by filtering
		if (!msg.condenseParent) {
			return { ...msg, condenseParent: condenseId }
		}
		return msg
	})

	// Append the summary message at the end
	newMessages.push(summaryMessage)

	// Count the tokens in the context for the next API request
	// After condense, the context will contain: system prompt + summary + tool definitions
	const systemPromptMessage: ApiMessage = { role: "user", content: systemPrompt }

	// Count actual summaryMessage content directly instead of using outputTokens as a proxy
	// This ensures we account for wrapper text (## Conversation Summary, <system-reminder>, <environment_details>)
	const contextBlocks = [systemPromptMessage, summaryMessage].flatMap((message) =>
		typeof message.content === "string" ? [{ text: message.content, type: "text" as const }] : message.content,
	)

	const messageTokens = await apiHandler.countTokens(contextBlocks)

	// Count tool definition tokens if tools are provided
	let toolTokens = 0
	if (metadata?.tools && metadata.tools.length > 0) {
		const toolsText = JSON.stringify(metadata.tools)
		toolTokens = await apiHandler.countTokens([{ text: toolsText, type: "text" }])
	}

	const newContextTokens = messageTokens + toolTokens
	return { messages: newMessages, summary, cost, newContextTokens, condenseId }
}

/**
 * Returns the list of all messages since the last summary message, including the summary.
 * Returns all messages if there is no summary.
 *
 * Note: Summary messages are always created with role: "user" (fresh-start model),
 * so the first message since the last summary is guaranteed to be a user message.
 */
export function getMessagesSinceLastSummary(messages: ApiMessage[]): ApiMessage[] {
	const lastSummaryIndexReverse = [...messages].reverse().findIndex((message) => message.isSummary)

	if (lastSummaryIndexReverse === -1) {
		return messages
	}

	const lastSummaryIndex = messages.length - lastSummaryIndexReverse - 1
	return messages.slice(lastSummaryIndex)
}

/**
 * Filters the API conversation history to get the "effective" messages to send to the API.
 *
 * Fresh Start Model:
 * - When a summary exists, return only messages from the summary onwards (fresh start)
 * - Messages with a condenseParent pointing to an existing summary are filtered out
 *
 * Messages with a truncationParent that points to an existing truncation marker are also filtered out,
 * as they have been hidden by sliding window truncation.
 *
 * This allows non-destructive condensing and truncation where messages are tagged but not deleted,
 * enabling accurate rewind operations while still sending condensed/truncated history to the API.
 *
 * @param messages - The full API conversation history including tagged messages
 * @returns The filtered history that should be sent to the API
 */
export function getEffectiveApiHistory(messages: ApiMessage[]): ApiMessage[] {
	// Find the most recent summary message
	const lastSummary = findLast(messages, (msg) => msg.isSummary === true)

	if (lastSummary) {
		// Fresh start model: return only messages from the summary onwards
		const summaryIndex = messages.indexOf(lastSummary)
		let messagesFromSummary = messages.slice(summaryIndex)

		// Filter out orphan tool_result blocks that reference tool_use IDs from
		// messages that were condensed away (or trimmed before the summary)
		messagesFromSummary = filterOrphanedToolResults(messagesFromSummary)

		// Still need to filter out any truncated messages within this range
		const existingTruncationIds = new Set<string>()
		for (const msg of messagesFromSummary) {
			if (msg.isTruncationMarker && msg.truncationId) {
				existingTruncationIds.add(msg.truncationId)
			}
		}

		return messagesFromSummary.filter((msg) => {
			// Filter out truncated messages if their truncation marker exists
			if (msg.truncationParent && existingTruncationIds.has(msg.truncationParent)) {
				return false
			}
			return true
		})
	}

	// No summary - filter based on condenseParent and truncationParent as before
	// This handles the case of orphaned condenseParent tags (summary was deleted via rewind)

	// Collect all condenseIds of summaries that exist in the current history
	const existingSummaryIds = new Set<string>()
	// Collect all truncationIds of truncation markers that exist in the current history
	const existingTruncationIds = new Set<string>()

	for (const msg of messages) {
		if (msg.isSummary && msg.condenseId) {
			existingSummaryIds.add(msg.condenseId)
		}
		if (msg.isTruncationMarker && msg.truncationId) {
			existingTruncationIds.add(msg.truncationId)
		}
	}

	// Filter out messages whose condenseParent points to an existing summary
	// or whose truncationParent points to an existing truncation marker.
	// Messages with orphaned parents (summary/marker was deleted) are included.
	return messages.filter((msg) => {
		// Filter out condensed messages if their summary exists
		if (msg.condenseParent && existingSummaryIds.has(msg.condenseParent)) {
			return false
		}
		// Filter out truncated messages if their truncation marker exists
		if (msg.truncationParent && existingTruncationIds.has(msg.truncationParent)) {
			return false
		}
		return true
	})
}

/**
 * Cleans up orphaned condenseParent and truncationParent references after a truncation operation (rewind/delete).
 * When a summary message or truncation marker is deleted, messages that were tagged with its ID
 * should have their parent reference cleared so they become active again.
 *
 * This function should be called after any operation that truncates the API history
 * to ensure messages are properly restored when their summary or truncation marker is deleted.
 *
 * @param messages - The API conversation history after truncation
 * @returns The cleaned history with orphaned condenseParent and truncationParent fields cleared
 */
export function cleanupAfterTruncation(messages: ApiMessage[]): ApiMessage[] {
	// Collect all condenseIds of summaries that still exist
	const existingSummaryIds = new Set<string>()
	// Collect all truncationIds of truncation markers that still exist
	const existingTruncationIds = new Set<string>()

	for (const msg of messages) {
		if (msg.isSummary && msg.condenseId) {
			existingSummaryIds.add(msg.condenseId)
		}
		if (msg.isTruncationMarker && msg.truncationId) {
			existingTruncationIds.add(msg.truncationId)
		}
	}

	// Clear orphaned parent references for messages whose summary or truncation marker was deleted
	return messages.map((msg) => {
		let needsUpdate = false

		// Check for orphaned condenseParent
		if (msg.condenseParent && !existingSummaryIds.has(msg.condenseParent)) {
			needsUpdate = true
		}

		// Check for orphaned truncationParent
		if (msg.truncationParent && !existingTruncationIds.has(msg.truncationParent)) {
			needsUpdate = true
		}

		if (needsUpdate) {
			// Create a new object without orphaned parent references
			const { condenseParent, truncationParent, ...rest } = msg
			const result: ApiMessage = rest as ApiMessage

			// Keep condenseParent if its summary still exists
			if (condenseParent && existingSummaryIds.has(condenseParent)) {
				result.condenseParent = condenseParent
			}

			// Keep truncationParent if its truncation marker still exists
			if (truncationParent && existingTruncationIds.has(truncationParent)) {
				result.truncationParent = truncationParent
			}

			return result
		}
		return msg
	})
}
