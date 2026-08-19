/**
 * Default LLM generate function using OpenRouter via the openai npm package.
 *
 * OpenRouter provides a unified API compatible with the OpenAI SDK.
 * Set baseURL to https://openrouter.ai/api/v1 and use OpenRouter model
 * names (e.g., "openai/gpt-5.3-chat", "anthropic/claude-sonnet-4-20250514").
 */

import type { GenerateFn } from "./types.js";

/**
 * Create a GenerateFn that calls OpenRouter (or any OpenAI-compatible API).
 *
 * @param config.apiKey - API key. Default: process.env.OPENROUTER_API_KEY
 * @param config.model - Model name. Default: "openai/gpt-5.3-chat"
 * @param config.baseUrl - API base URL. Default: "https://openrouter.ai/api/v1"
 */
export function createOpenRouterGenerateFn(config?: {
  apiKey?: string;
  model?: string;
  baseUrl?: string;
}): GenerateFn {
  const apiKey = config?.apiKey ?? process.env.OPENROUTER_API_KEY;
  const model = config?.model ?? "openai/gpt-5.3-chat";
  const baseUrl = config?.baseUrl ?? "https://openrouter.ai/api/v1";

  if (!apiKey) {
    throw new Error(
      "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable " +
        "or pass apiKey in config."
    );
  }

  // Lazy import to avoid requiring openai when not used
  let clientPromise: Promise<any> | null = null;

  function getClient(): Promise<any> {
    if (!clientPromise) {
      clientPromise = import("openai").then((mod) => {
        const OpenAI = mod.default;
        return new OpenAI({
          apiKey,
          baseURL: baseUrl,
          defaultHeaders: {
            "HTTP-Referer": "https://hyperfocal.dev",
            "X-Title": "Hyperfocal Rubric Evaluator",
          },
        });
      });
    }
    return clientPromise;
  }

  return async (systemPrompt: string, userPrompt: string): Promise<string> => {
    const client = await getClient();
    const response = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0,
      response_format: { type: "json_object" },
    });

    const content = response.choices?.[0]?.message?.content;
    if (!content) {
      throw new Error("Empty response from judge LLM");
    }
    return content;
  };
}
