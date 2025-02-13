import { createOpenAI, type OpenAIProviderSettings } from '@ai-sdk/openai';
import { getEncoding } from 'js-tiktoken';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { RecursiveCharacterTextSplitter } from './text-splitter';

interface CustomOpenAIProviderSettings extends OpenAIProviderSettings {
  baseURL?: string;
}

// Initialize Gemini
const gemini = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

// Providers configuration
const providers = {
  openai: createOpenAI({
    apiKey: process.env.OPENAI_KEY!,
    baseURL: process.env.OPENAI_ENDPOINT || 'https://api.openai.com/v1',
  } as CustomOpenAIProviderSettings),
  
  gemini: (model: string) => ({
    async generate(params: any) {
      const genModel = gemini.getGenerativeModel({ model });
      const result = await genModel.generateContent(params.prompt);
      const response = await result.response;
      return {
        content: response.text(),
        provider: 'gemini'
      };
    }
  })
};

// Select provider based on environment config
const selectedProvider = process.env.AI_PROVIDER?.toLowerCase() || 'openai';
const provider = providers[selectedProvider as keyof typeof providers];

if (!provider) {
  throw new Error(`Invalid AI provider: ${selectedProvider}`);
}

const customModel = process.env.OPENAI_MODEL || 'o3-mini';

// Export the model with the selected provider
export const o3MiniModel = provider(
  selectedProvider === 'gemini' ? 'gemini-pro' : customModel, 
  {
    reasoningEffort: customModel.startsWith('o') ? 'medium' : undefined,
    structuredOutputs: true,
  }
);

const MinChunkSize = 140;
const encoder = getEncoding('o200k_base');

// trim prompt to maximum context size
export function trimPrompt(
  prompt: string,
  contextSize = Number(process.env.CONTEXT_SIZE) || 128_000,
) {
  if (!prompt) {
    return '';
  }

  const length = encoder.encode(prompt).length;
  if (length <= contextSize) {
    return prompt;
  }

  const overflowTokens = length - contextSize;
  // on average it's 3 characters per token, so multiply by 3 to get a rough estimate of the number of characters
  const chunkSize = prompt.length - overflowTokens * 3;
  if (chunkSize < MinChunkSize) {
    return prompt.slice(0, MinChunkSize);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap: 0,
  });
  const trimmedPrompt = splitter.splitText(prompt)[0] ?? '';

  // last catch, there's a chance that the trimmed prompt is same length as the original prompt, due to how tokens are split & innerworkings of the splitter, handle this case by just doing a hard cut
  if (trimmedPrompt.length === prompt.length) {
    return trimPrompt(prompt.slice(0, chunkSize), contextSize);
  }

  // recursively trim until the prompt is within the context size
  return trimPrompt(trimmedPrompt, contextSize);
}
