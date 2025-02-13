// src/ai/providers/gemini.ts
import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai";
import { z } from 'zod';

export interface GeminiProviderOptions {
  apiKey: string;
  model?: string;
}

export class GeminiProvider {
  private genAI: GoogleGenerativeAI;
  private model: GenerativeModel;

  constructor(options: GeminiProviderOptions) {
    this.genAI = new GoogleGenerativeAI(options.apiKey);
    this.model = this.genAI.getGenerativeModel({ 
      model: options.model || "gemini-2.0-flash"
    });
  }

  async generateObject<T extends z.ZodType>({ 
    prompt, 
    schema,
    system = "",
  }: { 
    prompt: string;
    schema: T;
    system?: string;
  }) {
    // Combine system prompt and user prompt for Gemini
    const fullPrompt = `${system}\n\n${prompt}`;
    
    try {
      const result = await this.model.generateContent(fullPrompt);
      const text = result.response.text();
      
      // Try to parse the response as JSON
      try {
        const jsonStr = text.substring(
          text.indexOf('{'),
          text.lastIndexOf('}') + 1
        );
        const parsed = JSON.parse(jsonStr);
        return {
          object: schema.parse(parsed),
          raw: text
        };
      } catch (e) {
        throw new Error(`Failed to parse Gemini response as JSON: ${text}`);
      }
    } catch (error) {
      throw new Error(`Gemini API error: ${error}`);
    }
  }
}
