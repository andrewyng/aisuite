export interface GroqConfig {
  apiKey?: string;  // Optional - SDK reads from GROQ_API_KEY env var if not provided
  baseURL?: string;
  dangerouslyAllowBrowser?: boolean;
}
