export interface Word {
  text: string;
  start: number;
  end: number;
  speaker?: string;
  confidence?: number;
}

export interface Segment {
  text: string;
  start: number;
  end: number;
  speaker?: string;
}

export interface TranscriptionResult {
  text: string;
  language: string;
  confidence?: number;
  words: Word[];
  segments: Segment[];
}

export interface TranscriptionRequest {
  model: string; // "provider:model" format
  file: string | Buffer | Uint8Array;
  language?: string;
  timestamps?: boolean;
  word_confidence?: boolean;
  speaker_labels?: boolean;
  temperature?: number;
  // Provider-specific parameters
  [key: string]: any;
}
