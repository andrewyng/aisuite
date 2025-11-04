import { 
  ChatCompletionRequest, 
  ChatCompletionResponse, 
  ChatCompletionChunk,
  RequestOptions 
} from '../types';

export interface Provider {
  readonly name: string;
  
  chatCompletion(
    request: ChatCompletionRequest,
    options?: RequestOptions
  ): Promise<ChatCompletionResponse>;
  
  streamChatCompletion(
    request: ChatCompletionRequest,
    options?: RequestOptions
  ): AsyncIterable<ChatCompletionChunk>;
}