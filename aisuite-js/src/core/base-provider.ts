import { 
  ChatCompletionRequest, 
  ChatCompletionResponse, 
  ChatCompletionChunk,
  RequestOptions 
} from '../types';

export abstract class BaseProvider {
  abstract readonly name: string;
  
  abstract chatCompletion(
    request: ChatCompletionRequest,
    options?: RequestOptions
  ): Promise<ChatCompletionResponse>;
  
  abstract streamChatCompletion(
    request: ChatCompletionRequest,
    options?: RequestOptions
  ): AsyncIterable<ChatCompletionChunk>;
}