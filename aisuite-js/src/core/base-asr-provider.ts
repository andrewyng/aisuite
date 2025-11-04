import { 
  TranscriptionRequest, 
  TranscriptionResult,
  RequestOptions 
} from '../types';

export interface ASRProvider {
  readonly name: string;
  
  transcribe(
    request: TranscriptionRequest,
    options?: RequestOptions
  ): Promise<TranscriptionResult>;
  
  validateParams(    
    params: { [key: string]: any }
  ): void;
  
  translateParams(    
    params: { [key: string]: any }
  ): { [key: string]: any };
}