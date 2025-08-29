import { 
  TranscriptionRequest, 
  TranscriptionResult,
  RequestOptions 
} from '../types';

export abstract class BaseASRProvider {
  abstract readonly name: string;
  
  abstract transcribe(
    request: TranscriptionRequest,
    options?: RequestOptions
  ): Promise<TranscriptionResult>;
  
  abstract validateParams(    
    params: { [key: string]: any }
  ): void;
  
  abstract translateParams(    
    params: { [key: string]: any }
  ): { [key: string]: any };
}