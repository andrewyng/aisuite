export interface Tool {
  type: 'function';
  function: FunctionDefinition;
}

export interface FunctionDefinition {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, any>;
    required?: string[];
  };
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;  // JSON string
  };
}

export type ToolChoice = 
  | 'auto' 
  | 'none' 
  | { type: 'function'; function: { name: string } };