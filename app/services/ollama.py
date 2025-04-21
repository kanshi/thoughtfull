import requests
import json
from typing import Dict, List, Optional, Union, Any, Iterator

from app.config import OLLAMA_BASE_URL, DEFAULT_LLM_MODEL, CPU_LLM_MODEL, GPU_LLM_MODEL
from app.utils.logging import get_logger

class OllamaService:
    """Service for interacting with Ollama LLM API"""
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize Ollama service
        
        Args:
            model: Model name to use (defaults to configuration)
        """
        self.base_url = OLLAMA_BASE_URL
        self.model = model or DEFAULT_LLM_MODEL
        self.logger = get_logger(__name__)
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[List[Dict[str, str]]] = None,
                         system_prompt: Optional[str] = None,
                         stream: bool = False) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a response from the LLM
        
        Args:
            prompt: User message
            context: Optional conversation history
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing response content or an iterator of response chunks if streaming
        """
        url = f"{self.base_url}/api/chat"
        
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if provided
        if context:
            # Make sure we're only sending the actual messages, not the current message again
            for msg in context:
                if msg and isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }
        
        # Log the complete payload being sent to Ollama
        self.logger.info(f"Sending to Ollama model {self.model}:")
        self.logger.info(f"Messages: {json.dumps(messages, indent=2)}")
        
        try:
            if stream:
                # Return a generator that yields response chunks
                return self._stream_response(url, payload)
            else:
                # Return the complete response
                response = requests.post(url, json=payload)
                response.raise_for_status()
                response_json = response.json()
                
                # Log the response from Ollama
                self.logger.info(f"Received from Ollama: {json.dumps(response_json, indent=2)}")
                
                return response_json
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling Ollama API: {str(e)}")
            return {"error": f"Failed to generate response: {str(e)}"}
            
    def _stream_response(self, url: str, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Stream the response from the Ollama API
        
        Args:
            url: API endpoint URL
            payload: Request payload
            
        Yields:
            Response chunks as they are received
        """
        self.logger.info(f"Streaming request to {url} with model {payload.get('model')}")
        self.logger.debug(f"Payload: {json.dumps(payload)}")
        
        try:
            # Add a test response if Ollama is not available
            try:
                # Try to connect to Ollama
                # Log the complete request being sent to Ollama
                self.logger.info(f"Streaming request to Ollama model {self.model}:")
                self.logger.info(f"Messages: {json.dumps(payload['messages'], indent=2)}")
                
                with requests.post(url, json=payload, stream=True) as response:
                    if response.status_code != 200:
                        self.logger.error(f"Ollama returned status code: {response.status_code}")
                        # Use mock response instead
                        for chunk in self._generate_mock_response(payload):
                            yield chunk
                        return
                        
                    self.logger.info("Connected to Ollama, streaming response...")
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                self.logger.debug(f"Received chunk from Ollama: {json.dumps(chunk)}")
                                
                                # Ensure the chunk has the expected format
                                if "message" in chunk and "content" in chunk["message"]:
                                    content = chunk["message"]["content"]
                                    self.logger.debug(f"Extracted content: {content}")
                                    yield {"message": {"content": content}}
                                elif "response" in chunk:
                                    content = chunk["response"]
                                    self.logger.debug(f"Extracted response: {content}")
                                    yield {"message": {"content": content}}
                                else:
                                    # Return the raw chunk if we can't parse it
                                    self.logger.debug(f"Using raw chunk: {json.dumps(chunk)}")
                                    yield chunk
                            except json.JSONDecodeError as e:
                                self.logger.error(f"Error decoding JSON: {str(e)} - {line}")
                                yield {"error": f"Failed to decode response: {str(e)}"}
            except requests.exceptions.ConnectionError:
                self.logger.error("Cannot connect to Ollama API. Using fallback response.")
                # Use mock response
                for chunk in self._generate_mock_response(payload):
                    yield chunk
        except Exception as e:
            self.logger.error(f"Unexpected error in streaming: {str(e)}")
            yield {"error": f"Failed to stream response: {str(e)}"}
    
    def _generate_mock_response(self, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Generate a mock response when Ollama is not available
        
        Args:
            payload: The original request payload
            
        Yields:
            Mock response chunks
        """
        # Extract the user message from the payload
        user_message = ""
        conversation_history = []
        
        if "messages" in payload and len(payload["messages"]) > 0:
            # Get all messages to understand the conversation
            conversation_history = payload["messages"]
            # Get the latest user message
            for msg in reversed(payload["messages"]):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
        
        self.logger.info(f"Mock response generator received user message: {user_message}")
        
        # Generate a personalized response based on the user's query
        if user_message:
            # First word to indicate this is a mock response
            yield {"message": {"content": "[MOCK] "}}
            
            # Analyze the user's query to provide a more relevant response
            if "document" in user_message.lower() or "search" in user_message.lower() or "find" in user_message.lower():
                response = f"You asked: '{user_message}' which seems to be about document search. "
                response += "I would search your documents for relevant information, but the Ollama LLM service is not available. "
                response += "The semantic search feature is working, but I need the LLM to process and respond to your query properly."
            elif "hello" in user_message.lower() or "hi" in user_message.lower() or "hey" in user_message.lower():
                response = f"Hello! You said: '{user_message}'. I'd like to chat with you, but I'm currently running in mock mode "
                response += "because the Ollama LLM service is not available."
            elif "what" in user_message.lower() and ("you" in user_message.lower() or "your" in user_message.lower()):
                response = f"You asked: '{user_message}'. I am ThoughtFull, a semantic search assistant for your private documents. "
                response += "I can help you search through your documents and chat about their contents, but I need the Ollama LLM service to be running."
            elif "how" in user_message.lower() and "work" in user_message.lower():
                response = f"You asked: '{user_message}'. ThoughtFull works by using vector embeddings to find semantically similar content in your documents. "
                response += "When you ask a question, I search for relevant information and use an LLM to generate a helpful response. "
                response += "However, the LLM component (Ollama) is currently not available."
            else:
                response = f"You asked: '{user_message}'. I wish I could provide a proper response, but the Ollama LLM service is not available. "
                response += "Please check that Ollama is running (http://localhost:11434) and that the model "
                response += f"'{payload.get('model', 'unknown')}' is available. "
                response += "You can run 'ollama list' to see available models and 'ollama pull llama3' to download the llama3 model."
            
            # Yield the response word by word to simulate streaming
            for word in response.split():
                yield {"message": {"content": word + " "}}
        else:
            # Fallback if no user message was found
            yield {"message": {"content": "I couldn't understand your message. Please try again when the Ollama LLM service is available."}}
    
    def search_and_respond(self, 
                           query: str, 
                           search_results: List[Dict],
                          context: Optional[List[Dict[str, str]]] = None,
                          stream: bool = False) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a response based on search results
        
        Args:
            query: User query
            search_results: List of search results (can include both document and conversation results)
            context: Optional conversation history
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing response content or an iterator of response chunks if streaming
        """
        # Separate document and conversation results
        document_results = [result for result in search_results if result.get('type') == 'document']
        conversation_results = [result for result in search_results if result.get('type') == 'conversation']
        
        # Create document context section
        document_context = ""
        if document_results:
            document_context = "\n\n### RELEVANT DOCUMENTS:\n\n" + "\n\n".join([
                f"Document: {result['file_name']}\nContent: {result['content']}\nRelevance: {result['score']:.4f}"
                for result in document_results
            ])
        
        # Create conversation context section
        conversation_context = ""
        if conversation_results:
            conversation_context = "\n\n### RELEVANT PAST CONVERSATIONS:\n\n" + "\n\n".join([
                f"From: {result['role']}\nContent: {result['content']}\nRelevance: {result['score']:.4f}"
                for result in conversation_results
            ])
        
        # Combine both contexts
        context_text = document_context
        if document_context and conversation_context:
            context_text += "\n\n" + conversation_context
        elif conversation_context:
            context_text = conversation_context
        
        system_prompt = """You are an AI assistant that helps answer questions based on the provided context.
Base your answers primarily on the information in the documents and past conversations provided.
When citing information from documents, mention the document name.
Avoid talking too much about your sources without a specific reference.
If the provided context doesn't contain relevant information to answer the question, ignore that.
When referencing past conversations, you can refer to them as 'my previous conversation' or 'my last conversation'."""
        
        enhanced_prompt = f"""I have a question: {query}

Here is relevant context that might help answer the question:

{context_text}

Please provide a helpful answer based on this information."""
        
        return self.generate_response(
            prompt=enhanced_prompt, 
            context=context,
            system_prompt=system_prompt,
            stream=stream
        )
    
    @staticmethod
    def get_available_models() -> List[Dict[str, Any]]:
        """
        Get the list of available models from Ollama
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                return response.json().get("models", [])
            else:
                logging.error(f"Failed to get models: {response.status_code}")
                return []
        except Exception as e:
            logging.error(f"Error getting models: {str(e)}")
            return []
    
    @staticmethod
    def get_model_details(model_name: str) -> Dict[str, Any]:
        """
        Get details about a specific model
        
        Args:
            model_name: Name of the model to get details for
            
        Returns:
            Dictionary with model details
        """
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/show?name={model_name}")
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"Failed to get model details: {response.status_code}")
                return {}
        except Exception as e:
            logging.error(f"Error getting model details: {str(e)}")
            return {}
