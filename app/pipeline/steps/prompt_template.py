"""
Prompt template step for the chat context pipeline
"""
from typing import Dict, Any, List, Optional
from app.pipeline.pipeline import PipelineStep

class PromptTemplateStep(PipelineStep):
    """
    Formats the context into a final prompt template for the LLM
    """
    
    def __init__(self, template: Optional[str] = None):
        """
        Initialize the prompt template step
        
        Args:
            template: Optional custom template string
        """
        self.template = template or self._default_template()
    
    def _default_template(self) -> str:
        """
        Get the default prompt template
        
        Returns:
            Default template string
        """
        return """
        You are a helpful AI assistant with access to a knowledge base and past conversations.
        
        {{#if system_message}}
        {system_message.content}
        {{/if}}
        
        {{#if document_context}}
        Here is some relevant information from your knowledge base that might help you answer:
        {document_context}
        {{/if}}
        
        {{#if conversation_context}}
        Here are some relevant parts from previous conversations that might be helpful:
        {conversation_context}
        {{/if}}
        
        {{#if history}}
        Conversation history:
        {{#each history}}
        {role}: {content}
        {{/each}}
        {{/if}}
        
        Please respond to the user's message:
        User: {current_message.content}
        Assistant:
        """.strip()
    
    def format_prompt(self, built_context: Dict[str, Any]) -> str:
        """
        Format the built context into a prompt using the template
        
        Args:
            built_context: Built context from previous pipeline steps
            
        Returns:
            Formatted prompt string
        """
        # This is a simplified template rendering implementation
        # In a real application, use a proper template engine like Jinja2
        
        prompt = self.template
        
        # Handle system message
        if "system_message" in built_context:
            system_content = built_context["system_message"]["content"]
            prompt = prompt.replace("{system_message.content}", system_content)
            prompt = prompt.replace("{{#if system_message}}", "")
            prompt = prompt.replace("{{/if}}", "")
        else:
            # Remove system message section
            start = prompt.find("{{#if system_message}}")
            end = prompt.find("{{/if}}", start) + 7
            if start != -1 and end != -1:
                prompt = prompt[:start] + prompt[end:]
        
        # Handle document context
        if "document_context" in built_context:
            document_context = built_context["document_context"]
            prompt = prompt.replace("{document_context}", document_context)
            prompt = prompt.replace("{{#if document_context}}", "")
            prompt = prompt.replace("{{/if}}", "")
        else:
            # Remove document context section
            start = prompt.find("{{#if document_context}}")
            end = prompt.find("{{/if}}", start) + 7
            if start != -1 and end != -1:
                prompt = prompt[:start] + prompt[end:]
        
        # Handle conversation context
        if "conversation_context" in built_context:
            conversation_context = built_context["conversation_context"]
            prompt = prompt.replace("{conversation_context}", conversation_context)
            prompt = prompt.replace("{{#if conversation_context}}", "")
            prompt = prompt.replace("{{/if}}", "")
        else:
            # Remove conversation context section
            start = prompt.find("{{#if conversation_context}}")
            end = prompt.find("{{/if}}", start) + 7
            if start != -1 and end != -1:
                prompt = prompt[:start] + prompt[end:]
        
        # Handle conversation history
        if "history" in built_context and built_context["history"]:
            history_text = ""
            for msg in built_context["history"]:
                history_text += f"{msg['role']}: {msg['content']}\n"
            
            prompt = prompt.replace("{{#each history}}\n{role}: {content}\n{{/each}}", history_text.strip())
            prompt = prompt.replace("{{#if history}}", "")
            prompt = prompt.replace("{{/if}}", "")
        else:
            # Remove history section
            start = prompt.find("{{#if history}}")
            end = prompt.find("{{/if}}", start) + 7
            if start != -1 and end != -1:
                prompt = prompt[:start] + prompt[end:]
        
        # Handle current message
        if "current_message" in built_context:
            current_content = built_context["current_message"]["content"]
            prompt = prompt.replace("{current_message.content}", current_content)
        
        return prompt
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the context to generate a final prompt for the LLM
        
        Args:
            context: The current pipeline context
            
        Returns:
            Updated context with formatted prompt
        """
        # Get the built context
        built_context = context.get('built_context', {})
        
        # Format the prompt
        formatted_prompt = self.format_prompt(built_context)
        
        # Add the formatted prompt to the context
        context['formatted_prompt'] = formatted_prompt
        
        return context
