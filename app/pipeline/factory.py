"""
Factory for creating and configuring chat context pipelines
"""
from typing import Optional, List, Dict, Any
from app.pipeline.pipeline import ChatContextPipeline
from app.pipeline.steps.tokenizer import TokenizationStep
from app.pipeline.steps.vector_search import VectorSearchStep
from app.pipeline.steps.message_formatter import MessageFormatterStep
from app.pipeline.steps.history_retrieval import ChatHistoryStep
from app.pipeline.steps.context_builder import ContextBuilderStep
from app.pipeline.steps.prompt_template import PromptTemplateStep
from app.config import MAX_CONTEXT_CHUNKS, MAX_HISTORY_LENGTH

class PipelineFactory:
    """Factory for creating pre-configured chat context pipelines"""
    
    @staticmethod
    def create_default_pipeline() -> ChatContextPipeline:
        """
        Create a default pipeline with standard steps
        
        Returns:
            Configured ChatContextPipeline
        """
        pipeline = ChatContextPipeline()
        
        # Add pipeline steps in order of execution
        pipeline.add_step(TokenizationStep())
        pipeline.add_step(VectorSearchStep(max_results=MAX_CONTEXT_CHUNKS))
        pipeline.add_step(MessageFormatterStep())
        pipeline.add_step(ChatHistoryStep(max_history=MAX_HISTORY_LENGTH))
        pipeline.add_step(ContextBuilderStep())
        pipeline.add_step(PromptTemplateStep())
        
        return pipeline
    
    @staticmethod
    def create_minimal_pipeline() -> ChatContextPipeline:
        """
        Create a minimal pipeline with only essential steps
        
        Returns:
            Configured ChatContextPipeline
        """
        pipeline = ChatContextPipeline()
        
        # Add only essential pipeline steps
        pipeline.add_step(MessageFormatterStep())
        pipeline.add_step(ChatHistoryStep(max_history=MAX_HISTORY_LENGTH))
        pipeline.add_step(ContextBuilderStep(max_context_items=0))  # No context items
        pipeline.add_step(PromptTemplateStep())
        
        return pipeline
    
    @staticmethod
    def create_custom_pipeline(
        include_tokenization: bool = True,
        include_history: bool = True,
        max_context_items: int = MAX_CONTEXT_CHUNKS,
        max_history: int = MAX_HISTORY_LENGTH,
        custom_prompt_template: Optional[str] = None
    ) -> ChatContextPipeline:
        """
        Create a custom pipeline with configurable steps
        
        Args:
            include_tokenization: Whether to include tokenization step
            include_vector_search: Whether to include vector search step
            include_history: Whether to include chat history step
            max_context_items: Maximum number of context items to include
            max_history: Maximum number of history messages to include
            custom_prompt_template: Optional custom prompt template
            
        Returns:
            Configured ChatContextPipeline
        """
        pipeline = ChatContextPipeline()
        
        # Add optional tokenization step
        if include_tokenization:
            pipeline.add_step(TokenizationStep())
        
        # Always include vector search - it's a core feature
        pipeline.add_step(VectorSearchStep(max_results=max_context_items))
        
        # Always include message formatter
        pipeline.add_step(MessageFormatterStep())
        
        # Add optional history step
        if include_history:
            pipeline.add_step(ChatHistoryStep(max_history=max_history))
        
        # Always include context builder
        pipeline.add_step(ContextBuilderStep(
            max_context_items=max_context_items
        ))
        
        # Always include prompt template, optionally with custom template
        pipeline.add_step(PromptTemplateStep(template=custom_prompt_template))
        
        return pipeline
