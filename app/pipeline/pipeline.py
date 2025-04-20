"""
Pipeline architecture for chat message processing
"""
from typing import List, Dict, Any, Optional, Callable
import logging
from abc import ABC, abstractmethod

class PipelineStep(ABC):
    """Abstract base class for pipeline steps"""
    
    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the context and return the updated context
        
        Args:
            context: The current context dictionary
            
        Returns:
            Updated context dictionary
        """
        pass
    
    def __str__(self) -> str:
        return self.__class__.__name__


class ChatContextPipeline:
    """
    Pipeline for processing user chat messages and building comprehensive context
    before sending to an LLM
    """
    
    def __init__(self, steps: Optional[List[PipelineStep]] = None):
        """
        Initialize the pipeline with processing steps
        
        Args:
            steps: List of pipeline steps
        """
        self.steps = steps or []
        self.logger = logging.getLogger(__name__)
    
    def add_step(self, step: PipelineStep) -> 'ChatContextPipeline':
        """
        Add a step to the pipeline
        
        Args:
            step: Pipeline step to add
            
        Returns:
            Self for chaining
        """
        self.steps.append(step)
        return self
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message through the pipeline
        
        Args:
            context: Initial context dictionary containing at minimum the user message
            
        Returns:
            Enriched context dictionary with all pipeline steps applied
        """
        current_context = context.copy()
        
        self.logger.info(f"Starting pipeline processing with {len(self.steps)} steps")
        
        # Process each step in sequence
        for i, step in enumerate(self.steps):
            step_name = str(step)
            self.logger.info(f"Executing pipeline step {i+1}/{len(self.steps)}: {step_name}")
            
            try:
                current_context = step.process(current_context)
                self.logger.info(f"Step {step_name} completed successfully")
            except Exception as e:
                self.logger.error(f"Error in pipeline step {step_name}: {str(e)}", exc_info=True)
                # Continue with pipeline despite errors unless critical
                if getattr(step, 'critical', False):
                    raise
        
        self.logger.info("Pipeline processing completed")
        return current_context
