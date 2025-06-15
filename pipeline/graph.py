import time
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from pipeline.state import PipelineState
from pipeline.nodes import PipelineNodes, should_use_cache, has_error

class TextToImagePipeline:
    def __init__(self):
        """Initialize the LangGraph pipeline"""
        self.nodes = PipelineNodes()
        self.graph = self._create_graph()
        
        # Add memory for conversation tracking
        self.checkpointer = InMemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        # Initialize the graph with our state
        graph = StateGraph(PipelineState)
        
        # Add nodes
        graph.add_node("process_input", self.nodes.process_input)
        graph.add_node("check_cache", self.nodes.check_cache)
        graph.add_node("generate_prompt", self.nodes.generate_prompt)
        graph.add_node("generate_image", self.nodes.generate_image)
        graph.add_node("update_cache", self.nodes.update_cache)
        graph.add_node("finalize_response", self.nodes.finalize_response)
        
        # Define the flow
        graph.add_edge(START, "process_input")
        graph.add_edge("process_input", "check_cache")
        
        # Conditional routing based on cache
        graph.add_conditional_edges(
            "check_cache",
            should_use_cache,
            {
                "generate_prompt": "generate_prompt",
                "generate_image": "generate_image"
            }
        )
        
        graph.add_edge("generate_prompt", "generate_image")
        graph.add_edge("generate_image", "update_cache")
        graph.add_edge("update_cache", "finalize_response")
        graph.add_edge("finalize_response", END)
        
        return graph
    
    async def process_request(self, 
                            user_id: str,
                            title: str, 
                            keywords: list, 
                            description: str = None) -> Dict[str, Any]:
        """Process a text-to-image request"""
        
        # Initialize state
        initial_state = PipelineState(
            user_id=user_id,
            title=title,
            keywords=keywords,
            description=description,
            processed_keywords=[],
            entities={},
            cache_key=None,
            cached_prompt=None,
            generated_prompt=None,
            prompt_complexity=None,
            image_url=None,
            image_data=None,
            processing_time=0.0,
            used_cache=False,
            error=None,
            start_time=time.time()
        )
        
        # Configure the run
        config = {"configurable": {"thread_id": f"user_{user_id}"}}
        
        try:
            # Run the pipeline
            result = await self.compiled_graph.ainvoke(initial_state, config)
            
            return {
                "success": not bool(result.get('error')),
                "image_data": result.get('image_data'),
                "image_url": result.get('image_url'),
                "prompt_used": result.get('cached_prompt') or result.get('generated_prompt'),
                "processing_time": result.get('processing_time', 0),
                "used_cache": result.get('used_cache', False),
                "error": result.get('error')
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Pipeline execution failed: {str(e)}",
                "processing_time": time.time() - initial_state.get('start_time', time.time()),
                "used_cache": False
            }
