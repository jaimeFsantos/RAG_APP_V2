class EnhancedQASystem(QASystem):
    """Enhanced QA system with advanced retrieval and context integration"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 25):
        super().__init__(chunk_size, chunk_overlap)
        self.context_manager = ContextManager()
        self.orchestrator = None  # Will be initialized with create_chain
        
    def add_prediction_context(self, prediction_result: Dict[str, Any], metadata: Dict[str, Any]):
        """Add price prediction context"""
        context = AnalysisContext(
            source_type='price_prediction',
            content=prediction_result,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.context_manager.add_context(context)
        
    def add_market_context(self, market_analysis: Dict[str, Any], metadata: Dict[str, Any]):
        """Add market analysis context"""
        context = AnalysisContext(
            source_type='market_analysis',
            content=market_analysis,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.context_manager.add_context(context)
        
    def add_model_context(self, model_analysis: Dict[str, Any], metadata: Dict[str, Any]):
        """Add model analysis context"""
        context = AnalysisContext(
            source_type='model_analysis',
            content=model_analysis,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.context_manager.add_context(context)
        
    def create_chain(self, sources: List[Dict[str, Union[str, List[str]]]]) -> Chain:
        """Create enhanced QA chain with advanced retrieval"""
        try:
            # Process source documents
            documents = self.process_sources(sources)
            
            # Initialize embedding model
            embedding_model = OllamaEmbeddings(
                model="nomic-embed-text"
            )
            
            # Initialize retrieval orchestrator
            self.orchestrator = RetrievalOrchestrator(
                embeddings=embedding_model,
                context_manager=self.context_manager
            )
            
            # Initialize vector store with documents
            self.orchestrator.initialize_retriever(documents)
            
            # Create enhanced prompt template
            template = """Analyze the following question using all available context:

Question: {question}

Context:
{context}

Please provide a comprehensive response that:
1. Addresses the specific question
2. Incorporates relevant predictions, market trends, and model insights
3. Provides specific data points and analysis results
4. Explains the confidence level and any limitations
5. Suggests related insights or considerations

Response format:
- Start with a direct answer
- Include supporting evidence and analysis
- Provide relevant metrics and predictions
- End with additional insights or recommendations
"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Create retrieval function
            def retrieve_and_format(question: str) -> str:
                relevant_docs = self.orchestrator.retrieve(question, top_k=5)
                return "\n\n".join([doc.page_content for doc in relevant_docs])
                
            # Create the chain
            chain = (
                {"context": retrieve_and_format, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            return chain
            
        except Exception as e:
            logger.error(f"Error creating enhanced chain: {e}")
            return None
            
    def update_context(self, new_data: Dict[str, Any], context_type: str):
        """Update context with new analysis results"""
        metadata = {
            'updated_at': datetime.now().isoformat(),
            'type': context_type
        }
        
        if context_type == 'price_prediction':
            self.add_prediction_context(new_data, metadata)
        elif context_type == 'market_analysis':
            self.add_market_context(new_data, metadata)
        elif context_type == 'model_analysis':
            self.add_model_context(new_data, metadata)
            
        logger.info(f"Updated {context_type} context")
