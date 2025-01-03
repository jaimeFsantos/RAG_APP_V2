"""
Advanced retrieval system integrating multiple analysis outputs with ranking and query planning.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging
from langchain.docstore.document import Document
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class AnalysisContext:
    """Container for different types of analysis outputs"""
    source_type: str
    content: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]

class QueryPlanner:
    """Plans query execution based on content understanding"""
    
    def __init__(self):
        self.query_templates = {
            'price_prediction': [
                "What were the key factors in predicting {subject}?",
                "How accurate was the prediction for {subject}?",
                "What market trends affected {subject}?"
            ],
            'market_analysis': [
                "What is the market distribution of {subject}?",
                "How does {subject} compare to market averages?",
                "What are the trends for {subject}?"
            ],
            'model_analysis': [
                "How well did the model perform for {subject}?",
                "What features were most important for {subject}?",
                "What is the confidence level for {subject}?"
            ]
        }
        
    def analyze_query(self, query: str) -> Dict[str, float]:
        """
        Analyze query to determine relevance to different analysis types
        
        Args:
            query: User query
            
        Returns:
            Dict mapping analysis types to relevance scores
        """
        scores = {}
        # Add your query analysis logic here
        return scores
        
    def generate_sub_queries(self, query: str, context_types: List[str]) -> List[str]:
        """Generate focused sub-queries based on analysis"""
        sub_queries = []
        query_scores = self.analyze_query(query)
        
        for context_type in context_types:
            if query_scores.get(context_type, 0) > 0.3:  # Relevance threshold
                templates = self.query_templates.get(context_type, [])
                for template in templates:
                    try:
                        sub_query = template.format(subject=query)
                        sub_queries.append(sub_query)
                    except Exception as e:
                        logger.warning(f"Failed to format query template: {e}")
                        
        return sub_queries

class ContextManager:
    """Manages analysis contexts and their integration"""
    
    def __init__(self):
        self.contexts: Dict[str, List[AnalysisContext]] = {
            'price_prediction': [],
            'market_analysis': [],
            'model_analysis': [],
            'data_analysis': []
        }
        
    def add_context(self, context: AnalysisContext):
        """Add new analysis context"""
        context_list = self.contexts.get(context.source_type, [])
        context_list.append(context)
        self.contexts[context.source_type] = context_list
        
    def get_relevant_contexts(self, 
                            query: str, 
                            context_types: Optional[List[str]] = None) -> List[Document]:
        """
        Get relevant contexts based on query
        
        Args:
            query: User query
            context_types: Optional list of context types to consider
            
        Returns:
            List of relevant documents
        """
        if context_types is None:
            context_types = list(self.contexts.keys())
            
        documents = []
        for context_type in context_types:
            contexts = self.contexts.get(context_type, [])
            for context in contexts:
                doc = Document(
                    page_content=str(context.content),
                    metadata={
                        'source_type': context_type,
                        'timestamp': context.timestamp.isoformat(),
                        **context.metadata
                    }
                )
                documents.append(doc)
                
        return documents

class RetrievalOrchestrator:
    """Orchestrates the retrieval process across different sources"""
    
    def __init__(self, embeddings, context_manager: ContextManager):
        self.embeddings = embeddings
        self.context_manager = context_manager
        self.query_planner = QueryPlanner()
        self.vector_store = None
        
    def initialize_retriever(self, documents: List[Document]):
        """Initialize vector store and retriever"""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents using advanced ranking
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            # Generate sub-queries
            sub_queries = self.query_planner.generate_sub_queries(
                query, 
                list(self.context_manager.contexts.keys())
            )
            
            # Get initial candidates
            candidates = []
            for sub_query in sub_queries:
                docs = self.vector_store.similarity_search(sub_query, k=3)
                candidates.extend(docs)
            
            # Rank candidates
            ranked_docs = self._rank_documents(candidates, query)
            
            # Add recent analysis contexts
            context_docs = self.context_manager.get_relevant_contexts(query)
            
            # Combine and deduplicate
            all_docs = self._deduplicate_documents(ranked_docs + context_docs)
            
            return all_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []
            
    def _rank_documents(self, 
                       documents: List[Document], 
                       query: str) -> List[Document]:
        """
        Rank documents using multiple factors
        
        Args:
            documents: List of candidate documents
            query: Original query
            
        Returns:
            Ranked list of documents
        """
        scores = []
        for doc in documents:
            # Calculate relevance score
            relevance = self.vector_store.similarity_search_with_score(
                query, k=1, filter={"id": doc.metadata.get("id")}
            )[0][1]
            
            # Factor in recency
            timestamp = datetime.fromisoformat(doc.metadata.get('timestamp', '2024-01-01'))
            recency = 1.0 / (1.0 + (datetime.now() - timestamp).days)
            
            # Consider source type relevance
            source_scores = self.query_planner.analyze_query(query)
            source_relevance = source_scores.get(doc.metadata.get('source_type'), 0.5)
            
            # Combine scores
            final_score = 0.5 * relevance + 0.3 * recency + 0.2 * source_relevance
            scores.append((doc, final_score))
            
        # Sort by score
        ranked_docs = [doc for doc, score in sorted(scores, key=lambda x: x[1], reverse=True)]
        return ranked_docs
        
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity"""
        unique_docs = []
        seen_contents = set()
        
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(content_hash)
                
        return unique_docs
