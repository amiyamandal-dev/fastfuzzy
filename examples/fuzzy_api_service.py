#!/usr/bin/env python3
"""
Fuzzy Matching API Service

This script demonstrates how to build a web API service using FastAPI and fastfuzzy
for fuzzy string matching operations.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import fastfuzzy
import uvicorn
import time

# Initialize FastAPI app
app = FastAPI(
    title="FastFuzzy API Service",
    description="High-performance fuzzy string matching API using fastfuzzy",
    version="1.0.0"
)

# Initialize fastfuzzy components
fuzzy_ratio = fastfuzzy.FuzzyRatio()
string_matcher = fastfuzzy.StringMatcher()
process = fastfuzzy.process
utils = fastfuzzy.Utils()

# Pydantic models for request/response
class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    algorithm: str = "ratio"
    score_cutoff: Optional[float] = None

class SimilarityResponse(BaseModel):
    similarity: float
    algorithm: str
    execution_time_ms: float

class BatchSimilarityRequest(BaseModel):
    queries: List[str]
    targets: List[str]
    algorithm: str = "ratio"

class BatchSimilarityResponse(BaseModel):
    matrix: List[List[float]]
    algorithm: str
    execution_time_ms: float

class FuzzySearchRequest(BaseModel):
    query: str
    choices: List[str]
    limit: int = 10
    score_cutoff: float = 0.0
    scorer: str = "ratio"

class FuzzySearchResult(BaseModel):
    choice: str
    score: float
    index: int

class FuzzySearchResponse(BaseModel):
    results: List[FuzzySearchResult]
    query: str
    execution_time_ms: float

class TextProcessRequest(BaseModel):
    text: str
    operation: str = "default"

class TextProcessResponse(BaseModel):
    original: str
    processed: str
    operation: str
    execution_time_ms: float

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "FastFuzzy API Service",
        "version": "1.0.0",
        "description": "High-performance fuzzy string matching API"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """Calculate similarity between two strings."""
    start_time = time.perf_counter()
    
    try:
        # Select algorithm
        if request.algorithm == "ratio":
            similarity = fastfuzzy.ratio(request.text1, request.text2, request.score_cutoff)
        elif request.algorithm == "partial_ratio":
            similarity = fastfuzzy.partial_ratio(request.text1, request.text2, request.score_cutoff)
        elif request.algorithm == "token_sort_ratio":
            similarity = fastfuzzy.token_sort_ratio(request.text1, request.text2, request.score_cutoff)
        elif request.algorithm == "token_set_ratio":
            similarity = fastfuzzy.token_set_ratio(request.text1, request.text2, request.score_cutoff)
        elif request.algorithm == "wratio":
            similarity = fastfuzzy.wratio(request.text1, request.text2, request.score_cutoff)
        elif request.algorithm == "jaro_winkler":
            similarity = fastfuzzy.jaro_winkler(request.text1, request.text2) * 100
        elif request.algorithm == "levenshtein":
            max_len = max(len(request.text1), len(request.text2))
            if max_len == 0:
                similarity = 100.0
            else:
                distance = fastfuzzy.levenshtein(request.text1, request.text2)
                similarity = (1.0 - distance / max_len) * 100
        else:
            raise ValueError(f"Unknown algorithm: {request.algorithm}")
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return SimilarityResponse(
            similarity=similarity,
            algorithm=request.algorithm,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-similarity", response_model=BatchSimilarityResponse)
async def calculate_batch_similarity(request: BatchSimilarityRequest):
    """Calculate similarity matrix for batches of strings."""
    start_time = time.perf_counter()
    
    try:
        # Select algorithm
        if request.algorithm == "ratio":
            matrix = fastfuzzy.batch_ratio(request.queries, request.targets)
        elif request.algorithm == "levenshtein":
            matrix = fastfuzzy.batch_levenshtein(request.queries, request.targets)
        elif request.algorithm == "quick_ratio":
            matrix = fastfuzzy.batch_quick_ratio(request.queries, request.targets)
        else:
            raise ValueError(f"Unsupported batch algorithm: {request.algorithm}")
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return BatchSimilarityResponse(
            matrix=matrix,
            algorithm=request.algorithm,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/fuzzy-search", response_model=FuzzySearchResponse)
async def fuzzy_search(request: FuzzySearchRequest):
    """Perform fuzzy search in a list of choices."""
    start_time = time.perf_counter()
    
    try:
        # Perform fuzzy search
        results = process.extract(
            query=request.query,
            choices=request.choices,
            limit=request.limit,
            score_cutoff=request.score_cutoff,
            scorer=request.scorer
        )
        
        # Convert to response format
        search_results = [
            FuzzySearchResult(
                choice=choice,
                score=score,
                index=index
            )
            for choice, score, index in results
        ]
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return FuzzySearchResponse(
            results=search_results,
            query=request.query,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/text-process", response_model=TextProcessResponse)
async def text_process(request: TextProcessRequest):
    """Process text using various utilities."""
    start_time = time.perf_counter()
    
    try:
        # Select operation
        if request.operation == "default":
            processed = utils.default_process(request.text)
        elif request.operation == "ascii_only":
            processed = utils.ascii_only(request.text)
        elif request.operation == "trim_whitespace":
            processed = utils.trim_whitespace(request.text)
        elif request.operation == "remove_punctuation":
            processed = utils.remove_punctuation(request.text)
        else:
            raise ValueError(f"Unknown operation: {request.operation}")
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return TextProcessResponse(
            original=request.text,
            processed=processed,
            operation=request.operation,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/algorithms")
async def list_algorithms():
    """List available algorithms."""
    return {
        "similarity_algorithms": [
            "ratio",
            "partial_ratio",
            "token_sort_ratio",
            "token_set_ratio",
            "wratio",
            "jaro_winkler",
            "levenshtein"
        ],
        "batch_algorithms": [
            "ratio",
            "levenshtein",
            "quick_ratio"
        ],
        "scorers": [
            "ratio",
            "partial_ratio",
            "token_sort_ratio",
            "token_set_ratio",
            "wratio"
        ],
        "text_operations": [
            "default",
            "ascii_only",
            "trim_whitespace",
            "remove_punctuation"
        ]
    }

# Example usage documentation
@app.get("/examples")
async def get_examples():
    """Get example requests for API endpoints."""
    return {
        "similarity_example": {
            "endpoint": "/similarity",
            "method": "POST",
            "request_body": {
                "text1": "hello world",
                "text2": "hello word",
                "algorithm": "ratio",
                "score_cutoff": 0.0
            }
        },
        "batch_similarity_example": {
            "endpoint": "/batch-similarity",
            "method": "POST",
            "request_body": {
                "queries": ["hello", "world"],
                "targets": ["helo", "word", "hello world"],
                "algorithm": "ratio"
            }
        },
        "fuzzy_search_example": {
            "endpoint": "/fuzzy-search",
            "method": "POST",
            "request_body": {
                "query": "python programming",
                "choices": ["Python Programming", "Java Programming", "C++ Development"],
                "limit": 5,
                "score_cutoff": 60.0,
                "scorer": "ratio"
            }
        },
        "text_process_example": {
            "endpoint": "/text-process",
            "method": "POST",
            "request_body": {
                "text": "Hello, World!",
                "operation": "default"
            }
        }
    }

def run_demo():
    """Run a demo of the API service."""
    print("FastFuzzy API Service Demo")
    print("=" * 50)
    
    # Test data
    test_text1 = "hello world"
    test_text2 = "hello word"
    
    print(f"Testing similarity between '{test_text1}' and '{test_text2}':")
    
    # Test different algorithms
    algorithms = ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"]
    
    for algorithm in algorithms:
        try:
            similarity = fastfuzzy.ratio(test_text1, test_text2) if algorithm == "ratio" else \
                        fastfuzzy.partial_ratio(test_text1, test_text2) if algorithm == "partial_ratio" else \
                        fastfuzzy.token_sort_ratio(test_text1, test_text2) if algorithm == "token_sort_ratio" else \
                        fastfuzzy.token_set_ratio(test_text1, test_text2)
            
            print(f"  {algorithm}: {similarity:.1f}%")
        except Exception as e:
            print(f"  {algorithm}: Error - {e}")
    
    # Test fuzzy search
    print("\nTesting fuzzy search:")
    query = "python programming"
    choices = ["Python Programming", "Java Programming", "C++ Development", "JavaScript Framework"]
    
    results = process.extract(query, choices, limit=3)
    for choice, score, index in results:
        print(f"  {score:5.1f}% - {choice}")
    
    print("\nAPI Service is ready!")
    print("Run with: uvicorn fuzzy_api_service:app --reload")
    print("Docs available at: http://localhost:8000/docs")

if __name__ == "__main__":
    # If run directly, show demo
    run_demo()