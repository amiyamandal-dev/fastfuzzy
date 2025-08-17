#!/usr/bin/env python3
"""
Log Analysis with FastFuzzy

This script demonstrates how to use fastfuzzy for log analysis tasks,
such as grouping similar log messages and identifying patterns.
"""

import fastfuzzy
import re
import json
import time
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter

@dataclass
class LogEntry:
    """Represents a log entry."""
    timestamp: str
    level: str
    message: str
    source: str
    thread: str = ""
    
    def __str__(self):
        return f"[{self.timestamp}] {self.level} {self.source} - {self.message}"

class LogAnalyzer:
    """Analyzes logs using fuzzy matching to group similar messages."""
    
    def __init__(self, similarity_threshold: float = 85.0):
        self.similarity_threshold = similarity_threshold
        self.fuzzy = fastfuzzy.FuzzyRatio()
        self.process = fastfuzzy.process
        self.utils = fastfuzzy.Utils()
        
        # Templates for common log patterns
        self.log_templates = [
            r"Error connecting to database.*",
            r"Failed to process request.*",
            r"Timeout occurred while.*",
            r"Invalid input parameter.*",
            r"User authentication failed.*",
            r"File not found.*",
            r"Memory usage exceeded.*",
            r"Database query took.*",
            r"Cache miss for key.*",
            r"Rate limit exceeded.*"
        ]
    
    def preprocess_message(self, message: str) -> str:
        """Preprocess log message for comparison."""
        # Remove variable parts (timestamps, IDs, etc.)
        message = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '', message)  # Timestamps
        message = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '', message)  # UUIDs
        message = re.sub(r'\b\d+\b', '', message)  # Numbers
        message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', message)  # URLs
        message = re.sub(r'\S+@\S+\.\S+', '', message)  # Email addresses
        
        # Use fastfuzzy utils for standard processing
        return self.utils.default_process(message)
    
    def extract_template_candidates(self, message: str) -> List[str]:
        """Extract potential template candidates from a message."""
        candidates = []
        
        # Look for common patterns
        for template in self.log_templates:
            if re.search(template, message, re.IGNORECASE):
                candidates.append(template)
        
        # Extract key phrases
        words = message.split()
        if len(words) > 3:
            # Try first few words as template
            candidates.append(" ".join(words[:5]) + "...")
        
        return candidates
    
    def calculate_message_similarity(self, msg1: str, msg2: str) -> float:
        """Calculate similarity between two log messages."""
        # Preprocess messages
        processed1 = self.preprocess_message(msg1)
        processed2 = self.preprocess_message(msg2)
        
        if not processed1 or not processed2:
            return 0.0
        
        # Try multiple similarity measures
        ratio = self.fuzzy.ratio(processed1, processed2) or 0.0
        partial = self.fuzzy.partial_ratio(processed1, processed2) or 0.0
        token_sort = self.fuzzy.token_sort_ratio(processed1, processed2) or 0.0
        token_set = self.fuzzy.token_set_ratio(processed1, processed2) or 0.0
        
        # Weighted combination
        return ratio * 0.4 + partial * 0.3 + token_sort * 0.2 + token_set * 0.1
    
    def group_similar_messages(self, logs: List[LogEntry]) -> Dict[str, List[LogEntry]]:
        """Group similar log messages together."""
        print(f"Grouping {len(logs)} log messages...")
        
        groups: Dict[str, List[LogEntry]] = {}
        used_indices: Set[int] = set()
        
        # Sort by message length for better grouping
        sorted_logs = sorted(enumerate(logs), key=lambda x: len(x[1].message))
        
        for i, log1 in sorted_logs:
            if i in used_indices:
                continue
            
            # Start a new group
            group_key = log1.message[:50] + "..." if len(log1.message) > 50 else log1.message
            group = [log1]
            used_indices.add(i)
            
            # Find similar messages
            for j, log2 in sorted_logs[i+1:]:
                if j in used_indices:
                    continue
                
                similarity = self.calculate_message_similarity(log1.message, log2.message)
                if similarity >= self.similarity_threshold:
                    group.append(log2)
                    used_indices.add(j)
            
            # Only create groups with multiple entries or significant single entries
            if len(group) > 1 or (len(group) == 1 and len(group[0].message) > 30):
                groups[group_key] = group
        
        return groups
    
    def identify_frequent_patterns(self, logs: List[LogEntry], top_n: int = 10) -> List[Tuple[str, int]]:
        """Identify the most frequent log patterns."""
        # Preprocess all messages
        processed_messages = [self.preprocess_message(log.message) for log in logs]
        
        # Count occurrences
        pattern_counts = Counter(processed_messages)
        
        # Return top patterns
        return pattern_counts.most_common(top_n)
    
    def find_anomalies(self, logs: List[LogEntry], threshold: float = 2.0) -> List[LogEntry]:
        """Find anomalous log entries that don't match common patterns."""
        if not logs:
            return []
        
        # Identify common patterns
        frequent_patterns = self.identify_frequent_patterns(logs, top_n=5)
        common_patterns = [pattern for pattern, count in frequent_patterns]
        
        anomalies = []
        for log in logs:
            processed_msg = self.preprocess_message(log.message)
            is_anomaly = True
            
            # Check against common patterns
            for pattern in common_patterns:
                similarity = self.calculate_message_similarity(processed_msg, pattern)
                if similarity >= 80.0:  # High similarity to common pattern
                    is_anomaly = False
                    break
            
            if is_anomaly:
                anomalies.append(log)
        
        return anomalies
    
    def generate_summary(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """Generate a summary of log analysis."""
        if not logs:
            return {"total_logs": 0}
        
        # Basic statistics
        total_logs = len(logs)
        level_counts = Counter(log.level for log in logs)
        source_counts = Counter(log.source for log in logs)
        
        # Group similar messages
        groups = self.group_similar_messages(logs)
        
        # Identify frequent patterns
        frequent_patterns = self.identify_frequent_patterns(logs, top_n=5)
        
        # Find anomalies
        anomalies = self.find_anomalies(logs)
        
        return {
            "total_logs": total_logs,
            "level_distribution": dict(level_counts),
            "top_sources": dict(source_counts.most_common(5)),
            "similar_message_groups": len(groups),
            "frequent_patterns": frequent_patterns,
            "anomalies_count": len(anomalies),
            "anomaly_percentage": (len(anomalies) / total_logs * 100) if total_logs > 0 else 0
        }

def generate_sample_logs() -> List[LogEntry]:
    """Generate sample log entries for testing."""
    sample_logs = [
        LogEntry("2023-01-01T10:00:00", "INFO", "User login successful for user123", "auth-service", "thread-1"),
        LogEntry("2023-01-01T10:01:00", "INFO", "User login successful for user456", "auth-service", "thread-2"),
        LogEntry("2023-01-01T10:02:00", "ERROR", "Failed to connect to database at 192.168.1.100", "db-service", "thread-1"),
        LogEntry("2023-01-01T10:03:00", "ERROR", "Database connection failed for host 192.168.1.101", "db-service", "thread-2"),
        LogEntry("2023-01-01T10:04:00", "WARN", "High memory usage detected: 85%", "monitoring", "thread-1"),
        LogEntry("2023-01-01T10:05:00", "WARN", "Memory usage spike to 92%", "monitoring", "thread-2"),
        LogEntry("2023-01-01T10:06:00", "ERROR", "Timeout occurred while processing request /api/users", "api-service", "thread-1"),
        LogEntry("2023-01-01T10:07:00", "ERROR", "Request timeout for endpoint /api/orders", "api-service", "thread-2"),
        LogEntry("2023-01-01T10:08:00", "INFO", "Cache cleared successfully", "cache-service", "thread-1"),
        LogEntry("2023-01-01T10:09:00", "DEBUG", "Processing batch job #12345", "batch-service", "thread-1"),
        LogEntry("2023-01-01T10:10:00", "DEBUG", "Batch job #12346 started", "batch-service", "thread-2"),
        LogEntry("2023-01-01T10:11:00", "ERROR", "Invalid input parameter 'email' in request", "validation", "thread-1"),
        LogEntry("2023-01-01T10:12:00", "ERROR", "Validation failed for parameter 'username'", "validation", "thread-2"),
        LogEntry("2023-01-01T10:13:00", "WARN", "Rate limit exceeded for IP 192.168.1.50", "rate-limiter", "thread-1"),
        LogEntry("2023-01-01T10:14:00", "WARN", "Too many requests from 10.0.0.25", "rate-limiter", "thread-2"),
        # Some unique/error logs
        LogEntry("2023-01-01T10:15:00", "ERROR", "Unexpected exception in payment processing module", "payment-service", "thread-1"),
        LogEntry("2023-01-01T10:16:00", "FATAL", "System crash detected, restarting services", "system-monitor", "thread-1"),
        LogEntry("2023-01-01T10:17:00", "ERROR", "Security breach attempt from external source", "security", "thread-1"),
    ]
    
    return sample_logs

def demo_log_analysis():
    """Demonstrate log analysis capabilities."""
    print("=== Log Analysis Demo ===")
    
    # Generate sample logs
    logs = generate_sample_logs()
    print(f"Generated {len(logs)} sample log entries")
    
    # Initialize analyzer
    analyzer = LogAnalyzer(similarity_threshold=80.0)
    
    # Generate summary
    print("\nGenerating log summary...")
    start_time = time.perf_counter()
    summary = analyzer.generate_summary(logs)
    summary_time = time.perf_counter() - start_time
    
    print(f"Summary generated in {summary_time:.4f} seconds")
    print(f"Total logs: {summary['total_logs']}")
    print(f"Level distribution: {summary['level_distribution']}")
    print(f"Top sources: {summary['top_sources']}")
    print(f"Similar message groups: {summary['similar_message_groups']}")
    print(f"Anomalies: {summary['anomalies_count']} ({summary['anomaly_percentage']:.1f}%)")
    
    # Show frequent patterns
    print("\nFrequent Patterns:")
    for pattern, count in summary['frequent_patterns']:
        print(f"  {count:3d} occurrences - {pattern}")
    
    # Group similar messages
    print("\nGrouping similar messages...")
    start_time = time.perf_counter()
    groups = analyzer.group_similar_messages(logs)
    grouping_time = time.perf_counter() - start_time
    
    print(f"Grouped messages in {grouping_time:.4f} seconds")
    print(f"Found {len(groups)} groups:")
    
    for i, (group_key, group_logs) in enumerate(groups.items(), 1):
        print(f"\n  Group {i}: {len(group_logs)} similar messages")
        print(f"    Template: {group_key}")
        for log in group_logs[:3]:  # Show first 3
            print(f"    - {log}")
        if len(group_logs) > 3:
            print(f"    ... and {len(group_logs) - 3} more")
    
    # Find anomalies
    print("\nFinding anomalies...")
    start_time = time.perf_counter()
    anomalies = analyzer.find_anomalies(logs)
    anomaly_time = time.perf_counter() - start_time
    
    print(f"Found {len(anomalies)} anomalies in {anomaly_time:.4f} seconds:")
    for anomaly in anomalies:
        print(f"  - {anomaly}")

def demo_performance():
    """Demonstrate performance with larger dataset."""
    print("\n=== Performance Demo ===")
    
    # Generate larger dataset
    base_logs = generate_sample_logs()
    # Multiply to create larger dataset
    large_logs = []
    for i in range(100):  # 100x = ~1800 logs
        for log in base_logs:
            # Modify slightly to create variations
            modified_log = LogEntry(
                timestamp=log.timestamp,
                level=log.level,
                message=log.message + f" variation {i}",
                source=log.source,
                thread=log.thread
            )
            large_logs.append(modified_log)
    
    print(f"Testing with {len(large_logs)} log entries")
    
    analyzer = LogAnalyzer(similarity_threshold=85.0)
    
    # Time grouping
    start_time = time.perf_counter()
    groups = analyzer.group_similar_messages(large_logs[:500])  # Test with first 500
    grouping_time = time.perf_counter() - start_time
    
    print(f"Grouped 500 logs in {grouping_time:.4f} seconds")
    print(f"Found {len(groups)} groups")
    
    # Time pattern identification
    start_time = time.perf_counter()
    patterns = analyzer.identify_frequent_patterns(large_logs[:500])
    pattern_time = time.perf_counter() - start_time
    
    print(f"Identified frequent patterns in {pattern_time:.4f} seconds")
    print(f"Top pattern: {patterns[0] if patterns else 'None'}")

def demo_custom_analysis():
    """Demonstrate custom analysis scenarios."""
    print("\n=== Custom Analysis Demo ===")
    
    # Scenario: Web application logs
    web_logs = [
        LogEntry("2023-01-01T10:00:00", "INFO", "GET /api/users 200 45ms", "web-server", "thread-1"),
        LogEntry("2023-01-01T10:01:00", "INFO", "GET /api/users 200 42ms", "web-server", "thread-2"),
        LogEntry("2023-01-01T10:02:00", "ERROR", "POST /api/login 401 12ms", "web-server", "thread-1"),
        LogEntry("2023-01-01T10:03:00", "ERROR", "POST /api/login 401 15ms", "web-server", "thread-2"),
        LogEntry("2023-01-01T10:04:00", "WARN", "GET /api/orders 429 5ms", "web-server", "thread-1"),
        LogEntry("2023-01-01T10:05:00", "WARN", "GET /api/orders 429 7ms", "web-server", "thread-2"),
        LogEntry("2023-01-01T10:06:00", "ERROR", "GET /api/products 500 23ms", "web-server", "thread-1"),
        LogEntry("2023-01-01T10:07:00", "ERROR", "Database query failed for products", "web-server", "thread-1"),
        LogEntry("2023-01-01T10:08:00", "INFO", "User session expired for user-abc", "web-server", "thread-1"),
        LogEntry("2023-01-01T10:09:00", "INFO", "User session expired for user-def", "web-server", "thread-2"),
    ]
    
    analyzer = LogAnalyzer(similarity_threshold=80.0)
    
    print("Web Application Log Analysis:")
    
    # Group similar HTTP requests
    groups = analyzer.group_similar_messages(web_logs)
    print(f"Found {len(groups)} groups of similar requests:")
    
    for group_key, group_logs in groups.items():
        print(f"\n  Group: {group_key}")
        status_codes = Counter()
        response_times = []
        
        for log in group_logs:
            # Extract status code and response time
            match = re.search(r'(\d{3})\s+(\d+)ms', log.message)
            if match:
                status_code, response_time = match.groups()
                status_codes[status_code] += 1
                response_times.append(int(response_time))
        
        print(f"    Count: {len(group_logs)}")
        print(f"    Status codes: {dict(status_codes)}")
        if response_times:
            avg_response = sum(response_times) / len(response_times)
            print(f"    Average response time: {avg_response:.1f}ms")

def main():
    """Run log analysis demos."""
    print("Log Analysis with FastFuzzy")
    print("=" * 50)
    
    try:
        demo_log_analysis()
        demo_performance()
        demo_custom_analysis()
        
        print("\n" + "=" * 50)
        print("Log analysis demo completed!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()