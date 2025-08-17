#!/usr/bin/env python3
"""
Data Deduplication Tool

This script demonstrates how to use fastfuzzy for data deduplication tasks,
such as identifying and removing duplicate records in a dataset.
"""

import fastfuzzy
import csv
import json
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import time

@dataclass
class Record:
    """Represents a data record."""
    id: str
    name: str
    email: str
    address: str
    
    def __str__(self):
        return f"Record(id={self.id}, name='{self.name}', email='{self.email}', address='{self.address}')"

class DataDeduplicator:
    """Deduplicates data records using fuzzy matching."""
    
    def __init__(self, threshold: float = 85.0):
        self.threshold = threshold
        self.fuzzy = fastfuzzy.FuzzyRatio()
        self.process = fastfuzzy.process
        self.utils = fastfuzzy.Utils()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for comparison."""
        return self.utils.default_process(text)
    
    def calculate_similarity(self, record1: Record, record2: Record) -> float:
        """Calculate similarity between two records."""
        # Combine all fields for comparison
        text1 = f"{record1.name} {record1.email} {record1.address}"
        text2 = f"{record2.name} {record2.email} {record2.address}"
        
        # Use weighted approach
        name_sim = self.fuzzy.token_sort_ratio(record1.name, record2.name) or 0.0
        email_sim = self.fuzzy.ratio(record1.email, record2.email) or 0.0
        address_sim = self.fuzzy.token_set_ratio(record1.address, record2.address) or 0.0
        
        # Weighted combination (name is most important)
        total_sim = (name_sim * 0.5 + email_sim * 0.3 + address_sim * 0.2)
        return total_sim
    
    def find_duplicates(self, records: List[Record]) -> List[Tuple[Record, Record, float]]:
        """Find duplicate records in a list."""
        duplicates = []
        processed_pairs: Set[Tuple[str, str]] = set()
        
        print(f"Finding duplicates among {len(records)} records...")
        
        for i, record1 in enumerate(records):
            for j, record2 in enumerate(records[i+1:], i+1):
                # Skip if already processed
                pair_key = tuple(sorted([record1.id, record2.id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                # Calculate similarity
                similarity = self.calculate_similarity(record1, record2)
                
                if similarity >= self.threshold:
                    duplicates.append((record1, record2, similarity))
        
        # Sort by similarity (highest first)
        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates
    
    def cluster_records(self, records: List[Record]) -> List[List[Record]]:
        """Group similar records into clusters."""
        clusters: List[List[Record]] = []
        used_indices: Set[int] = set()
        
        # Sort records by name for better clustering
        sorted_records = sorted(enumerate(records), key=lambda x: x[1].name)
        
        for i, record in sorted_records:
            if i in used_indices:
                continue
            
            # Start a new cluster
            cluster = [record]
            used_indices.add(i)
            
            # Find similar records
            for j, other_record in sorted_records[i+1:]:
                if j in used_indices:
                    continue
                
                similarity = self.calculate_similarity(record, other_record)
                if similarity >= self.threshold:
                    cluster.append(other_record)
                    used_indices.add(j)
            
            if len(cluster) > 1:  # Only keep clusters with duplicates
                clusters.append(cluster)
        
        return clusters
    
    def merge_cluster(self, cluster: List[Record]) -> Record:
        """Merge a cluster of similar records into one."""
        if not cluster:
            raise ValueError("Cannot merge empty cluster")
        
        if len(cluster) == 1:
            return cluster[0]
        
        # Use the record with the most complete information as base
        best_record = max(cluster, key=lambda r: (
            len(r.name),
            len(r.email),
            len(r.address)
        ))
        
        # Create merged record with best available data
        merged_id = f"merged_{'_'.join(r.id for r in cluster)}"
        
        return Record(
            id=merged_id,
            name=best_record.name,
            email=best_record.email,
            address=best_record.address
        )

def generate_sample_data() -> List[Record]:
    """Generate sample data with duplicates for testing."""
    sample_records = [
        Record("1", "John Smith", "john.smith@email.com", "123 Main St, Anytown"),
        Record("2", "Jon Smith", "john.smith@email.com", "123 Main Street, Anytown"),
        Record("3", "Jane Doe", "jane.doe@email.com", "456 Oak Ave, Somewhere"),
        Record("4", "Jane Doe", "jane.d@email.com", "456 Oak Avenue, Somewhere"),
        Record("5", "Bob Johnson", "bob.johnson@email.com", "789 Pine Rd, Elsewhere"),
        Record("6", "Robert Johnson", "b.johnson@email.com", "789 Pine Road, Elsewhere"),
        Record("7", "Alice Brown", "alice.brown@email.com", "321 Elm St, Nowhere"),
        Record("8", "Alice Brown", "a.brown@email.com", "321 Elm Street, Nowhere"),
        Record("9", "Charlie Wilson", "charlie.wilson@email.com", "654 Maple Dr, Anywhere"),
        Record("10", "Chuck Wilson", "chuck.wilson@email.com", "654 Maple Drive, Anywhere"),
        # Exact duplicates
        Record("11", "John Smith", "john.smith@email.com", "123 Main St, Anytown"),
        Record("12", "Jane Doe", "jane.doe@email.com", "456 Oak Ave, Somewhere"),
    ]
    
    return sample_records

def demo_deduplication():
    """Demonstrate data deduplication."""
    print("=== Data Deduplication Demo ===")
    
    # Generate sample data
    records = generate_sample_data()
    print(f"Generated {len(records)} sample records")
    
    # Initialize deduplicator
    deduplicator = DataDeduplicator(threshold=80.0)
    
    # Find duplicates
    start_time = time.perf_counter()
    duplicates = deduplicator.find_duplicates(records)
    find_time = time.perf_counter() - start_time
    
    print(f"\nFound {len(duplicates)} duplicate pairs in {find_time:.4f} seconds:")
    for record1, record2, similarity in duplicates[:10]:  # Show top 10
        print(f"  {similarity:5.1f}% - {record1.name} <-> {record2.name}")
    
    # Cluster records
    start_time = time.perf_counter()
    clusters = deduplicator.cluster_records(records)
    cluster_time = time.perf_counter() - start_time
    
    print(f"\nFound {len(clusters)} clusters in {cluster_time:.4f} seconds:")
    for i, cluster in enumerate(clusters, 1):
        print(f"  Cluster {i}:")
        for record in cluster:
            print(f"    - {record}")
    
    # Merge clusters
    merged_records = []
    for cluster in clusters:
        merged = deduplicator.merge_cluster(cluster)
        merged_records.append(merged)
        print(f"\nMerged cluster into: {merged}")

def demo_fuzzy_search():
    """Demonstrate fuzzy search in records."""
    print("\n=== Fuzzy Search Demo ===")
    
    # Sample records
    records = [
        Record("1", "John Smith", "john.smith@email.com", "123 Main St"),
        Record("2", "Jane Doe", "jane.doe@email.com", "456 Oak Ave"),
        Record("3", "Bob Johnson", "bob.johnson@email.com", "789 Pine Rd"),
        Record("4", "Alice Brown", "alice.brown@email.com", "321 Elm St"),
        Record("5", "Charlie Wilson", "charlie.wilson@email.com", "654 Maple Dr"),
    ]
    
    # Create search choices (name + email)
    choices = [f"{r.name} ({r.email})" for r in records]
    record_map = {choice: record for choice, record in zip(choices, records)}
    
    # Search queries
    queries = [
        "John Smith",
        "Jane Doew",
        "Bob Jonson",
        "Alic Brown",
    ]
    
    process = fastfuzzy.process
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = process.extract(query, choices, limit=3, scorer="token_sort_ratio")
        
        for choice, score, index in results:
            record = record_map[choice]
            print(f"  {score:5.1f}% - {record.name} <{record.email}>")

def main():
    """Run deduplication demos."""
    print("Data Deduplication Tool with FastFuzzy")
    print("=" * 50)
    
    try:
        demo_deduplication()
        demo_fuzzy_search()
        
        print("\n" + "=" * 50)
        print("Deduplication demo completed!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()