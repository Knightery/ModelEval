import html
import io
import json
import os
import pickle
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

from bs4 import BeautifulSoup
import fasttext
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# Configuration
DISTILBERT_MODEL_PATH = "./distilbert_skills_model"
FASTTEXT_MODEL_PATH = "./fasttextclassifier.bin"
JSON_RESULTS_PATH = "evaluation_results/skill_evaluation_results.json"
RETRIEVED_JOBS_PATH = os.getenv('RETRIEVED_JOBS_PATH', 'retrieved_jobs.json')
SKILL_VECTORS_PATH = "skill_vectorization_results/skill_vectors.pkl"
SKILL_STATISTICS_PATH = "skill_vectorization_results/skill_statistics.json"
SIMILARITY_THRESHOLD = 0.65
RELEVANCE_THRESHOLD = 0.3

class SkillEvaluator:
    def __init__(self, json_path: str = None, retrieved_jobs_path: str = None, expected_skills: List[str] = None, query: str = None):
        self.query = query or "Python Developer in San Francisco with Salary 100000"
        print("Skill Extraction Evaluator")
        print(f"   Job Query: {self.query}")
        print(f"   Using similarity threshold: {SIMILARITY_THRESHOLD}")
        print(f"   DistilBERT Model: {DISTILBERT_MODEL_PATH}")
        
        self.json_path = json_path or JSON_RESULTS_PATH
        self.retrieved_jobs_path = retrieved_jobs_path or 'retrieved_jobs.json'
        self.expected_skills = expected_skills or ['python']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {self.device}")
        print(f"   Expected skills: {self.expected_skills}")
        
        # Initialize models
        self.distilbert_model = None
        self.tokenizer = None
        self.fasttext_model = None
        self.sentence_transformer = None
        
        # Load models and vectorized skills
        self.load_models()
        self.skill_vectors = {}
        self.skill_statistics = {}
        self.load_vectorized_skills()
        
        # Statistics tracking
        self.stats = {
            "total_jobs": 0,
            "jobs_with_skills": 0,
            "total_expected_skills": 0,
            "total_extracted_skills": 0,
            "matched_skills": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "skill_matches": [],
            "job_results": [],
            "query_analysis": {}
        }

    def load_jobs_from_retrieved(self) -> List[Dict]:
        """Load jobs from retrieved_jobs.json"""
        if not os.path.exists(self.retrieved_jobs_path):
            print(f"Retrieved jobs file not found: {self.retrieved_jobs_path}")
            return []
        
        with open(self.retrieved_jobs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if 'jobs' in data:
                return data['jobs'] or []
            if 'results' in data:
                return data['results'] or []
            print("Unrecognized structure in retrieved_jobs.json")
            return []
        
        if isinstance(data, list):
            return data
        
        print("Unrecognized JSON root type in retrieved_jobs.json")
        return []
    
    def load_models(self):
        """Load all required models"""
        print("Loading models...")
        
        # Load DistilBERT
        print("   Loading DistilBERT skill extraction model...")
        self.distilbert_model = AutoModelForTokenClassification.from_pretrained(DISTILBERT_MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
        self.distilbert_model.to(self.device)
        self.distilbert_model.eval()
        print("     DistilBERT model loaded successfully")
        
        # Load Sentence Transformer
        print("   Loading Sentence Transformer for skill similarity...")
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        print("     Sentence Transformer loaded successfully")
        
        # Load FastText
        print(f"   Loading FastText model from {FASTTEXT_MODEL_PATH}")
        if os.path.exists(FASTTEXT_MODEL_PATH):
            self.fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            print("     FastText model loaded successfully")
        else:
            print(f"FastText model not found at {FASTTEXT_MODEL_PATH}")
            print("Continuing without FastText (will use full text for skill extraction)")
            self.fasttext_model = None
        
        print("   Models loaded and ready for evaluation")

    def load_vectorized_skills(self):
        """Load pre-computed skill vectors and statistics"""
        print("Loading vectorized skills for similarity comparison...")
        
        # Load skill vectors
        if os.path.exists(SKILL_VECTORS_PATH):
            with open(SKILL_VECTORS_PATH, 'rb') as f:
                self.skill_vectors = pickle.load(f)
            print(f"   Loaded {len(self.skill_vectors)} skill vectors")
        else:
            print(f"   Skill vectors not found at {SKILL_VECTORS_PATH}")
        
        # Load skill statistics
        if os.path.exists(SKILL_STATISTICS_PATH):
            with open(SKILL_STATISTICS_PATH, 'r') as f:
                self.skill_statistics = json.load(f)
            print(f"   Loaded statistics for {self.skill_statistics.get('unique_skills', 0)} unique skills")
        else:
            print(f"   Skill statistics not found at {SKILL_STATISTICS_PATH}")

    def extract_skills_from_query(self, query: str) -> List[str]:
        """Extract expected skills from the job query using provided expected skills"""
        print(f"Using provided expected skills for query: '{query}'")
        
        # Normalize the expected skills
        normalized_skills = [self.normalize_skill(skill) for skill in self.expected_skills]
        
        print(f"   Expected skills: {normalized_skills}")
        print(f"   Total expected skills: {len(normalized_skills)}")
        return normalized_skills

    def normalize_skill(self, skill: str) -> str:
        """Normalize skill text for consistency"""
        return skill.lower().strip()

    def clean_html_description(self, html_text: str) -> str:
        """Clean HTML tags and decode entities from job descriptions"""
        if not html_text:
            return ""
        
        soup = BeautifulSoup(html_text, "html.parser")
        clean_text = soup.get_text(separator=" ", strip=True)
        clean_text = html.unescape(clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        return clean_text.strip()

    def find_safe_split_point(self, text: str, target_pos: int, window_size: int) -> int:
        """Find a safe split point that doesn't break words or sentences"""
        if target_pos >= len(text):
            return len(text)
        
        # Look for sentence boundaries first (period, exclamation, question mark)
        search_start = max(0, target_pos - 50)  # Look back up to 50 chars
        search_end = min(len(text), target_pos + 50)  # Look forward up to 50 chars
        
        # Find sentence endings in the search area
        for i in range(target_pos, search_end):
            if i < len(text) and text[i] in '.!?':
                # Check if there's whitespace after the punctuation
                if i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
        
        # If no sentence boundary found, look for word boundaries
        for i in range(target_pos, search_end):
            if i < len(text) and text[i].isspace():
                return i
        
        # If no word boundary found, look backwards
        for i in range(target_pos, search_start, -1):
            if i > 0 and text[i].isspace():
                return i
        
        # Last resort: use target position
        return target_pos
    
    def extract_skills_from_text(self, text: str, confidence_threshold: float = 0.35) -> List[str]:
        """Extract skills from text using DistilBERT model with sliding window for long texts"""
        if not self.distilbert_model or not self.tokenizer:
            print("   DistilBERT model not available")
            return []
        
        print(f"   DistilBERT processing {len(text)} characters")
        print(f"   Text: {text[:150]}..." if len(text) > 150 else f"   Text: {text}")
        
        # Check if text is too long for single processing
        max_chars = 1000  # Conservative limit for single processing
        if len(text) > max_chars:
            return self.extract_skills_with_sliding_window(text, confidence_threshold)
        
        return self.extract_skills_single_chunk(text, confidence_threshold)
    
    def extract_skills_with_sliding_window(self, text: str, confidence_threshold: float = 0.35) -> List[str]:
        """Extract skills using sliding window approach for long texts"""
        window_size = 800  # Characters per window
        stride = 400  # Overlap between windows
        all_skills = []
        
        print(f"   Using sliding window: {len(text)} chars, window_size={window_size}, stride={stride}")
        
        # Process text in overlapping windows
        start_pos = 0
        while start_pos < len(text):
            # Calculate end position with safe boundary
            end_pos = min(start_pos + window_size, len(text))
            if end_pos < len(text):
                end_pos = self.find_safe_split_point(text, end_pos, window_size)
            
            window_text = text[start_pos:end_pos]
            
            if window_text.strip():  # Skip empty windows
                print(f"   Processing window {start_pos}-{end_pos} ({len(window_text)} chars)")
                window_skills = self.extract_skills_single_chunk(window_text, confidence_threshold)
                all_skills.extend(window_skills)
            
            # Move to next window with safe stride
            next_start = start_pos + stride
            if next_start < len(text):
                next_start = self.find_safe_split_point(text, next_start, stride)
            start_pos = next_start
            
            # Prevent infinite loop
            if start_pos >= end_pos:
                start_pos = end_pos
        
        # Remove duplicates while preserving order
        unique_skills = []
        seen = set()
        for skill in all_skills:
            if skill not in seen:
                unique_skills.append(skill)
                seen.add(skill)
        
        print(f"   Sliding window extracted {len(unique_skills)} unique skills: {unique_skills}")
        return unique_skills
    
    def extract_skills_single_chunk(self, text: str, confidence_threshold: float = 0.35) -> List[str]:
        """Extract skills from a single chunk of text"""
        # Tokenize
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        offset_mapping = encoded['offset_mapping'][0].numpy()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.distilbert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
        
        # Get predictions and probabilities
        probs = torch.nn.functional.softmax(logits, dim=2)
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
        
        # Label mapping
        label_map = {0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'}
        
        # Extract skills
        skills = []
        current_skill_tokens = []
        current_confidences = []
        
        for i in range(1, len(predictions)):  # Skip [CLS] token
            if attention_mask[0][i] == 0:  # Skip padding
                break
                
            pred_label = label_map.get(predictions[i], 'O')
            confidence = probs[0][i][predictions[i]].item()
            
            # Track token positions for original text extraction
            token_start = None
            token_end = None
            if i < len(offset_mapping):
                token_start, token_end = offset_mapping[i]
            
            if pred_label == 'B-SKILL':
                # Save previous skill if exists
                if current_skill_tokens and current_confidences:
                    skill_text = self.extract_skill_from_original_text(current_skill_tokens, text)
                    avg_confidence = sum(current_confidences) / len(current_confidences)
                    if avg_confidence >= confidence_threshold and skill_text and len(skill_text.strip()) >= 3:
                        skills.append(skill_text.lower())
                
                # Start new skill
                current_skill_tokens = [(token_start, token_end)] if token_start is not None else []
                current_confidences = [confidence]
                
            elif pred_label == 'I-SKILL' and current_skill_tokens:
                # Continue current skill
                if token_start is not None:
                    current_skill_tokens.append((token_start, token_end))
                    current_confidences.append(confidence)
                    
            else:
                # End current skill
                if current_skill_tokens and current_confidences:
                    skill_text = self.extract_skill_from_original_text(current_skill_tokens, text)
                    avg_confidence = sum(current_confidences) / len(current_confidences)
                    if avg_confidence >= confidence_threshold and skill_text and len(skill_text.strip()) >= 3:
                        skills.append(skill_text.lower())
                current_skill_tokens = []
                current_confidences = []
        
        # Don't forget the last skill
        if current_skill_tokens and current_confidences:
            skill_text = self.extract_skill_from_original_text(current_skill_tokens, text)
            avg_confidence = sum(current_confidences) / len(current_confidences)
            if avg_confidence >= confidence_threshold and skill_text and len(skill_text.strip()) >= 3:
                skills.append(skill_text.lower())
        
        return skills
    
    def extract_relevant_sentences_with_fasttext(self, text: str, confidence_threshold: float = 0.5) -> str:
        """Use FastText to identify and extract relevant sentences that likely contain skills"""
        if not self.fasttext_model:
            print("   FastText not available, using full text")
            return text
                
        # Split text into sentences
        sentences = re.split(r'[.!?]\s+', text)
        relevant_sentences = []
                
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            # Use FastText to classify if sentence is relevant for skills
            predictions = self.fasttext_model.predict(sentence, k=1)
            labels, confidences = predictions
            
            # Check if the predicted label is 'relevant' and meets confidence threshold
            if (len(labels) > 0 and len(confidences) > 0 and 
                labels[0] == '__label__relevant' and confidences[0] >= confidence_threshold):
                relevant_sentences.append(sentence)
            else:
                label = labels[0] if len(labels) > 0 else 'unknown'
                conf = confidences[0] if len(confidences) > 0 else 0.0
        
        # Join relevant sentences back together
        relevant_text = '. '.join(relevant_sentences)
        
        print(f"   FastText selected {len(relevant_sentences)}/{len(sentences)} sentences ({len(relevant_text)} chars)")
        
        # If no relevant sentences found, return original text as fallback
        return relevant_text if relevant_text.strip() else text

    def extract_skills_from_text_enhanced(self, text: str, job_title: str = "", confidence_threshold: float = 0.35) -> List[str]:
        """Enhanced skill extraction: FastText for relevant sentences + DistilBERT for skills + job title inclusion"""
        print(f"Starting enhanced extraction")
        print(f"Job Title: '{job_title}'")
        print(f"Description Length: {len(text)} chars")
        print(f"Description Preview: {text[:200]}..." if len(text) > 200 else f"Description: {text}")
        
        # Combine job title and text
        full_text = f"{job_title}. {text}" if job_title else text
        print(f"Full text length: {len(full_text)} chars")
        
        # Step 1: Use FastText to extract relevant sentences
        print(f"Running FastText sentence filtering...")
        relevant_text = self.extract_relevant_sentences_with_fasttext(full_text, confidence_threshold)
        print(f"Filtered text length: {len(relevant_text)} chars")
        print(f"Filtered text: {relevant_text[:300]}..." if len(relevant_text) > 300 else f"Filtered text: {relevant_text}")
        
        # Step 2: Use DistilBERT on the relevant text
        print(f"Running DistilBERT skill extraction...")
        skills = self.extract_skills_from_text(relevant_text, confidence_threshold)
        
        print(f"Extracted {len(skills)} skills: {skills}")
        return skills

    def extract_skill_from_original_text(self, token_offsets: List[Tuple[int, int]], original_text: str) -> str:
        """Extract skill text from original source using token offsets"""
        if not token_offsets or not original_text:
            return ""
        
        # Find the start and end positions of the entire skill
        start_pos = token_offsets[0][0]
        end_pos = token_offsets[-1][1]
        
        # Extract the exact text from original source
        if start_pos < len(original_text) and end_pos <= len(original_text):
            skill_text = original_text[start_pos:end_pos]
        else:
            skill_parts = []
            for start, end in token_offsets:
                if start < len(original_text) and end <= len(original_text):
                    skill_parts.append(original_text[start:end])
            skill_text = ' '.join(skill_parts)
        
        return skill_text.strip()
    
    def calculate_skill_relevance(self, extracted_skills: List[str], query_skills: List[str]) -> Dict[str, float]:
        """Calculate relevance score for each extracted skill based on vector similarity to query skills"""
        if not self.skill_vectors or not query_skills:
            return {skill: 0.0 for skill in extracted_skills}
        
        relevance_scores = {}
        
        print(f"   Calculating vector-based relevance for {len(extracted_skills)} extracted skills")
        print(f"   Query skills: {query_skills}")
        
        for extracted_skill in extracted_skills:
            # Skip skills with less than 3 characters
            if len(extracted_skill.strip()) < 3:
                print(f"   Skipping short skill: '{extracted_skill}' (less than 3 characters)")
                relevance_scores[extracted_skill] = 0.0
                continue
                
            max_similarity = 0.0
            best_match = None
            
            # Get embedding for extracted skill
            extracted_skill_normalized = self.normalize_skill(extracted_skill)
            extracted_embedding = self._get_skill_embedding(extracted_skill_normalized)
            
            if extracted_embedding is None:
                relevance_scores[extracted_skill] = 0.0
                continue
            
            # Compare with each query skill using pure vector similarity
            for query_skill in query_skills:
                query_skill_normalized = self.normalize_skill(query_skill)
                query_embedding = self._get_skill_embedding(query_skill_normalized)
                
                if query_embedding is None:
                    continue
                
                # Calculate cosine similarity between vectors
                raw_similarity = cosine_similarity([extracted_embedding], [query_embedding])[0][0]
                
                # Apply exponential transformation to make differences more pronounced
                # This will make 0.5 -> ~0.25, 0.6 -> ~0.36, 0.8 -> ~0.64, etc.
                similarity = raw_similarity ** 2
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = query_skill
            
            # Use pure vector similarity as relevance score with minimal threshold
            relevance_score = max(0.0, max_similarity)
            
            # Apply minimum relevance threshold to filter out completely unrelated skills
            if relevance_score < RELEVANCE_THRESHOLD:
                relevance_score = 0.0
            
            relevance_scores[extracted_skill] = relevance_score
            
            # Calculate raw similarity for debug output if we have a best match
            if best_match:
                best_match_embedding = self._get_skill_embedding(self.normalize_skill(best_match))
                if best_match_embedding is not None:
                    raw_best_similarity = cosine_similarity([extracted_embedding], [best_match_embedding])[0][0]
                    print(f"   '{extracted_skill}' -> '{best_match}' (raw: {raw_best_similarity:.3f}, transformed: {max_similarity:.3f}, relevance: {relevance_score:.3f})")
                else:
                    print(f"   '{extracted_skill}' -> '{best_match}' (transformed: {max_similarity:.3f}, relevance: {relevance_score:.3f})")
            else:
                print(f"   '{extracted_skill}' -> No match (relevance: {relevance_score:.3f})")
        
        # Log similarity distribution for analysis
        all_similarities = [score for score in relevance_scores.values() if score > 0]
        if all_similarities:
            avg_sim = sum(all_similarities) / len(all_similarities)
            max_sim = max(all_similarities)
            min_sim = min(all_similarities)
            print(f"   Similarity stats - Avg: {avg_sim:.3f}, Max: {max_sim:.3f}, Min: {min_sim:.3f}, Above threshold: {len([s for s in all_similarities if s >= RELEVANCE_THRESHOLD])}/{len(all_similarities)}")
        
        return relevance_scores

    def _get_skill_embedding(self, skill_normalized: str) -> Optional[np.ndarray]:
        """Get embedding for a skill, either from pre-computed vectors or by generating new one"""
        # Filter out skills with less than 3 characters
        if len(skill_normalized) < 3:
            print(f"   Skipping skill '{skill_normalized}' (less than 3 characters)")
            return None
            
        if skill_normalized in self.skill_vectors:
            return self.skill_vectors[skill_normalized]
        
        # Generate embedding for skills not in pre-computed vectors
        context = f"Technical skill: {skill_normalized}. Professional competency in {skill_normalized}."
        embedding = self.sentence_transformer.encode([context])[0]
        # Cache the new embedding for future use
        self.skill_vectors[skill_normalized] = embedding
        print(f"   Generated new embedding for: {skill_normalized}")
        return embedding
    
    def evaluate_query_skills(self) -> Dict:
        """Evaluate search results in retrieved_jobs.json by percent match to expected skills."""
        print(f"Evaluating skill extraction for query...")

        # Extract expected skills from query
        expected_skills = self.extract_skills_from_query(self.query)

        # Load retrieved jobs
        jobs = self.load_jobs_from_retrieved()
        if not jobs:
            # Fallback: analyze the query text only
            extracted_from_query = self.extract_skills_from_text_enhanced(self.query)
            normalized_extracted = [self.normalize_skill(skill) for skill in extracted_from_query]
            query_analysis = {
                "job_query": self.query,
                "expected_skills": expected_skills,
                "extracted_from_query": normalized_extracted,
                "self_extraction_match": len(set(expected_skills) & set(normalized_extracted)) / len(expected_skills) if expected_skills else 0
            }
            self.stats["query_analysis"] = query_analysis
            return query_analysis

        print(f"Analyzing skills from {len(jobs)} retrieved jobs...")

        job_results = []
        total_percent = 0.0
        nonempty_expected = len(expected_skills) > 0

        for job in jobs:
            job_id = job.get('job_id') or job.get('id')
            title = job.get('job_title') or job.get('title') or 'N/A'
            company = job.get('company_name') or job.get('company') or 'N/A'
            raw_description = job.get('description') or ''

            # Clean HTML using BeautifulSoup
            clean_desc = self.clean_html_description(raw_description) if isinstance(raw_description, str) else ''

            extracted = self.extract_skills_from_text_enhanced(clean_desc, title)
            normalized_extracted = [self.normalize_skill(skill) for skill in extracted]

            # Calculate relevance scores for extracted skills
            relevance_scores = self.calculate_skill_relevance(normalized_extracted, expected_skills)
            
            # Find highly relevant skills (above threshold)
            relevant_skills = [(skill, score) for skill, score in relevance_scores.items() if score >= RELEVANCE_THRESHOLD]
            
            # Calculate relevance-based score
            if relevant_skills:
                # Average relevance score of all relevant skills
                relevance_based_score = sum(score for _, score in relevant_skills) / len(relevant_skills)
                # Weight by coverage (how many relevant skills found vs expected)
                coverage_weight = min(1.0, len(relevant_skills) / len(expected_skills)) if expected_skills else 0.0
                final_score = relevance_based_score * (0.7 + 0.3 * coverage_weight)  # 70% relevance + 30% coverage
            else:
                final_score = 0.0
            
            total_percent += final_score

            job_results.append({
                "job_id": job_id,
                "job_title": title,
                "company_name": company,
                "extracted_skills": normalized_extracted,
                "relevant_skills": [skill for skill, _ in relevant_skills],
                "relevance_scores": relevance_scores,
                "relevance_based_score": final_score,
                "coverage": len(relevant_skills) / len(expected_skills) if expected_skills else 0.0
            })

        average_percent_match = total_percent / len(jobs) if jobs else 0.0

        query_analysis = {
            "job_query": self.query,
            "expected_skills": expected_skills,
            "average_percent_match": average_percent_match,
            "job_results": job_results
        }

        self.stats["query_analysis"] = query_analysis
        return query_analysis
    
    def print_query_analysis(self):
        """Print analysis of the job query skill extraction"""
        if not self.stats.get("query_analysis"):
            print("No query analysis results to display")
            return
        
        analysis = self.stats["query_analysis"]
        
        print("="*60)
        print("JOB QUERY SKILL ANALYSIS")
        print("="*60)
        
        print(f"QUERY: {analysis['job_query']}")
        
        print(f"EXPECTED SKILLS FROM QUERY ({len(analysis['expected_skills'])}):")
        for skill in analysis['expected_skills']:
            print(f"   {skill}")

        if 'extracted_from_query' in analysis and 'job_results' not in analysis:
            print(f"EXTRACTED FROM QUERY ({len(analysis['extracted_from_query'])}):")
            for skill in analysis['extracted_from_query']:
                print(f"   {skill}")
            
            print(f"SELF-EXTRACTION MATCH: {analysis['self_extraction_match']:.4f}")
        
        if 'job_results' in analysis:
            print(f"AVERAGE RELEVANCE-BASED SCORE ACROSS JOBS: {analysis['average_percent_match']*100:.2f}%")
            # Show top 10 by relevance score
            sorted_jobs = sorted(analysis['job_results'], key=lambda j: j.get('relevance_based_score', 0), reverse=True)
            print(f"TOP RELEVANT JOBS:")
            for job in sorted_jobs[:10]:
                score = job.get('relevance_based_score', 0) * 100
                coverage = job.get('coverage', 0) * 100
                relevant_skills = job.get('relevant_skills', [])
                print(f"   {score:6.2f}% (coverage: {coverage:5.1f}%) | {job.get('job_title','N/A')} @ {job.get('company_name','N/A')}")
                if relevant_skills:
                    print(f"     Relevant skills: {', '.join(relevant_skills[:5])}{'...' if len(relevant_skills) > 5 else ''}")
        
        print("="*60)

def main():
    """Main function with command line argument support"""
    import argparse
    parser = argparse.ArgumentParser(description="Test skill accuracy of job search results")
    parser.add_argument('--retrieved-jobs-path', type=str, default='retrieved_jobs.json',
                       help='Path to retrieved jobs JSON file')
    parser.add_argument('--json-results-path', type=str, default='evaluation_results/skill_evaluation_results.json',
                       help='Path to save JSON results')
    parser.add_argument('--expected-skills', nargs='+', type=str, default=['python'],
                       help='List of expected skills to evaluate (default: python)')
    parser.add_argument('--query', type=str, default='Python Developer in San Francisco with Salary 100000',
                       help='Job query being evaluated (default: Python Developer in San Francisco with Salary 100000)')
    
    args = parser.parse_args()
    
    # Use the query from command line arguments
    query = args.query
    
    # Initialize evaluator with custom paths and expected skills
    evaluator = SkillEvaluator(json_path=args.json_results_path, 
                              retrieved_jobs_path=args.retrieved_jobs_path,
                              expected_skills=args.expected_skills,
                              query=query)
    
    # Load jobs from retrieved results
    print("Loading retrieved job results...")
    jobs = evaluator.load_jobs_from_retrieved()
    
    if not jobs:
        print("No jobs found in retrieved results")
        return 0.0
    
    print(f"   Found {len(jobs)} jobs")
    
    # Process all jobs for skill extraction
    print("Processing jobs for skill extraction...")
    all_results = []
    
    for i, job in enumerate(tqdm(jobs, desc="Processing jobs")):
        # Extract job information
        job_title = job.get('job_title', '')
        company = job.get('company_name', '')
        raw_description = job.get('description', '')
        
        print(f"="*80)
        print(f"PROCESSING JOB {i+1}/{len(jobs)}")
        print(f"Title: {job_title}")
        print(f"Company: {company}")
        print(f"="*80)
        
        if not raw_description:
            print(f"Skipping job {i+1}: No description available")
            continue
        
        # Clean HTML from description using BeautifulSoup
        clean_description = evaluator.clean_html_description(raw_description)
        print(f"Cleaned description length: {len(clean_description)} chars (was {len(raw_description)} chars)")
        
        # Extract skills from cleaned job description using enhanced method with title
        extracted_skills = evaluator.extract_skills_from_text_enhanced(clean_description, job_title)
        
        # Calculate detailed evaluation scores for this job
        expected_skills = evaluator.extract_skills_from_query(query)
        normalized_extracted = [evaluator.normalize_skill(skill) for skill in extracted_skills]
        
        # Calculate relevance scores for extracted skills
        relevance_scores = evaluator.calculate_skill_relevance(normalized_extracted, expected_skills)
        
        # Find highly relevant skills (above threshold)
        relevant_skills = [(skill, score) for skill, score in relevance_scores.items() if score >= RELEVANCE_THRESHOLD]
        
        # Calculate job-level scores
        if relevant_skills:
            # Average relevance score of all relevant skills
            relevance_based_score = sum(score for _, score in relevant_skills) / len(relevant_skills)
            # Weight by coverage (how many relevant skills found vs expected)
            coverage_weight = min(1.0, len(relevant_skills) / len(expected_skills)) if expected_skills else 0.0
            final_score = relevance_based_score * (0.7 + 0.3 * coverage_weight)  # 70% relevance + 30% coverage
        else:
            relevance_based_score = 0.0
            coverage_weight = 0.0
            final_score = 0.0
        
        # Create detailed result entry
        result = {
            'job_id': i,
            'job_title': job_title,
            'company_name': company,
            'raw_description_length': len(raw_description),
            'cleaned_description_length': len(clean_description),
            'extracted_skills': extracted_skills,
            'normalized_extracted_skills': normalized_extracted,
            'num_skills': len(extracted_skills),
            'evaluation_scores': {
                'expected_skills': expected_skills,
                'relevance_scores': {skill: float(score) for skill, score in relevance_scores.items()},
                'relevant_skills_count': len(relevant_skills),
                'relevant_skills': [skill for skill, _ in relevant_skills],
                'relevance_based_score': float(relevance_based_score),
                'coverage_ratio': float(coverage_weight),
                'final_score': float(final_score),
                'above_threshold_count': len([s for s in relevance_scores.values() if s >= RELEVANCE_THRESHOLD])
            },
            'skill_matches': [
                {
                    'skill': skill,
                    'relevance_score': float(score),
                    'is_relevant': bool(score >= RELEVANCE_THRESHOLD)
                }
                for skill, score in relevance_scores.items()
            ]
        }
        
        all_results.append(result)
    
    # Calculate comprehensive summary statistics
    if all_results:
        all_final_scores = [r['evaluation_scores']['final_score'] for r in all_results]
        all_relevance_scores = [r['evaluation_scores']['relevance_based_score'] for r in all_results]
        all_coverage_ratios = [r['evaluation_scores']['coverage_ratio'] for r in all_results]
        jobs_with_relevant_skills = len([r for r in all_results if r['evaluation_scores']['relevant_skills_count'] > 0])
        
        # Sort jobs by final score for ranking
        sorted_results = sorted(all_results, key=lambda x: x['evaluation_scores']['final_score'], reverse=True)
        top_5_jobs = sorted_results[:5]
        
        summary = {
            'total_skills_found': sum(r['num_skills'] for r in all_results),
            'avg_skills_per_job': sum(r['num_skills'] for r in all_results) / len(all_results),
            'jobs_with_relevant_skills': jobs_with_relevant_skills,
            'jobs_with_relevant_skills_percentage': (jobs_with_relevant_skills / len(all_results)) * 100,
            'overall_metrics': {
                'average_final_score': float(sum(all_final_scores) / len(all_final_scores)),
                'average_relevance_score': float(sum(all_relevance_scores) / len(all_relevance_scores)),
                'average_coverage_ratio': float(sum(all_coverage_ratios) / len(all_coverage_ratios)),
                'max_final_score': float(max(all_final_scores)),
                'min_final_score': float(min(all_final_scores)),
                'jobs_above_30_percent': len([s for s in all_final_scores if s >= 0.3]),
                'jobs_above_50_percent': len([s for s in all_final_scores if s >= 0.5]),
                'jobs_above_70_percent': len([s for s in all_final_scores if s >= 0.7])
            },
            'top_5_jobs': [
                {
                    'rank': i + 1,
                    'job_id': job['job_id'],
                    'job_title': job['job_title'],
                    'company_name': job['company_name'],
                    'final_score': float(job['evaluation_scores']['final_score']),
                    'relevance_score': float(job['evaluation_scores']['relevance_based_score']),
                    'coverage_ratio': float(job['evaluation_scores']['coverage_ratio']),
                    'relevant_skills_count': job['evaluation_scores']['relevant_skills_count'],
                    'relevant_skills': job['evaluation_scores']['relevant_skills']
                }
                for i, job in enumerate(top_5_jobs)
            ]
        }
    else:
        summary = {
            'total_skills_found': 0,
            'avg_skills_per_job': 0,
            'jobs_with_relevant_skills': 0,
            'jobs_with_relevant_skills_percentage': 0,
            'overall_metrics': {},
            'top_5_jobs': []
        }
    
    # Save results with enhanced structure
    output_data = {
        'metadata': {
            'query': query,
            'expected_skills': expected_skills,
            'relevance_threshold': RELEVANCE_THRESHOLD,
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'timestamp': datetime.now().isoformat(),
            'total_jobs_processed': len(jobs),
            'jobs_with_skills': len(all_results)
        },
        'summary': summary,
        'job_results': all_results
    }
    
    with open(evaluator.json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {evaluator.json_path}")
    print(f"   Jobs processed: {len(all_results)}")
    print(f"   Total skills found: {output_data['summary']['total_skills_found']}")
    print(f"   Average skills per job: {output_data['summary']['avg_skills_per_job']:.2f}")
    
    # Use relevance-based evaluation instead of just counting skills
    print("Calculating relevance-based accuracy...")
    
    # Extract expected skills from query for relevance comparison
    expected_skills = evaluator.extract_skills_from_query(query)
    
    total_relevance_score = 0.0
    relevant_jobs = 0
    
    for result in all_results:
        extracted_skills = result['extracted_skills']
        if extracted_skills:
            # Calculate relevance scores
            relevance_scores = evaluator.calculate_skill_relevance(extracted_skills, expected_skills)
            relevant_skills = [(skill, score) for skill, score in relevance_scores.items() if score >= RELEVANCE_THRESHOLD]
            
            if relevant_skills:
                relevance_based_score = sum(score for _, score in relevant_skills) / len(relevant_skills)
                coverage_weight = min(1.0, len(relevant_skills) / len(expected_skills)) if expected_skills else 0.0
                final_score = relevance_based_score * (0.7 + 0.3 * coverage_weight)
                total_relevance_score += final_score
                relevant_jobs += 1
    
    # Calculate final relevance-based accuracy
    final_accuracy = total_relevance_score / len(all_results) if all_results else 0.0
    
    print(f"Relevance-based accuracy: {final_accuracy:.4f}")
    print(f"Jobs with relevant skills: {relevant_jobs}/{len(all_results)}")
    print(f"{final_accuracy:.4f}")
    
    return final_accuracy

if __name__ == "__main__":
    main()
