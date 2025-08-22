import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
import torch
import fasttext
import asyncio
import aiofiles
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Set
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')

# Configuration
DISTILBERT_MODEL_PATH = "DistilbertCopy/distilbert_skills_model"
FASTTEXT_MODEL_PATH = "DistilbertCopy/fasttextclassifier.bin"
JOBS_DB_PATH = "DistilbertCopy/jobs.db"  # SQLite database path

class SkillVectorizationPipeline:
    def __init__(self, batch_size: int = None):
        """Initialize the skill vectorization pipeline focused on skills only"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print("🎯 SKILLS-ONLY VECTORIZATION PIPELINE (OPTIMIZED)")
        print("   Ignoring: job titles, seniority, responsibilities")
        print("   Focusing: pure skill extraction and alignment")
        
        # Hardware optimization
        self.gpu_memory = self._get_gpu_memory()
        self.batch_size = batch_size or self._calculate_optimal_batch_size()
        self.max_workers = min(8, os.cpu_count() or 4)  # Conservative for I/O
        
        print(f"🚀 Optimizations:")
        print(f"   GPU Memory: {self.gpu_memory:.1f}GB")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Max Workers: {self.max_workers}")
        
        # Models
        self.distilbert_model = None
        self.tokenizer = None
        self.fasttext_model = None
        self.sentence_transformer = None
        
        # Data storage
        self.job_descriptions = []
        self.extracted_skills_data = []
        self.skill_vectors = {}
        self.skill_statistics = {}
        
        # Performance tracking
        self.start_time = None
        self.processed_jobs = 0
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load all required models"""
        print("Loading models...")
        
        # Load DistilBERT model
        try:
            self.distilbert_model = AutoModelForTokenClassification.from_pretrained(DISTILBERT_MODEL_PATH)
            self.tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
            self.distilbert_model.to(self.device)
            self.distilbert_model.eval()
            print("✓ DistilBERT model loaded")
        except Exception as e:
            print(f"✗ Error loading DistilBERT model: {e}")
            
        # Load FastText model
        try:
            self.fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            print("✓ FastText model loaded")
        except Exception as e:
            print(f"✗ Error loading FastText model: {e}")
            
        # Load sentence transformer for skill embeddings
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Sentence Transformer loaded")
        except Exception as e:
            print(f"✗ Error loading Sentence Transformer: {e}")
    
    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0.0
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU memory"""
        if not torch.cuda.is_available():
            return 4  # CPU fallback
        
        # Conservative estimates for 4070 (12GB)
        if self.gpu_memory >= 10:  # 4070 or better
            return 32
        elif self.gpu_memory >= 8:  # 4060 Ti or similar
            return 24
        elif self.gpu_memory >= 6:  # 4060 or similar
            return 16
        else:
            return 8
    
    def _estimate_time_remaining(self, processed: int, total: int, elapsed: float) -> str:
        """Estimate remaining processing time"""
        if processed == 0:
            return "calculating..."
        
        rate = processed / elapsed
        remaining = (total - processed) / rate
        
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    async def load_job_descriptions_from_db_async(self, db_path: str = None, limit: int = None):
        """Load job descriptions from SQLite database asynchronously"""
        if db_path is None:
            db_path = JOBS_DB_PATH
            
        print(f"📋 Loading job descriptions from database: {db_path}")
        print("   Reading from: jobs table, raw_text column")
        
        def load_from_db():
            try:
                conn = sqlite3.connect(db_path)
                
                # Build query
                query = "SELECT id, raw_text FROM jobs WHERE raw_text IS NOT NULL AND raw_text != ''"
                if limit:
                    query += f" LIMIT {limit}"
                
                # Execute query
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                conn.close()
                return rows
                
            except Exception as e:
                print(f"✗ Error loading from database: {e}")
                return []
        
        # Run database operation in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            rows = await loop.run_in_executor(executor, load_from_db)
        
        # Process results
        self.job_descriptions = []
        for job_id, raw_text in rows:
            if raw_text and len(raw_text.strip()) > 50:  # Filter out very short descriptions
                self.job_descriptions.append({
                    'id': job_id,
                    'raw_text': raw_text.strip()
                })
        
        print(f"✓ Loaded {len(self.job_descriptions)} job descriptions")
        
        # Time estimation
        estimated_time = len(self.job_descriptions) * 0.3 / self.batch_size  # Rough estimate
        hours = int(estimated_time // 3600)
        minutes = int((estimated_time % 3600) // 60)
        if hours > 0:
            print(f"⏱️  Estimated processing time: ~{hours}h {minutes}m")
        else:
            print(f"⏱️  Estimated processing time: ~{minutes}m")
        
        return len(self.job_descriptions) > 0

    def load_job_descriptions_from_db(self, db_path: str = None, limit: int = None):
        """Sync wrapper for async database loading"""
        return asyncio.run(self.load_job_descriptions_from_db_async(db_path, limit))
    
    def filter_relevant_sentences(self, text: str) -> str:
        """Use FastText to filter skill-relevant sentences"""
        if not self.fasttext_model:
            return text
            
        # Split into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            try:
                prediction = self.fasttext_model.predict(sentence)
                label = prediction[0][0]
                confidence = prediction[1][0]
                
                # Include if predicted as relevant with decent confidence
                if label == '__label__relevant' and confidence > 0.6:
                    relevant_sentences.append(sentence)
                elif confidence < 0.8:  # Include uncertain cases to be safe
                    relevant_sentences.append(sentence)
                    
            except Exception as e:
                # Include sentence if prediction fails
                relevant_sentences.append(sentence)
        
        return ' '.join(relevant_sentences) if relevant_sentences else text
    
    def extract_skills_from_text(self, text: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """Extract ONLY skills using DistilBERT model - ignore other entities"""
        if not self.distilbert_model or not self.tokenizer:
            return []
            
        # Filter relevant sentences first
        filtered_text = self.filter_relevant_sentences(text)
        
        # Tokenize
        encoded = self.tokenizer(
            filtered_text,
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
        
        # Label mapping - FOCUS ONLY ON SKILLS
        label_map = {0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'}
        # Note: Ignoring B-DEGREE, I-DEGREE, B-FIELD, I-FIELD, B-EXPERIENCE, I-EXPERIENCE
        
        # Extract ONLY skills
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
                        skill_text = self.extract_skill_from_original_text(current_skill_tokens, filtered_text)
                        avg_confidence = sum(current_confidences) / len(current_confidences)
                        
                        if avg_confidence >= confidence_threshold and skill_text:
                            skills.append({
                                'skill': skill_text,
                                'confidence': avg_confidence,
                                'tokens': current_skill_tokens.copy()
                            })
                    
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
                        skill_text = self.extract_skill_from_original_text(current_skill_tokens, filtered_text)
                        avg_confidence = sum(current_confidences) / len(current_confidences)
                        
                        if avg_confidence >= confidence_threshold and skill_text:
                            skills.append({
                                'skill': skill_text,
                                'confidence': avg_confidence,
                                'tokens': current_skill_tokens.copy()
                            })
                    
                    current_skill_tokens = []
                    current_confidences = []
            
            # Don't forget the last skill
            if current_skill_tokens and current_confidences:
                skill_text = self.extract_skill_from_original_text(current_skill_tokens, filtered_text)
                avg_confidence = sum(current_confidences) / len(current_confidences)
                
                if avg_confidence >= confidence_threshold and skill_text:
                    skills.append({
                        'skill': skill_text,
                        'confidence': avg_confidence,
                        'tokens': current_skill_tokens.copy()
                    })
        
        return skills
     
    def extract_skills_batch(self, texts: List[str], confidence_threshold: float = 0.5) -> List[List[Dict]]:
        """Extract skills from multiple texts in a single batch - MUCH FASTER!"""
        if not self.distilbert_model or not self.tokenizer:
            return [[] for _ in texts]
        
        # Filter relevant sentences for all texts
        filtered_texts = []
        for text in texts:
            filtered_text = self.filter_relevant_sentences(text)
            filtered_texts.append(filtered_text)
        
        # Batch tokenization
        encoded = self.tokenizer(
            filtered_texts,
            add_special_tokens=True,
            max_length=512,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        offset_mappings = encoded['offset_mapping']
        
        # Batch forward pass - THIS IS THE KEY OPTIMIZATION!
        with torch.no_grad():
            outputs = self.distilbert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
        
        # Get predictions and probabilities for entire batch
        probs = torch.nn.functional.softmax(logits, dim=2)
        predictions = torch.argmax(logits, dim=2).cpu().numpy()
        
        # Label mapping - FOCUS ONLY ON SKILLS
        label_map = {0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'}
        
        # Process each text in the batch
        all_skills = []
        for batch_idx, (text, offset_mapping) in enumerate(zip(filtered_texts, offset_mappings)):
            offset_mapping = offset_mapping.numpy()
            
            # Extract skills for this specific text
            skills = []
            current_skill_tokens = []
            current_confidences = []
            
            for i in range(1, len(predictions[batch_idx])):  # Skip [CLS] token
                if attention_mask[batch_idx][i] == 0:  # Skip padding
                    break
                    
                pred_label = label_map.get(predictions[batch_idx][i], 'O')
                confidence = probs[batch_idx][i][predictions[batch_idx][i]].item()
                
                # Get token text
                if i < len(offset_mapping):
                    start, end = offset_mapping[i]
                    if start < len(text) and end <= len(text):
                        token_text = text[start:end].strip()
                    else:
                        token_text = ""
                else:
                    token_text = ""
                
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
                        
                        if avg_confidence >= confidence_threshold and skill_text:
                            skills.append({
                                'skill': skill_text,
                                'confidence': avg_confidence,
                                'tokens': current_skill_tokens.copy()
                            })
                    
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
                        
                        if avg_confidence >= confidence_threshold and skill_text:
                            skills.append({
                                'skill': skill_text,
                                'confidence': avg_confidence,
                                'tokens': current_skill_tokens.copy()
                            })
                    
                    current_skill_tokens = []
                    current_confidences = []
            
            # Don't forget the last skill
            if current_skill_tokens and current_confidences:
                skill_text = self.extract_skill_from_original_text(current_skill_tokens, text)
                avg_confidence = sum(current_confidences) / len(current_confidences)
                
                if avg_confidence >= confidence_threshold and skill_text:
                    skills.append({
                        'skill': skill_text,
                        'confidence': avg_confidence,
                        'tokens': current_skill_tokens.copy()
                    })
            
            all_skills.append(skills)
        
        return all_skills
    
    def extract_skill_from_original_text(self, token_offsets: List[Tuple[int, int]], original_text: str) -> str:
        """Extract skill text from original source using token offsets - preserves original spacing"""
        if not token_offsets or not original_text:
            return ""
        
        # Find the start and end positions of the entire skill
        start_pos = token_offsets[0][0]  # Start of first token
        end_pos = token_offsets[-1][1]   # End of last token
        
        # Extract the exact text from original source
        if start_pos < len(original_text) and end_pos <= len(original_text):
            skill_text = original_text[start_pos:end_pos]
        else:
            # Fallback: try to extract individual tokens if offsets are invalid
            skill_parts = []
            for start, end in token_offsets:
                if start < len(original_text) and end <= len(original_text):
                    skill_parts.append(original_text[start:end])
            skill_text = ' '.join(skill_parts)
        
        # Basic cleaning only
        skill_text = skill_text.strip()
        
        # Remove obvious artifacts but preserve intended spacing
        skill_text = re.sub(r'\s+', ' ', skill_text)  # Multiple spaces -> single space
        skill_text = re.sub(r'^(experience|knowledge|skills?|proficiency)\s+(in|with|of)\s+', '', skill_text, flags=re.IGNORECASE)
        skill_text = re.sub(r'\s+(experience|skills?|knowledge|proficiency)$', '', skill_text, flags=re.IGNORECASE)
        
        return skill_text.strip()
     
    def reconstruct_skill_text(self, tokens: List[str]) -> str:
        """Reconstruct skill text from tokens, handling subword tokenization properly"""
        if not tokens:
            return ""
        
        # Handle subword tokens (tokens starting with ## in BERT/DistilBERT)
        reconstructed_tokens = []
        for token in tokens:
            if token.startswith('##'):
                # This is a subword token - merge with previous token
                if reconstructed_tokens:
                    reconstructed_tokens[-1] += token[2:]  # Remove ## prefix
                else:
                    reconstructed_tokens.append(token[2:])  # Edge case
            else:
                reconstructed_tokens.append(token)
        
        # Join tokens with spaces
        skill_text = ' '.join(reconstructed_tokens)
        
        # Basic cleaning
        skill_text = re.sub(r'\s+', ' ', skill_text)  # Multiple spaces -> single space
        skill_text = skill_text.strip()
        
        # Fix common tokenization issues
        skill_text = re.sub(r'\b([a-z])\s+([a-z])\s+([a-z])\s+([a-z])\b', r'\1\2\3\4', skill_text)  # "h v a c" -> "hvac"
        skill_text = re.sub(r'\b([a-z]+)\s+([a-z]{1,2})\s+([a-z]+)\b', r'\1\2\3', skill_text)  # "spread s heet" -> "spreadsheet"
        skill_text = re.sub(r'\b([a-z]+)\s+([a-z]{1,3})\s+([a-z]+)\s+([a-z]+)\b', r'\1\2\3\4', skill_text)  # "java script" -> "javascript"
        
        # Specific common fixes
        skill_text = re.sub(r'\bspread\s*s?\s*hee?t\b', 'spreadsheet', skill_text, flags=re.IGNORECASE)
        skill_text = re.sub(r'\bh\s*v\s*a\s*c\b', 'hvac', skill_text, flags=re.IGNORECASE)
        skill_text = re.sub(r'\bp\s*lum\s*bing\b', 'plumbing', skill_text, flags=re.IGNORECASE)
        skill_text = re.sub(r'\bjava\s*script\b', 'javascript', skill_text, flags=re.IGNORECASE)
        skill_text = re.sub(r'\bdata\s*base\b', 'database', skill_text, flags=re.IGNORECASE)
        skill_text = re.sub(r'\bweb\s*site\b', 'website', skill_text, flags=re.IGNORECASE)
        skill_text = re.sub(r'\bsoft\s*ware\b', 'software', skill_text, flags=re.IGNORECASE)
        skill_text = re.sub(r'\bmicro\s*soft\b', 'microsoft', skill_text, flags=re.IGNORECASE)
        skill_text = re.sub(r'\bin\s*vent\b', 'inventory', skill_text, flags=re.IGNORECASE)  # Common context
        
        # Remove common prefixes/suffixes that might be artifacts
        skill_text = re.sub(r'^(experience|knowledge|skills?|proficiency)\s+(in|with|of)\s+', '', skill_text, flags=re.IGNORECASE)
        skill_text = re.sub(r'\s+(experience|skills?|knowledge|proficiency)$', '', skill_text, flags=re.IGNORECASE)
        
        return skill_text.strip()
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize skill text for consistency"""
        # Convert to lowercase
        skill = skill.lower().strip()
        
        # Common normalizations
        normalizations = {
            'javascript': 'javascript',
            'js': 'javascript',
            'node.js': 'node.js',
            'nodejs': 'node.js',
            'react.js': 'react',
            'reactjs': 'react',
            'vue.js': 'vue',
            'vuejs': 'vue',
            'angular.js': 'angular',
            'angularjs': 'angular',
            'c++': 'c++',
            'cplusplus': 'c++',
            'c#': 'c#',
            'csharp': 'c#',
            'sql server': 'sql server',
            'microsoft sql server': 'sql server',
            'mysql': 'mysql',
            'postgresql': 'postgresql',
            'postgres': 'postgresql',
            'mongodb': 'mongodb',
            'mongo': 'mongodb',
            'amazon web services': 'aws',
            'amazon aws': 'aws',
            'google cloud platform': 'gcp',
            'google cloud': 'gcp',
            'machine learning': 'machine learning',
            'ml': 'machine learning',
            'artificial intelligence': 'artificial intelligence',
            'ai': 'artificial intelligence',
            'data science': 'data science',
            'rest api': 'rest api',
            'restful api': 'rest api',
            'api': 'api',
        }
        
        return normalizations.get(skill, skill)
    
    def process_all_job_descriptions(self, max_jobs: int = None):
        """Process all job descriptions through the SKILLS-ONLY pipeline"""
        print("🎯 Processing job descriptions - SKILLS EXTRACTION ONLY")
        print("   Ignoring: degrees, experience levels, job titles, responsibilities")
        print("   Focusing: technical skills, tools, technologies")
        
        jobs_to_process = self.job_descriptions[:max_jobs] if max_jobs else self.job_descriptions
        
        all_skills = []
        skill_frequency = Counter()
        job_skill_data = []
        
        for i, job_data in enumerate(tqdm(jobs_to_process, desc="Extracting skills")):
            job_id = job_data['id']
            raw_text = job_data['raw_text']
            
            if not raw_text or len(raw_text) < 50:  # Skip very short descriptions
                continue
            
            # Extract ONLY skills
            extracted_skills = self.extract_skills_from_text(raw_text)
            
            # Normalize and filter skills
            job_skills = []
            for skill_data in extracted_skills:
                normalized_skill = self.normalize_skill(skill_data['skill'])
                
                # Filter out very short or common words
                if len(normalized_skill) < 2 or normalized_skill in ['and', 'or', 'with', 'the', 'for', 'in', 'to', 'of', 'a', 'an']:
                    continue
                
                skill_entry = {
                    'skill': normalized_skill,
                    'original': skill_data['skill'],
                    'confidence': skill_data['confidence']
                }
                
                job_skills.append(skill_entry)
                all_skills.append(skill_entry)
                skill_frequency[normalized_skill] += 1
            
            job_skill_data.append({
                'job_id': job_id,
                'db_id': job_id,  # Keep reference to database ID
                'skills_only': job_skills,  # Only skills, no other data
                'skill_count': len(job_skills),
                'confidence_avg': sum(s['confidence'] for s in job_skills) / len(job_skills) if job_skills else 0
            })
        
        self.extracted_skills_data = job_skill_data
        self.skill_statistics = {
            'total_jobs_processed': len(job_skill_data),
            'total_skills_extracted': len(all_skills),
            'unique_skills': len(skill_frequency),
            'skill_frequency': dict(skill_frequency),
            'top_skills': skill_frequency.most_common(100),  # More top skills for analysis
            'avg_skills_per_job': len(all_skills) / len(job_skill_data) if job_skill_data else 0,
            'avg_confidence': sum(s['confidence'] for s in all_skills) / len(all_skills) if all_skills else 0
        }
        
        print(f"✓ Processed {len(job_skill_data)} jobs")
        print(f"✓ Extracted {len(all_skills)} skill instances")
        print(f"✓ Found {len(skill_frequency)} unique skills")
        print(f"✓ Average skills per job: {self.skill_statistics['avg_skills_per_job']:.1f}")
        print(f"✓ Average confidence: {self.skill_statistics['avg_confidence']:.2f}")
        
        return all_skills, skill_frequency
    
    async def process_all_job_descriptions_async(self, max_jobs: int = None):
        """Process all job descriptions through the SKILLS-ONLY pipeline with async optimization"""
        print("🎯 Processing job descriptions - SKILLS EXTRACTION ONLY (ASYNC + BATCHED)")
        print("   Ignoring: degrees, experience levels, job titles, responsibilities")
        print("   Focusing: technical skills, tools, technologies")
        print(f"   Batch size: {self.batch_size} jobs per GPU batch")
        
        jobs_to_process = self.job_descriptions[:max_jobs] if max_jobs else self.job_descriptions
        total_jobs = len(jobs_to_process)
        
        all_skills = []
        skill_frequency = Counter()
        job_skill_data = []
        
        # Initialize timing
        self.start_time = time.time()
        self.processed_jobs = 0
        
        # Process in batches
        for batch_start in range(0, total_jobs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_jobs)
            batch_jobs = jobs_to_process[batch_start:batch_end]
            
            # Extract job texts and IDs for this batch
            batch_texts = [job['raw_text'] for job in batch_jobs if job['raw_text'] and len(job['raw_text']) >= 50]
            batch_ids = [job['id'] for job in batch_jobs if job['raw_text'] and len(job['raw_text']) >= 50]
            
            if not batch_texts:
                continue
            
            # BATCH PROCESS - This is where the magic happens!
            batch_skills = self.extract_skills_batch(batch_texts)
            
            # Process results for each job in the batch
            for job_idx, (job_id, extracted_skills) in enumerate(zip(batch_ids, batch_skills)):
                # Normalize and filter skills
                job_skills = []
                for skill_data in extracted_skills:
                    normalized_skill = self.normalize_skill(skill_data['skill'])
                    
                    # Filter out very short or common words
                    if len(normalized_skill) < 2 or normalized_skill in ['and', 'or', 'with', 'the', 'for', 'in', 'to', 'of', 'a', 'an']:
                        continue
                    
                    skill_entry = {
                        'skill': normalized_skill,
                        'original': skill_data['skill'],
                        'confidence': skill_data['confidence']
                    }
                    
                    job_skills.append(skill_entry)
                    all_skills.append(skill_entry)
                    skill_frequency[normalized_skill] += 1
                
                job_skill_data.append({
                    'job_id': job_id,
                    'db_id': job_id,
                    'skills_only': job_skills,
                    'skill_count': len(job_skills),
                    'confidence_avg': sum(s['confidence'] for s in job_skills) / len(job_skills) if job_skills else 0
                })
            
            # Update progress
            self.processed_jobs = batch_end
            elapsed_time = time.time() - self.start_time
            eta = self._estimate_time_remaining(self.processed_jobs, total_jobs, elapsed_time)
            
            # Progress update
            progress = (self.processed_jobs / total_jobs) * 100
            rate = self.processed_jobs / elapsed_time
            print(f"⏳ Progress: {progress:.1f}% ({self.processed_jobs}/{total_jobs}) | "
                  f"Rate: {rate:.1f} jobs/sec | ETA: {eta}")
            
            # Memory cleanup
            if batch_start % (self.batch_size * 10) == 0:  # Every 10 batches
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Allow other async tasks to run
            await asyncio.sleep(0.001)
        
        # Store results
        self.extracted_skills_data = job_skill_data
        self.skill_statistics = {
            'total_jobs_processed': len(job_skill_data),
            'total_skills_extracted': len(all_skills),
            'unique_skills': len(skill_frequency),
            'skill_frequency': dict(skill_frequency),
            'top_skills': skill_frequency.most_common(100),
            'avg_skills_per_job': len(all_skills) / len(job_skill_data) if job_skill_data else 0,
            'avg_confidence': sum(s['confidence'] for s in all_skills) / len(all_skills) if all_skills else 0,
            'processing_time_seconds': time.time() - self.start_time,
            'processing_rate_jobs_per_second': len(job_skill_data) / (time.time() - self.start_time)
        }
        
        total_time = time.time() - self.start_time
        print(f"\n✅ ASYNC PROCESSING COMPLETED!")
        print(f"✓ Processed {len(job_skill_data)} jobs in {total_time:.1f} seconds")
        print(f"✓ Extracted {len(all_skills)} skill instances")
        print(f"✓ Found {len(skill_frequency)} unique skills")
        print(f"✓ Average skills per job: {self.skill_statistics['avg_skills_per_job']:.1f}")
        print(f"✓ Average confidence: {self.skill_statistics['avg_confidence']:.2f}")
        print(f"✓ Processing rate: {self.skill_statistics['processing_rate_jobs_per_second']:.1f} jobs/second")
        
        return all_skills, skill_frequency
     
    def create_skill_embeddings(self):
        """Create embeddings for all extracted skills"""
        if not self.sentence_transformer:
            print("✗ Sentence transformer not loaded")
            return
        
        print("🧠 Creating skill embeddings for similarity analysis...")
        
        unique_skills = list(self.skill_statistics['skill_frequency'].keys())
        
        # Create context-aware skill descriptions for better embeddings
        skill_contexts = []
        for skill in unique_skills:
            # Create a context description to help with embedding quality
            context = f"Technical skill: {skill}. Professional competency in {skill}."
            skill_contexts.append(context)
        
        # Generate embeddings
        skill_embeddings = self.sentence_transformer.encode(
            skill_contexts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Store embeddings
        self.skill_vectors = {}
        for skill, embedding in zip(unique_skills, skill_embeddings):
            self.skill_vectors[skill] = embedding
        
        print(f"✓ Created embeddings for {len(unique_skills)} skills")
        print(f"✓ Vector dimension: {skill_embeddings.shape[1]}")
        
        return self.skill_vectors
    
    def analyze_skill_similarity(self, top_n: int = 10):
        """Analyze skill similarities using embeddings"""
        if not self.skill_vectors:
            print("✗ No skill vectors available")
            return
        
        print("📊 Analyzing skill similarities...")
        
        skills = list(self.skill_vectors.keys())
        embeddings = np.array(list(self.skill_vectors.values()))
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find most similar skills for each skill
        skill_similarities = {}
        
        for i, skill in enumerate(skills):
            # Get similarity scores for this skill
            similarities = similarity_matrix[i]
            
            # Get indices of most similar skills (excluding self)
            similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
            
            similar_skills = []
            for idx in similar_indices:
                similar_skills.append({
                    'skill': skills[idx],
                    'similarity': similarities[idx],
                    'frequency': self.skill_statistics['skill_frequency'].get(skills[idx], 0)
                })
            
            skill_similarities[skill] = similar_skills
        
        # Show some examples
        print("\n🎯 Skill Similarity Examples:")
        example_skills = ['python', 'javascript', 'react', 'aws', 'machine learning', 'sql']
        
        for skill in example_skills:
            if skill in skill_similarities:
                print(f"\n{skill.upper()}:")
                for similar in skill_similarities[skill][:5]:
                    print(f"  {similar['skill']} (similarity: {similar['similarity']:.3f})")
        
        return skill_similarities
    
    def cluster_skills(self, n_clusters: int = 20):
        """Cluster skills into related groups"""
        if not self.skill_vectors:
            print("✗ No skill vectors available")
            return
        
        print(f"🔗 Clustering skills into {n_clusters} related groups...")
        
        skills = list(self.skill_vectors.keys())
        embeddings = np.array(list(self.skill_vectors.values()))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Organize skills by cluster
        clusters = defaultdict(list)
        for skill, cluster_id in zip(skills, cluster_labels):
            clusters[cluster_id].append({
                'skill': skill,
                'frequency': self.skill_statistics['skill_frequency'].get(skill, 0)
            })
        
        # Sort skills within each cluster by frequency
        for cluster_id in clusters:
            clusters[cluster_id].sort(key=lambda x: x['frequency'], reverse=True)
        
        # Display clusters
        print("\n🎯 Skill Clusters (Related Technologies):")
        for cluster_id, cluster_skills in clusters.items():
            if len(cluster_skills) > 1:  # Only show clusters with multiple skills
                print(f"\nCluster {cluster_id}:")
                for skill_data in cluster_skills[:8]:  # Show top 8 skills in cluster
                    print(f"  {skill_data['skill']} (freq: {skill_data['frequency']})")
        
        return dict(clusters)
    
    def save_results(self, output_dir: str = "skill_vectorization_results"):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving results to {output_dir}/...")
        
        # Save skill statistics
        with open(f"{output_dir}/skill_statistics.json", 'w') as f:
            json.dump(self.skill_statistics, f, indent=2)
        
        # Save skill vectors
        with open(f"{output_dir}/skill_vectors.pkl", 'wb') as f:
            pickle.dump(self.skill_vectors, f)
        
        # Save extracted skills data
        with open(f"{output_dir}/extracted_skills_data.json", 'w') as f:
            json.dump(self.extracted_skills_data, f, indent=2)
        
        # Save top skills as CSV for easy viewing
        top_skills_df = pd.DataFrame(
            self.skill_statistics['top_skills'],
            columns=['skill', 'frequency']
        )
        top_skills_df.to_csv(f"{output_dir}/top_skills.csv", index=False)
        
        print(f"✓ Results saved to {output_dir}/")

async def main_async():
    """Main execution function - OPTIMIZED SKILLS-ONLY PIPELINE"""
    print("🚀" + "="*60)
    print("OPTIMIZED SKILLS-ONLY VECTORIZATION PIPELINE")
    print("="*65)
    print("Focus: Pure skill extraction and alignment")
    print("Ignore: Job titles, seniority, responsibilities, degrees")
    print("Optimizations: Async processing + GPU batching")
    print("="*65)
    
    # Initialize pipeline with auto-optimized batch size
    pipeline = SkillVectorizationPipeline()
    
    # Load job descriptions from database (async)
    success = await pipeline.load_job_descriptions_from_db_async()
    if not success:
        print("Failed to load job descriptions from database. Please check the database path.")
        return
    
    # Process job descriptions with async optimization
    print(f"\n📋 Processing {len(pipeline.job_descriptions)} jobs with ASYNC + BATCHING...")
    all_skills, skill_frequency = await pipeline.process_all_job_descriptions_async()
    
    # Create skill embeddings
    skill_vectors = pipeline.create_skill_embeddings()
    
    # Analyze similarities
    similarities = pipeline.analyze_skill_similarity()
    
    # Cluster skills
    clusters = pipeline.cluster_skills(n_clusters=25)
    
    # Save results
    pipeline.save_results()
    
    print("\n" + "="*60)
    print("🚀 OPTIMIZED SKILLS-ONLY VECTORIZATION COMPLETED")
    print("="*62)
    print(f"✓ Processed {pipeline.skill_statistics['total_jobs_processed']} jobs")
    print(f"✓ Found {pipeline.skill_statistics['unique_skills']} unique skills")
    print(f"✓ Created {len(skill_vectors)} skill embeddings")
    print(f"✓ Average {pipeline.skill_statistics['avg_skills_per_job']:.1f} skills per job")
    print(f"✓ Processing time: {pipeline.skill_statistics.get('processing_time_seconds', 0):.1f} seconds")
    print(f"✓ Processing rate: {pipeline.skill_statistics.get('processing_rate_jobs_per_second', 0):.1f} jobs/sec")
    print(f"✓ Results saved to 'skill_vectorization_results/'")
    print("\n🔍 Use skill_visualization.py to explore skill relationships!")

def main():
    """Sync wrapper for async main function"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 