# Multi-Model Job Search Evaluation Script

This script allows you to evaluate and compare different types of machine learning models for job search and retrieval tasks.

## Features

- **Multiple Model Support**: Load and compare different model types
- **Flexible Model Types**: Supports SentenceTransformer, HuggingFace AutoModel, DistilBERT, and more
- **Automatic .pt Weight Loading**: Automatically detects and loads recovered weights from .pt files
- **Comprehensive Evaluation**: Generates detailed results and comparison reports
- **GPU/CPU Support**: Automatic device detection with CUDA support

## Supported Model Types

1. **`sentence-transformer`** - Standard SentenceTransformer models
2. **`sentence-transformer-recovered`** - SentenceTransformer with additional .pt weight files
3. **`huggingface-auto`** - Direct HuggingFace AutoModel loading
4. **`distilbert`** - DistilBERT for token classification (future support)

## Installation

Ensure you have the required dependencies:

```bash
pip install torch sentence-transformers transformers beautifulsoup4 numpy
```

## Usage Examples

### Single Model Evaluation

```bash
python model_evaluation.py \
  --model-1 ./nomic-bert-2048 \
  --model-1-type sentence-transformer-recovered \
  --jobs-data-path ./nomic-bert-2048/jobs.json
```

### Compare Two Models

```bash
python model_evaluation.py \
  --model-1 ./nomic-bert-2048 \
  --model-1-type sentence-transformer-recovered \
  --model-2 ./other-model \
  --model-2-type sentence-transformer \
  --jobs-data-path ./jobs.json
```

### Custom Query

```bash
python model_evaluation.py \
  --model-1 ./nomic-bert-2048 \
  --model-1-type sentence-transformer-recovered \
  --jobs-data-path ./nomic-bert-2048/jobs.json \
  --query "Senior Data Scientist in New York with Machine Learning experience"
```

### Force GPU Usage with Custom Cache Directory

```bash
python model_evaluation.py \
  --model-1 ./nomic-bert-2048 \
  --model-1-type sentence-transformer-recovered \
  --jobs-data-path ./nomic-bert-2048/jobs.json \
  --device cuda \
  --cache-dir ./custom_cache
```

### Force Recomputation (Ignore Cache)

```bash
python model_evaluation.py \
  --model-1 ./nomic-bert-2048 \
  --model-1-type sentence-transformer-recovered \
  --jobs-data-path ./nomic-bert-2048/jobs.json \
  --force-recompute
```

### Disable Caching Completely

```bash
python model_evaluation.py \
  --model-1 ./nomic-bert-2048 \
  --model-1-type sentence-transformer-recovered \
  --jobs-data-path ./nomic-bert-2048/jobs.json \
  --no-cache
```

### Two Model Comparison with All Options

```bash
python model_evaluation.py \
  --model-1 ./nomic-bert-2048 \
  --model-1-type sentence-transformer-recovered \
  --model-2 ./huggingface-model \
  --model-2-type huggingface-auto \
  --jobs-data-path ./jobs.json \
  --query "Python Engineer with AI experience" \
  --device auto \
  --cache-dir ./evaluation_cache
```

## Command Line Arguments

### Required Arguments

- **`--model-1`** (required): Path to first model directory
  - Example: `--model-1 ./nomic-bert-2048`
  - Must be a valid directory containing model files

- **`--model-1-type`** (required): Type of first model
  - Choices: `sentence-transformer`, `sentence-transformer-recovered`, `huggingface-auto`, `distilbert`
  - Example: `--model-1-type sentence-transformer-recovered`

- **`--jobs-data-path`** (required): Path to jobs JSON data file
  - Example: `--jobs-data-path ./nomic-bert-2048/jobs.json`
  - Must be a valid JSON file containing job data

### Optional Model Comparison Arguments

- **`--model-2`** (optional): Path to second model directory for comparison
  - Example: `--model-2 ./other-model`
  - Must be specified together with `--model-2-type`

- **`--model-2-type`** (optional): Type of second model
  - Choices: `sentence-transformer`, `sentence-transformer-recovered`, `huggingface-auto`, `distilbert`
  - Example: `--model-2-type sentence-transformer`
  - Must be specified together with `--model-2`

### Evaluation Configuration Arguments

- **`--query`** (optional): Search query to evaluate
  - Example: `--query "Senior Data Scientist in New York with Machine Learning experience"`
  - If not provided, will be collected interactively during script execution
  - Default interactive fallback: "Python Developer in San Francisco with Salary 100000"

- **`--device`** (optional): Computing device to use
  - Choices: `cpu`, `cuda`, `auto`
  - Default: `auto` (automatically detects CUDA availability)
  - Example: `--device cuda` to force GPU usage

### Caching and Performance Arguments

- **`--cache-dir`** (optional): Directory to store embedding cache files
  - Default: `embedding_cache`
  - Example: `--cache-dir ./my_cache`
  - Embeddings are cached to speed up repeated evaluations

- **`--force-recompute`** (optional): Force recomputation of embeddings
  - Flag: No value needed
  - Example: `--force-recompute`
  - Ignores existing cache and recomputes all embeddings
  - Useful when model or data has changed

- **`--no-cache`** (optional): Disable embedding caching completely
  - Flag: No value needed
  - Example: `--no-cache`
  - Embeddings will not be saved or loaded from cache
  - Useful for testing or when cache directory is not writable

### Parameter Validation Rules

1. **Required Parameters**: `--model-1`, `--model-1-type`, and `--jobs-data-path` must always be provided
2. **Model Comparison**: If using `--model-2`, you must also specify `--model-2-type` (and vice versa)
3. **Model Types**: Must be one of the supported types: `sentence-transformer`, `sentence-transformer-recovered`, `huggingface-auto`, `distilbert`
4. **Device Selection**: `auto` will automatically choose CUDA if available, otherwise CPU
5. **Caching**: `--force-recompute` and `--no-cache` are mutually exclusive (use one or the other, not both)

### Common Parameter Combinations

| Use Case | Recommended Parameters |
|----------|----------------------|
| **Quick single model test** | `--model-1`, `--model-1-type`, `--jobs-data-path`, `--query` |
| **Model comparison** | Add `--model-2`, `--model-2-type` to above |
| **First time setup** | Add `--cache-dir ./cache` to store embeddings |
| **Development/testing** | Add `--no-cache` to avoid caching |
| **Model updated** | Add `--force-recompute` to refresh embeddings |
| **CPU-only systems** | Add `--device cpu` |
| **Production evaluation** | Use all parameters with specific cache directory |

### Interactive vs Non-Interactive Mode

- **Non-Interactive**: Provide `--query` parameter to run completely automated
- **Interactive**: Omit `--query` to be prompted for search query and evaluation parameters
- **Hybrid**: Provide `--query` but still get prompted for accuracy testing parameters

## Output Files

The script generates several output files in the `evaluation_results/` directory:

### Single Model
- `model1_retrieved_jobs_TIMESTAMP.json` - Search results for model 1
- `model2_retrieved_jobs_TIMESTAMP.json` - Search results for model 2 (if provided)

### Comparison (when multiple models)
- `model_comparison_TIMESTAMP.json` - Comparison metrics between models

## Output Format

Each results file contains:

```json
{
  "query": "Python Developer in San Francisco with Salary 100000",
  "results": [
    {
      "rank": 1,
      "score": 0.6549582481384277,
      "job_id": 671,
      "job_title": "Développeur Python R&D - H/F",
      "company_name": "MARGO",
      "location_raw": "Paris",
      "job_type": "Permanent contract /",
      "salary_range": "",
      "description": "..."
    }
  ]
}
```

## Model Requirements

### For `sentence-transformer-recovered` type:
- Model directory must contain standard SentenceTransformer files
- Optional `.pt` files will be automatically detected and loaded
- Looks for files containing "recovered" in the name first

### For `sentence-transformer` type:
- Standard SentenceTransformer model directory structure
- Must contain `config.json`, `modules.json`, etc.

### For `huggingface-auto` type:
- HuggingFace compatible model directory
- Must contain `config.json` and model weights

## Example Output

```
Using device: cuda
================================================================================
MULTI-MODEL JOB SEARCH EVALUATION
================================================================================
Start time: 2025-08-19 16:22:15.260442
Results directory: evaluation_results

==================================================
Loading model1: ./nomic-bert-2048 (sentence-transformer-recovered)
==================================================
Loading sentence-transformer-recovered model from: nomic-bert-2048
Loading recovered weights from: nomic-bert-2048\model_recovered.pt
✓ SentenceTransformer model with recovered weights loaded successfully
Loading jobs from: nomic-bert-2048\jobs.json
✓ Loaded 1010 job documents
Encoding job embeddings...
✓ Job embeddings encoded successfully
✓ model1 ready for evaluation

============================================================
RUNNING SEARCH EVALUATION
Query: Python Developer in San Francisco with Salary 100000
============================================================

Top 5 results:
  1. (0.655) Développeur Python R&D - H/F at MARGO | Paris
  2. (0.618) Senior Software Engineer (Golang/Ruby on Rails) at Crypto.com | Kuala Lumpur, Malaysia
  3. (0.605) Principal Scientist, Computational Biology at Enveda | Boulder, CO.
  4. (0.600) Accounting Technology Consultant, AI at FloQast | San Francisco, California
  5. (0.595) Store Person at Planet Innovation | Box Hill, VIC
✓ Results exported to: evaluation_results\model1_retrieved_jobs_20250819_162215.json

================================================================================
EVALUATION COMPLETE
Total time: 145.89 seconds
================================================================================
```

## Technical Details

The script efficiently handles:
- **Memory Management**: Optimized batch processing for large job datasets
- **Device Selection**: Automatic GPU/CPU detection and usage
- **Model Loading**: Robust error handling for different model types
- **Embedding Normalization**: Applies proper layer normalization and L2 normalization
- **Progress Tracking**: Shows progress bars for encoding operations

## Performance Notes

- **GPU Recommended**: For faster embedding computation with large datasets
- **Memory Usage**: Approximately ~2GB VRAM for 1000 jobs with nomic-bert-2048
- **Processing Time**: ~2-3 minutes for 1000 jobs on GPU, ~10-15 minutes on CPU

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in the code or use CPU
2. **Model not found**: Verify the model path exists and contains required files
3. **Jobs file not found**: Ensure the jobs JSON file path is correct
4. **Slow performance**: Make sure CUDA is available and being used

### Error Messages

- `Model path does not exist`: Check the `--model-1` path
- `Unsupported model type`: Use one of the supported types listed above
- `Jobs file not found`: Verify the `--jobs-data-path` parameter

## Future Enhancements

- Support for additional model types (BERT, RoBERTa, etc.)
- Batch evaluation with multiple queries
- Integration with existing evaluation scripts
- Performance benchmarking metrics
- Custom embedding dimensions support
