import argparse
import torch
import json
from transformers import AutoTokenizer
from openrlhf.models.modeling_clara import CLaRa, CLaRaConfig

def ingest_documents(args):
    print(f"Loading CLaRa model from {args.model_path}...")
    
    # Load model (assumes Stage 1 or Stage 2 checkpoint)
    # We load it in bf16 to match training
    model = CLaRa.from_pretrained(
        args.model_path,
        training_stage="stage2", # Load in stage 2 mode to get full capability
        generation_top_k=1,
        doc_max_length=args.doc_max_length,
        compress_rate=args.compress_rate,
        quantization=args.quantization,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model.config.decoder_model_name, 
        trust_remote_code=True,
        padding_side="left" # Encoder side usually prefers left padding for generation, but right for pure encoding. 
                            # However, CLaRa compressor handles padding internally.
    )
    
    # Read documents
    with open(args.input_file, 'r') as f:
        docs = [line.strip() for line in f if line.strip()]
        
    print(f"Processing {len(docs)} documents...")
    
    results = []
    
    with torch.no_grad():
        for doc in docs:
            # Tokenize
            inputs = tokenizer(
                doc, 
                return_tensors="pt", 
                max_length=args.doc_max_length, 
                truncation=True,
                padding="max_length"
            ).to(model.decoder.device)
            
            # Compress
            # model.compress returns (compressed_embeddings, loss)
            # We only need the embeddings [1, N_Mem_Tokens, Hidden_Dim]
            compressed_embs, _ = model.compress(inputs.input_ids, inputs.attention_mask)
            
            # Generate Index Vector (Mean Pooling of compressed embeddings)
            # This single vector is what you use for Faiss/Milvus retrieval
            index_vector = torch.mean(compressed_embs, dim=1).cpu().numpy().tolist()[0]
            
            # Store the full memory vectors as payload
            memory_vectors = compressed_embs.cpu().numpy().tolist()[0]
            
            results.append({
                "text_preview": doc[:50],
                "index_vector": index_vector,
                "memory_vectors": memory_vectors
            })
            
    # Save to output
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Ingestion complete. Vectors saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained CLaRa checkpoint")
    parser.add_argument("--input_file", type=str, required=True, help="Text file with documents (one per line)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file with vectors")
    parser.add_argument("--doc_max_length", type=int, default=512)
    parser.add_argument("--compress_rate", type=int, default=32)
    parser.add_argument("--quantization", type=str, default="int4")
    args = parser.parse_args()
    
    ingest_documents(args)
