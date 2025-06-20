import os
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('NEBIUS_API_KEY')

from argparse import ArgumentParser

from scripts import (
    simple_RAG, 
    RAG_with_semantic_chunking,
    context_enriched_retrieval,
    context_headers_RAG
)

def arg_parser():
    parser = ArgumentParser(description="Run a simple RAG pipeline with OpenAI.")
    parser.add_argument("--txt_file_path", type=str, required=True, help="Path to the text file to read.")
    parser.add_argument("--pdf_file_path", type=str, required=True, help="Path to the pdf file to read.")
    parser.add_argument("--user_query", type=str, required=True, help="User query to answer.")
    parser.add_argument("--RAG_version", type=str, default="simpleRAG", help="Version of the RAG to use.")
    return parser.parse_args()

def main():
    args = arg_parser()

    ## Initialize OpenAI client
    openai_client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=api_key
    )

    ## Create an instance of RAG based on the specified version
    if args.RAG_version == "simpleRAG":
        logger.info("Running simpleRAG version of RAG.")
        ## Initialize the simpleRAG class
        rag = simple_RAG.simpleRAG(
            txt_file_path=args.txt_file_path,
            user_query=args.user_query,
            client=openai_client
        )
        ## Run the RAG pipeline
        ai_generated_answer = rag.pipeline()
        logger.info(f"AI Generated Answer: {ai_generated_answer}")
    elif args.RAG_version == "RAG with semantic chunking":
        logger.info("Running RAG with semantic chunking version of RAG.")
        ## Initialize the simpleRAG class with semantic chunking
        rag = RAG_with_semantic_chunking.ragWithSemanticChunking(
            txt_file_path=args.txt_file_path,
            user_query=args.user_query,
            client=openai_client
        )
        ## Run the RAG pipeline
        ai_generated_answer = rag.pipeline()
        logger.info(f"AI Generated Answer: {ai_generated_answer}")
    elif args.RAG_version == "context enriched retrieval":
        logger.info("Running context enriched retrieval.")
        ## Initialize the context_enriched_retrieval class
        rag = context_enriched_retrieval.contextEnrichedRAG(
            txt_file_path=args.txt_file_path,
            user_query=args.user_query,
            client=openai_client
        )
        ## Run the RAG pipeline
        ai_generated_answer = rag.pipeline()
        logger.info(f"AI Generated Answer: {ai_generated_answer}")
    elif args.RAG_version == "context headers RAG":
        logger.info("Running context headers RAG.")
        ## Initialize the context_headers_RAG class
        rag = context_headers_RAG.contextHeadersRAG(
            pdf_file_path=args.pdf_file_path,
            user_query=args.user_query,
            client=openai_client
        )
        ## Run the RAG pipeline
        rag.pipeline()
        # logger.info(f"AI Generated Answer: {ai_generated_answer}")
    else:
        raise ValueError(f"Unsupported RAG version: {args.RAG_version}")
    
    
if __name__ == "__main__":
    main()

