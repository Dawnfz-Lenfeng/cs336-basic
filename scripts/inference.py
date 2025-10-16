import argparse

import torch
import yaml

from cs336_basics.data import load_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import BPETokenizer
from scripts.config import Config


def load_config(config_path: str) -> Config:
    """Load config from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Config,
) -> TransformerLM:
    """Load model from checkpoint."""
    model = TransformerLM(**config.model.model_dump()).to(config.training.device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    return model


def generate_text(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str = "cuda",
) -> str:
    """Generate text from prompt."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)

    # Generate
    output_ids = model.generate(
        input_tensor,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode
    generated_text = tokenizer.decode(output_ids.tolist())

    return generated_text


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="tokenizer/vocab.json",
        help="Path to vocabulary JSON file (default: tokenizer/vocab.json)",
    )
    parser.add_argument(
        "--merges",
        type=str,
        default="tokenizer/merges.txt",
        help="Path to merges file (default: tokenizer/merge.txt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Prompt to generate from (default: 'Once upon a time')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold (default: None)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Load tokenizer
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=["<|endoftext|>"],
    )

    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
    )

    # Generate text
    print(f"\nPrompt: {args.prompt}\n")
    print("Generating...\n")
    print("-" * 80)

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=config.training.device,
    )

    print(generated_text)
    print("-" * 80)


if __name__ == "__main__":
    main()
