# Dataset Format

RUAR training parquet rows must contain `prompt`, `data_source`, and
`reward_info.ground_truth`. `prompt` is stored in chat-message format so the
tokenizer can apply the model chat template directly.

Math problems
```json
{
    "data_source": "openai/gsm8k",
    "prompt": [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\""}],
    "reward_info": {
        "ground_truth": ["72"]
    }
}
```
