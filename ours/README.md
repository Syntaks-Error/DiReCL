# Eureka-style Numeric ML-IRL (ours)

This implementation follows your method with an exact Eureka-style outer loop:

1. **Outer loop (exact Eureka pattern)**: generate reward code by LLM API prompting, then reflect by adding metric feedback into the next prompt.
2. **Numeric parsing**: parse numeric literals in each reward code into trainable parameters.
3. **Inner loop**: update those numeric parameters using ML-IRL loss from expert demos in `./expert_data`.

## Files

- `run_eureka_numeric_irl.py`: entry script
- `eureka_numeric_irl/reward_parser.py`: parses numeric literals to trainable `torch.nn.Parameter`
- `eureka_numeric_irl/optimizer.py`: ML-IRL inner-loop optimization
- `eureka_numeric_irl/outer_loop.py`: exact Eureka-style prompting/reflection outer loop
- `configs/eureka_numeric_irl_ant.yaml`: default config for Ant

## Run

From `./ours`:

```bash
export OPENAI_API_KEY=your_key
python run_eureka_numeric_irl.py --config configs/eureka_numeric_irl_ant.yaml
```

## Output

A timestamped folder under `outputs/eureka_numeric_irl/` with:

- `best_reward.py`: best reward skeleton code
- `summary.json`: iteration history and learned numeric parameters
- `messages.json`: saved prompt/response history for reflection
