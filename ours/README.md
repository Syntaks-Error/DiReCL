# Eureka-style Numeric ML-IRL (ours)

This implementation follows your method:

1. **Outer loop**: generate reward code skeletons (Eureka-like candidate + reflection rounds).
2. **Numeric parsing**: parse numeric literals in each reward code into trainable parameters.
3. **Inner loop**: update those numeric parameters using ML-IRL loss from expert demos in `./expert_data`.

## Files

- `run_eureka_numeric_irl.py`: entry script
- `eureka_numeric_irl/reward_parser.py`: parses numeric literals to trainable `torch.nn.Parameter`
- `eureka_numeric_irl/optimizer.py`: ML-IRL inner-loop optimization
- `eureka_numeric_irl/outer_loop.py`: Eureka-like outer loop and reflection
- `configs/eureka_numeric_irl_ant.yaml`: default config for Ant

## Run

From `./ours`:

```bash
python run_eureka_numeric_irl.py --config configs/eureka_numeric_irl_ant.yaml
```

## Output

A timestamped folder under `outputs/eureka_numeric_irl/` with:

- `best_reward.py`: best reward skeleton code
- `summary.json`: round history and learned numeric parameters
