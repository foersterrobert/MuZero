import torch


def _get_random_action(env, state):
    valid_actions = torch.nonzero(env.get_valid_actions(state), as_tuple=False).squeeze(1)
    random_index = torch.randint(len(valid_actions), (1,)).item()
    return int(valid_actions[random_index].item())


def _get_greedy_policy_action(model, env, state, player):
    neutral_state = env.change_perspective(state, player)
    encoded_state = env.get_encoded_state(neutral_state).unsqueeze(0).to(model.device)

    hidden_state = model.represent(encoded_state)
    policy_logits, _ = model.predict(hidden_state)
    policy_logits = policy_logits.squeeze(0).detach().cpu()

    valid_actions = env.get_valid_actions(state).cpu()
    masked_logits = policy_logits.clone()
    masked_logits[valid_actions == 0] = -float("inf")

    return int(torch.argmax(masked_logits).item())


@torch.no_grad()
def evaluate_model_vs_random(model, env, num_games=10):
    results = {"model_wins": 0, "random_wins": 0, "draws": 0}

    was_training = model.training
    model.eval()

    for game_idx in range(num_games):
        state = env.get_initial_state()

        # Alternate starting side across games to reduce first-player bias.
        model_player = 1 if game_idx % 2 == 0 else -1
        player = 1

        while True:
            if player == model_player:
                action = _get_greedy_policy_action(model, env, state, player)
            else:
                action = _get_random_action(env, state)

            state = env.get_next_state(state, action, player)
            value, is_terminal = env.get_value_and_terminated(state, action)

            if is_terminal:
                if value == 0.0:
                    results["draws"] += 1
                elif player == model_player:
                    results["model_wins"] += 1
                else:
                    results["random_wins"] += 1
                break

            player = env.get_opponent(player)

    if was_training:
        model.train()

    return results
