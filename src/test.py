for i, scenario in enumerate(scenarios):
    print(f"\n--- Scenario {i + 1}/{len(scenarios)}: {scenario.upper()} ---")
    
    # Recreate env with correct scenario
    env = gym.make(
        "RallyDriving-v0",
        renders=render,
        isDiscrete=False,
        reward_callback=custom_reward,
        observation_callback=None,
        scenario=scenario,           # ← pass scenario here
    )
    model = PPO.load(model_path, env=env)
    
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        if render:
            time.sleep(0.005)

    unwrapped = env.unwrapped
    completed = unwrapped.current_checkpoint_idx
    total = len(unwrapped.checkpoints)
    print(f"  Reward: {total_reward:+.2f}   Steps: {steps}   "
          f"Checkpoints: {completed}/{total}")
    env.close()