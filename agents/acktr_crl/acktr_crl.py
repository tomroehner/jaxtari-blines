# ACKTR for continual reinforcement learning: one agent trained on a sequence of
# tasks (a game plus mods)
# After each task, every task seen so far is evaluated, giving average performance,
# backward transfer and forgetting.
#
# ACKTR itself is agents/acktr/acktr.py (networks, rollout, GAE, loss, chunked training).
import os
import random
import time
from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np
import kfac_jax
import wandb
from flax.training.train_state import TrainState
import jaxtari
from jaxtari.wrappers import NormalizeObservationWrapper, ObjectCentricWrapper, PixelObsWrapper, AtariWrapper, LogWrapper, FlattenObservationWrapper
from jaxtari import spaces
from agents.acktr.acktr import Network, MLP_Network, Actor, Critic, AgentParams, Storage
from agents.ppo_crl.ppo_crl_eval import make_evaluator
from agents.run_utils import plan_chunks
from rtpt import RTPT


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False, full_action_space=False):
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None

    def thunk():
        env = jaxtari.make(env_id, mods=mods)
        env = AtariWrapper(
                env,
                sticky_actions=0.0,
                episodic_life=not eval, # only active during training
                first_fire=True,
                noop_max=30,
                # several games share one actor head: all 18 actions
                full_action_space=full_action_space,
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                use_native_downscaling=native_downscaling,
                smooth_image=False,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=not eval,
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=not eval,
                    )
                )
            )
        env = LogWrapper(env)
        return env
    return thunk


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    pixel_based = config["PIXEL_BASED"]

    tasks = list(config["TASKS"])
    num_tasks = len(tasks)
    # mods per task: TASK_TRAIN_MODS[i] is the list trained on in task i,
    # TASK_EVAL_MODS[i] the list of mod lists it is evaluated on (default: its train mods)
    task_train_mods = [list(m) for m in (config.get("TASK_TRAIN_MODS") or [])]
    task_train_mods += [[] for _ in range(num_tasks - len(task_train_mods))]
    task_eval_mods = [[list(m) for m in mods] for mods in (config.get("TASK_EVAL_MODS") or [])]
    task_eval_mods += [[task_train_mods[i]] for i in range(len(task_eval_mods), num_tasks)]

    run_name = f"{'-'.join(tasks)}_{config['EXP_NAME']}_{'pixel' if pixel_based else 'oc'}_{config['SEED']}"
    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
        mode=config.get("WANDB_MODE", "online"),
    )
    wandb.define_metric("*", step_metric="charts/global_step")

    # do not modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    num_envs = config["NUM_ENVS"]
    num_steps = config["NUM_STEPS"]
    batch_size = num_envs * num_steps
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(config["TIMESTEPS_PER_TASK"], batch_size, config.get("SCAN_STEPS", 1000))
    num_updates = scan_steps * num_chunks   # per task

    # several games: one actor head over the full action space
    multi_game = len(set(tasks)) > 1
    env_kwargs = dict(pixel_based=pixel_based, native_downscaling=config.get("NATIVE_DOWNSCALING", True),
                      full_action_space=multi_game)
    train_envs = [make_env(t, task_train_mods[i], eval=False, **env_kwargs)() for i, t in enumerate(tasks)]
    for env in train_envs:
        assert isinstance(env.action_space(), spaces.Discrete), "only discrete action space is supported"
    action_dim = train_envs[0].action_space().n

    # Object-centric observations differ in size between games: zero-pad all of
    # them to the largest one, so one network fits every task.
    if pixel_based:
        obs_shape = (4, 84, 84)
    else:
        obs_shape = (max(env.observation_space().shape[0] for env in train_envs),)

    def prepare_obs(obs):
        """One env's observation -> network input (drops the pixel channel, pads OC)."""
        if pixel_based:
            return obs.reshape(obs_shape)
        return jnp.pad(obs, (0, obs_shape[0] - obs.shape[0]))

    def linear_schedule(count):
        # anneal learning rate linearly after one update which contains
        # (NUM_MINIBATCHES * UPDATE_EPOCHS) gradient steps
        frac = 1.0 - (count / num_updates)
        return config["LEARNING_RATE"] * frac

    network = Network() if pixel_based else MLP_Network()
    actor = Actor(action_dim=action_dim)
    critic = Critic()

    # same root split as ppo
    key, init_key = jax.random.split(key, 2)
    network_key, actor_key, critic_key = jax.random.split(init_key, 3)
    network_params = network.init(network_key, jnp.zeros((1, *obs_shape)))
    hidden_example = network.apply(network_params, jnp.zeros((1, *obs_shape)))
    
    agent_params = AgentParams(
            network_params=network_params,
            actor_params=actor.init(actor_key, hidden_example),
            critic_params=critic.init(critic_key, hidden_example),
        )

    def get_action_and_value(params, obs, key):
        """sample action, calculate value and logprob"""
        hidden = network.apply(params.network_params, obs)
        logits = actor.apply(params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(params.critic_params, hidden)
        return action, logprob, value.squeeze(1), key

    def a2c_loss(params, batch):
            x, a, advantages, returns = batch
            
            hidden = network.apply(params.network_params, x)
            logits = actor.apply(params.actor_params, hidden)
            value = critic.apply(params.critic_params, hidden).squeeze()
    
            kfac_jax.register_softmax_cross_entropy_loss(logits, weight=1.0)
            kfac_jax.register_squared_error_loss(value[..., None], weight=config["VF_COEF"])
    
            logprob = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a]
            # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
            logits_norm = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
            logits_norm = logits_norm.clip(min=jnp.finfo(logits_norm.dtype).min)
            p_log_p = logits_norm * jax.nn.softmax(logits_norm)
            entropy = -p_log_p.sum(-1)
    
            if config["NORM_ADV"]:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
            # Policy loss
            pg_loss = -(logprob * advantages).mean()
    
            # Value loss
            v_loss = 0.5 * ((value - returns) ** 2).mean()
    
            entropy_loss = entropy.mean()
            loss = pg_loss - config["ENT_COEF"] * entropy_loss + v_loss * config["VF_COEF"]
            return loss, (pg_loss, v_loss, entropy_loss)

    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(a2c_loss, has_aux=True),
        l2_reg=0.0,
        value_func_has_aux=True,
        value_func_has_rng=False,
        use_adaptive_learning_rate=False,
        learning_rate_schedule=linear_schedule if config["ANNEAL_LR"] else lambda s: config["LEARNING_RATE"],
        momentum_schedule=lambda step: 0.0,
        damping_schedule=lambda step: 0.001,
        num_burnin_steps=0,
        norm_constraint=config["MAX_GRAD_NORM"],
        inverse_update_period=1,
        multi_device=False
    )

    def make_task_fns(env):
        """Rollout + GAE and the A2C update for one task's training env."""

        def reset_envs(rng):
            obs, state = jax.vmap(env.reset)(rng)
            return jax.vmap(prepare_obs)(obs), state

        def step_envs(state, action):
            obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
            done = jnp.logical_or(terminated, truncated)
            return jax.vmap(prepare_obs)(obs), state, reward, done, info

        @jax.jit
        def rollout(params, env_state, obs, last_done, key):
            """num_steps per env, then GAE computed backwards through the rollout."""
            def rollout_step(c, _):
                env_state, last_obs, last_done, key = c
                action, logprob, value, key = get_action_and_value(params, last_obs, key)
                next_obs, next_state, reward, next_done, info = step_envs(env_state, action)
                storage = Storage(
                    obs=last_obs,
                    actions=action,
                    logprobs=logprob,
                    dones=last_done,
                    values=value,
                    rewards=reward,
                    returns=jnp.zeros_like(reward),
                    advantages=jnp.zeros_like(reward),
                )
                return (next_state, next_obs, next_done, key), (storage, info)

            (env_state, next_obs, next_done, key), (storage, infos) = jax.lax.scan(
                rollout_step, (env_state, obs, last_done, key), None, length=num_steps
            )

            next_value = critic.apply(params.critic_params, network.apply(params.network_params, next_obs)).squeeze(1)
            dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0).astype(jnp.float32)
            values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)

            def gae_step(advantages, inp):
                nextdone, nextvalues, curvalues, reward = inp
                nextnonterminal = 1.0 - nextdone
                delta = reward + config["GAMMA"] * nextvalues * nextnonterminal - curvalues
                advantages = delta + config["GAMMA"] * config["GAE_LAMBDA"] * nextnonterminal * advantages
                return advantages, advantages

            _, advantages = jax.lax.scan(
                gae_step, jnp.zeros((num_envs,)), (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
            )
            storage = storage.replace(advantages=advantages, returns=advantages + storage.values)
            flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), storage)
            batch = (flat.obs, flat.actions, flat.advantages, flat.returns)
            return (env_state, next_obs, next_done, key), batch, infos

        def train_chunk(carry, params, opt_state, global_step):
            infos_list, losses_list, lr_list = [], [], []
            
            for _ in range(scan_steps):
                carry, batch, infos = rollout(params, carry[0], carry[1], carry[2], carry[3])
                
                update_key, key = jax.random.split(carry[3])
                # carry: (env_state, next_obs, next_done, key)
                carry = (carry[0], carry[1], carry[2], key)
                
                params, opt_state, stats = optimizer.step(
                    params=params,
                    state=opt_state,
                    rng=update_key,
                    batch=batch
                )
                
                loss = stats.get('loss', jnp.array(0.0))
                pg_loss, v_loss, entropy_loss = stats.get('aux', (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)))
                learning_rate = linear_schedule(opt_state.step_counter) if config["ANNEAL_LR"] else config["LEARNING_RATE"]
                
                infos_list.append(infos)
                losses_list.append((loss, pg_loss, v_loss, entropy_loss))
                lr_list.append(learning_rate)
                global_step += batch_size
    
            stacked_infos = jax.tree.map(lambda *args: jnp.stack(args), *infos_list)
            stacked_losses = jax.tree.map(lambda *args: jnp.stack(args), *losses_list)
            stacked_lrs = jnp.stack(lr_list)
            
            return carry, params, opt_state, global_step, (stacked_infos, stacked_losses, stacked_lrs)

        return jax.jit(reset_envs), rollout, train_chunk


    # --- evaluation on every task seen so far ---------------------------------
    seen_tasks = []          # (task index (1-based), game, train mods, eval mods)
    evaluators = {}          # compiled once per eval env
    crl_state = {"max_performances": {}, "a_j_j": {}}

    def task_key(game, train_mods, eval_mods):
        return f"{game}+tr:{train_mods}+ev:{eval_mods}"

    def save_and_eval(params, task_idx, step_count, is_end_of_task):
        game, train_mods = tasks[task_idx], task_train_mods[task_idx]
        if is_end_of_task and config.get("SAVE_PATH") is not None:
            save_dir = os.path.join(config["SAVE_PATH"], run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_task{task_idx + 1}_{step_count}_{int(time.time())}.cleanrl_model')

            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes(
                    [config, [params.network_params, params.actor_params, params.critic_params]]))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")
        performances = {}
        for eval_task_idx, eval_game, eval_train_mods, eval_mods in seen_tasks:
            eval_key = task_key(eval_game, eval_train_mods, eval_mods)
            if eval_key not in evaluators:
                eval_env = make_env(eval_game, eval_mods, eval=True, **env_kwargs)()
                evaluators[eval_key] = (make_evaluator(eval_env, network, actor, config["EVAL_EPISODES"], prepare_obs),
                                        jaxtari.make(eval_game, mods=eval_mods or None).renderer)
            evaluate, renderer = evaluators[eval_key]

            net_params, act_params = params.network_params, params.actor_params
            episodic_returns, env_states = evaluate(net_params, act_params, seed=config["SEED"] + 42)
            performances[eval_key] = float(np.mean(jax.device_get(episodic_returns)))
            wandb.log({f"{eval_key}/eval/episodic_return": performances[eval_key]}, step=step_count)
            print(f"[acktr_crl] eval {eval_key}: {performances[eval_key]:.2f}")

            # videos only of the current task's envs
            capture_video = config.get("CAPTURE_VIDEO", False) and eval_game == game and eval_train_mods == train_mods
            if capture_video and (is_end_of_task or not config.get("VIDEO_AT_END", True)):
                frames = jnp.transpose(jax.vmap(renderer.render)(env_states), (0, 3, 1, 2))
                wandb.log({f"{eval_key}/eval/video": wandb.Video(np.array(frames), fps=30, format="mp4")}, step=step_count)
                print(f"Video ({eval_key}/eval) logged to wandb with {frames.shape[0]} frames.")

        if not is_end_of_task:
            return performances

        # CRL metrics: a_k_j = performance on task j after training task k
        metrics = {"crl_metrics/Average_Performance": sum(performances.values()) / len(performances)}
        is_current = lambda t: t[1] == game and t[2] == train_mods
        bwt, fm = [], []
        for t in seen_tasks:
            eval_key = task_key(*t[1:])
            if not is_current(t) and eval_key in crl_state["a_j_j"]:
                bwt.append(performances[eval_key] - crl_state["a_j_j"][eval_key])             # backward transfer
                fm.append(crl_state["max_performances"][eval_key] - performances[eval_key])  # forgetting
        if bwt:
            metrics["crl_metrics/Backward_Transfer"] = sum(bwt) / len(bwt)
            metrics["crl_metrics/Forgetting_Measure"] = sum(fm) / len(fm)
        wandb.log(metrics, step=step_count)
        print(f"[acktr_crl] after task {task_idx + 1}: " + " | ".join(f"{k.split('/')[-1]} {v:.2f}" for k, v in metrics.items()))

        # update max performances and lock in the baseline a_j_j of the current task
        for t in seen_tasks:
            eval_key = task_key(*t[1:])
            crl_state["max_performances"][eval_key] = max(crl_state["max_performances"].get(eval_key, -np.inf), performances[eval_key])
            if is_current(t):
                crl_state["a_j_j"][eval_key] = performances[eval_key]
        return {**performances, **{k.split("/")[-1]: v for k, v in metrics.items()}}

    # --- pre-compile K-FAC step once across all tasks -------------------------
    dummy_batch = (
        jnp.zeros((batch_size, *obs_shape)),
        jnp.zeros((batch_size,), dtype=jnp.int32),
        jnp.zeros((batch_size,)),
        jnp.zeros((batch_size,))
    )
    # Initialize once to get a valid opt_state
    opt_state = optimizer.init(agent_params, jax.random.PRNGKey(0), dummy_batch)

    print("[acktr_crl] compiling K-FAC optimizer step...")
    dummy_params = jax.tree.map(lambda x: jnp.array(x, copy=True), agent_params)
    dummy_opt_state = jax.tree.map(lambda x: jnp.array(x, copy=True), opt_state)
    _, _, _ = optimizer.step(
        params=dummy_params, 
        state=dummy_opt_state, 
        rng=jax.random.PRNGKey(0), 
        batch=dummy_batch
    )

    # --- train the tasks in sequence -----------------------------------------
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks * num_tasks)
    rtpt.start()
    run_time = time.perf_counter()
    global_step = 0
    for i, game in enumerate(tasks):
        for eval_mods in task_eval_mods[i]:
            if not any(t[1:] == (game, task_train_mods[i], eval_mods) for t in seen_tasks):
                seen_tasks.append((i + 1, game, task_train_mods[i], eval_mods))
        if task_train_mods[i]:
            print(f"[WARNING] Training on mods {task_train_mods[i]}!")

        # reset the optimizer state (step count & Kronecker Factor EMAs) for every new task
        opt_state = optimizer.init(agent_params, jax.random.PRNGKey(config["SEED"] + i), dummy_batch)

        reset_envs, rollout, train_chunk = make_task_fns(train_envs[i])
        key, reset_key = jax.random.split(key)
        obs, env_state = reset_envs(jax.random.split(reset_key, num_envs))
        carry = (env_state, obs, jnp.zeros(num_envs, dtype=jnp.bool_), key)

        print(f"[acktr_crl] task {i + 1}/{num_tasks}: {game} {task_train_mods[i]} | start compile...")
        compile_start = time.perf_counter()
        # Compile the environment-specific rollout function for this task
        rollout.lower(agent_params, carry[0], carry[1], carry[2], carry[3]).compile()
        print(f"[acktr_crl] compilation time: {time.perf_counter() - compile_start:.2f}s")

        print(f"[acktr_crl] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
        for chunk in range(num_chunks):
            rtpt.step()
            if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
                save_and_eval(agent_params, i, global_step, is_end_of_task=False)

            chunk_start = time.perf_counter()
            carry, agent_params, opt_state, global_step, (infos, (loss, pg_loss, v_loss, entropy_loss), learning_rate) = train_chunk(
                carry, agent_params, opt_state, global_step
            )
            task_prefix = f"{game}+{task_train_mods[i]}"
            # one wandb point per update of the chunk, not only its last one
            update_metrics = {
                "charts/avg_episodic_return": infos["returned_episode_returns"].mean(axis=(1, 2)),
                "charts/avg_episodic_length": infos["returned_episode_lengths"].mean(axis=(1, 2)),
                f"{task_prefix}/charts/avg_episodic_return": infos["returned_episode_returns"].mean(axis=(1, 2)),
                "charts/learning_rate": learning_rate,
                "losses/value_loss": v_loss,
                "losses/policy_loss": pg_loss,
                "losses/entropy": entropy_loss,
                "losses/loss": loss,
            }
            update_metrics = {k: np.asarray(v) for k, v in update_metrics.items()}
            for step_idx in range(scan_steps - 1):
                step = global_step - (scan_steps - 1 - step_idx) * batch_size
                wandb.log({**{k: float(v[step_idx]) for k, v in update_metrics.items()}, "charts/global_step": step}, step=step)
            metrics = {
                **{k: float(v[-1]) for k, v in update_metrics.items()},
                "charts/task": i + 1,
                "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
                "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
                "charts/time": time.perf_counter() - run_time,
                "charts/global_step": global_step,
            }
            wandb.log(metrics, step=global_step)
            print(
                f"[acktr_crl] task {i + 1} chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
                f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
                f"avg_length {metrics['charts/avg_episodic_length']:.2f} | "
                f"loss {metrics['losses/loss']:.4f} | " # TODO: log cl_loss
                f"SPS {metrics['charts/SPS']}"
            )

        eval_metrics = save_and_eval(agent_params, i, global_step, is_end_of_task=True)

    wandb.finish()
    return eval_metrics
