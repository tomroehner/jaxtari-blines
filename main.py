import hydra
from omegaconf import OmegaConf

from agents.run_utils import code_versions


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {**config, **config.get("alg", {})}
    merged_config.update(code_versions())
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    n_seeds = merged_config.get("NUM_SEEDS", 1)
    starting_seed = merged_config.get("SEED", 0)

    alg = merged_config["ALG"]

    all_metrics = []
    for seed in range(n_seeds):
        if alg == "PPO":
            from agents.ppo.ppo import single_run
        elif alg == "DQN":
            from agents.dqn.dqn import single_run
        elif alg == "PQN":
            from agents.pqn.pqn import single_run
        elif alg == "RAINBOW":
            from agents.rainbow.rainbow import single_run
        elif alg == "C51":
            from agents.c51.c51 import single_run
        elif alg == "SCOBOTS":
            from agents.scobots.scobots import single_run
        elif alg == "RM_DQN":
            from agents.rm_dqn.rm_dqn import single_run
        elif alg == "HRM":
            from agents.hrm.hrm import single_run
        elif alg == "APQN":
            from agents.apqn.apqn import single_run
        elif alg == "DROQ":
            from agents.droq.droq import single_run
        elif alg == "IQN":
            from agents.iqn.iqn import single_run
        elif alg == "SIMBA_PQN":
            from agents.simba_pqn.simba_pqn import single_run
        elif alg == "SIMBA_SAC":
            from agents.simba_sac.simba_sac import single_run
        elif alg == "SAC":
            from agents.sac.sac import single_run
        elif alg == "SAC_CALE":
            from agents.sac_cale.sac import single_run
        elif alg == "TD3":
            from agents.td3.td3 import single_run
        elif alg == "DDPG":
            from agents.ddpg.ddpg import single_run
        elif alg == "CROSSQ":
            from agents.crossq.crossq import single_run
        elif alg == "AGENT57":
            from agents.agent57.agent57 import single_run
        elif alg == "PPO_CRL":
            from agents.ppo_crl.ppo_crl import single_run
        elif alg == "DREAMERV3":
            from agents.dreamerv3.dreamerv3 import single_run
        elif alg == "A2C":
            from agents.a2c.a2c import single_run
        elif alg == "ACKTR":
            from agents.acktr.acktr import single_run
        elif alg == "ACKTR_CRL":
            from agents.acktr_crl.acktr_crl import single_run
        else:
            raise ValueError(f"Unknown ALG: {alg}")

        used_seed = starting_seed + seed
        print(f"Running seed {used_seed} ...")
        merged_config["SEED"] = used_seed
        metrics = single_run(merged_config)
        metrics["ALG"] = alg
        metrics["ENV_ID"] = merged_config["ENV_ID"]
        metrics["PIXEL_BASED"] = merged_config.get("PIXEL_BASED", False)
        metrics["SEED"] = used_seed
        all_metrics.append(metrics)

    print("Metrics: ", all_metrics)


if __name__ == "__main__":
    main()
