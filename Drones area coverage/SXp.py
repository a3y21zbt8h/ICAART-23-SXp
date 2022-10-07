
from copy import deepcopy
import csv

#  Compute SXp's scores after simulating n randomly-generated scenarios
#  Input: HE/P/FE-scenarios rewards (float list), environment (DroneCoverageAreaEnv), agents (Agent list), the number of
#  steps to look forward (int), number of scenarios to simulate (int), model (DQN), type of transition (String),
#  device (String), the way of computing SXp score (boolean), min/max rewards (float list) and timestep from which the
#  HE/FE-scenario's reward is extracted (int)
#  Output: HE/P/FE-scores (float)
def computeMetric(xp_reward, env, agents, k, number_scenarios, model, move, device, concise, extremum_reward, step_limit_HE=0, step_limit_FE=0):
    he_reward = xp_reward[0]
    p_reward = xp_reward[1]
    fe_reward = xp_reward[2]
    # Store rewards of n scenarios
    rewards = []
    rewards_he = []
    rewards_fe = []

    #  Simulate the n scenarios
    for i in range(number_scenarios):
        r, he_r, fe_r = scenario(env, agents, k, model, device, concise, move)
        rewards.append(r)
        if concise:
            rewards_he.append(he_r)
            rewards_fe.append(fe_r)

    #  Decide which rewards to use for comparison
    if concise:
        rs_H = rewards_he if rewards_he and len(rewards_he) == len(rewards) else rewards
        rs_F = rewards_fe if rewards_fe and len(rewards_fe) == len(rewards) else rewards
    else:
        rs_H = rewards
        rs_F = rewards

    print("Reward P_scenario scenario : {}".format(p_reward))
    print("Step limit : {}, Reward HE-scenario : {}".format(step_limit_HE, he_reward))
    print("Step limit : {}, Reward FE-scenario : {}".format(step_limit_FE, fe_reward))

    #  Compute the scores
    he_score, p_score, fe_score = metrics(he_reward, fe_reward, p_reward, rs_H, rs_F, rewards, number_scenarios, extremum_reward)
    return he_score, p_score, fe_score

#  Compute HE/P/FE-scores
#  Input: HE/FE/P rewards (float), lists of rewards from the randomly-generated scenarios (float list),
#  number of produced scenarios (int) and min/max reward (float)
#  Output: HE/P/FE-scores (float)
def metrics(he_reward, fe_reward, p_reward, scenar_h_rewards, scenar_f_rewards, scenar_last_rewards, number_scenarios, extremum_reward):
    #  Used for cardinality
    he_l = [he_reward <= scenar_h_rewards[i] for i in range(number_scenarios)]
    fe_l = [fe_reward >= scenar_f_rewards[i] for i in range(number_scenarios)]
    #  Average normalized reward of n randomly-generated scenarios
    mean_rewards = normalizeReward(sum(scenar_last_rewards) / len(scenar_last_rewards), extremum_reward[0],
                                   extremum_reward[1])
    # Compute HE/P/FE-scores
    return he_l.count(1) / number_scenarios, abs(p_reward - mean_rewards), fe_l.count(1) / number_scenarios


#  Compute a scenario based on the already learnt model
#  Input: environment (DroneCoverageAreaEnv), agents (Agent list), the number of steps to look forward (int),
#  model (DQN), device (String), the way of computing SXp score (boolean) and the type of transition (String)
#  Output: cumulative rewards for computing HE/P/FE-scores (float)
def scenario(env, agents, k, model, device, concise, move="all"):
    sum_r_he = None
    sum_r_fe = None
    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)
    agents_copy = deepcopy(agents)
    for agent in agents_copy:
        agent.env = env_copy

    #  Sequence loop
    for i in range(k):
        actions = []
        #  Choose actions
        for agent in agents_copy:
            actions.append(agent.chooseAction(model, device=device))
        #  Step
        _, _, _, dones, _ = env_copy.step(agents_copy, actions, move=move)
        #  Rewards
        rewards = env_copy.getReward(agents_copy, actions, dones, reward_type="B")

        #  Save best/worst reward encountered during the scenario if mm_reward
        if concise:
            #  Update sum_r_he and sum_r_fe
            if sum_r_he is None or sum_r_he > sum(rewards):
                sum_r_he = sum(rewards)
            if sum_r_fe is None or sum_r_fe < sum(rewards):
                sum_r_fe = sum(rewards)
        #  Check end condition
        if dones.count(True) == len(dones) and i != k - 1:
            #  Save best/worst reward encountered during the scenario if mm_reward
            if concise:
                if sum_r_he is None or sum_r_he > sum(rewards):
                    sum_r_he = sum(rewards)
                if sum_r_fe is None or sum_r_fe < sum(rewards):
                    sum_r_fe = sum(rewards)
            break

    if sum_r_fe is None and sum_r_he is None:
        return sum(rewards), sum(rewards), sum(rewards)
    else:
        return sum(rewards), sum_r_he, sum_r_fe

#  Compute a P-scenario
#  Input: environment (DroneCoverageArea), agents (Agent list), the number of steps to look forward (int),
#  the model (DQN), min/max rewards (float list), device (String), type of transition (String)
#  and use of render() (boolean)
#  Output: normalized reward of P-scenario
def P_scenario(env, agents, k, model, extremum_reward, device, move="all", render=False):
    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)
    agents_copy = deepcopy(agents)
    for agent in agents_copy:
        agent.env = env_copy
    #  Sequence loop
    for i in range(k):
        actions = []
        #  Choose actions
        for agent in agents_copy:
            actions.append(agent.chooseAction(model, device=device))
        #  Get the most probable transition
        if env.windless:
            _, _, _, dones, _ = env_copy.step(agents_copy, actions, move=move)
        else:
            _, _, _, dones, _ = env_copy.step(agents_copy, actions, most_probable_transitions=True, move=move)
        #  Extract rewards
        rewards = env_copy.getReward(agents_copy, actions, dones, reward_type="B")
        #  Render
        if render:
            env_copy.render(agents_copy)
        #  Check end condition
        if dones.count(True) == len(dones) and i != k - 1:
            break

    print("Cumulative normailzed reward {}".format(normalizeReward(sum(rewards), extremum_reward[0], extremum_reward[1])))
    return normalizeReward(sum(rewards), extremum_reward[0], extremum_reward[1])


#  Normalize a reward between 0 and 1
#  Input: reward to normalize (float) and min, max reachable reward (float)
#  Output: normalized reward (float)
def normalizeReward(reward, mini=-12, maxi=12):
    return ((reward - mini) / (maxi - mini))

#  Compute a HE/FE-scenario, depending on the environment type
#  Input: environment (DroneCoverageAreaEnv), agents (Agent list), wind agents (WindAgent list), the number of steps to
#  look forward (int), agent's model and wind agent's model (DQN), device (String), the way of computing SXp score
#  (boolean), the type of transition (String) and use of render() (boolean)
#  Output: HE/FE-scenario's reward (float) and the step from where the reward was reached (int)
def E_scenario(env, agents, wind_agent, k, model, wind_model, device, concise, move="all", render=False):
    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)
    agents_copy = deepcopy(agents)
    for agent in agents_copy:
        agent.env = env_copy

    if concise:
        best_reward = None
        envs_agents = []
        n = 0
    #  Sequence loop
    for i in range(k):
        #  Choose actions
        actions = []
        for agent in agents_copy:

            action = agent.chooseAction(model, device=device)
            actions.append(action)
        #  Step using wind model as transition
        _, _, _, dones, _ = env_copy.step(agents_copy, actions, wind_agent=wind_agent,
                                          wind_net=wind_model, device=device, move=move)
        #  Extract reward
        rewards = env_copy.getReward(agents_copy, actions, dones, reward_type="B")
        sum_reward = sum(rewards)
        #  Render
        if concise:
            if best_reward is not None:
                if wind_agent.behaviour == "hostile":
                    condition = best_reward > sum_reward
                else:
                    condition = best_reward < sum_reward
            #  Unroll configurations depending on condition
            if best_reward is None or condition:
                best_reward = sum_reward
                #  Render
                if envs_agents:
                    if render:
                        for env, agents in envs_agents:
                            env.render(agents)
                            pass
                    envs_agents = []

                if render:
                    env_copy.render(agents_copy)
                    #print("Dones True : {}".format(dones.count(True)))
                    #print("Rewards : {} \n --> {} ".format(rewards, sum_reward))
                n = i + 1
            else:
                #  Store the environment
                envs_agents.append((deepcopy(env_copy), deepcopy(agents_copy)))
        else:
            if render:
                env_copy.render(agents_copy)
                #print("Dones True : {}".format(dones.count(True)))
                #print("Rewards : {} \n --> {} ".format(rewards, sum_reward))
        #  Check end condition
        if dones.count(True) == len(dones) and i != k - 1:
            #print("H/F-E-scenario : Help, a drone is crashed")
            break

    if concise:
        return best_reward, n
    else:
        return sum_reward, i+1


#  Compute SXps from a specific states and verify how good they are with scores and store them in a CSV file
#  Input: environment (DroneCoverageAreaEnv), agents (Agent list), wind agents (WindAgent list), the number of steps to
#  look forward (int), agent's model and wind agent's models (DQN), device (String), type of transition (String),
#  number of scenarios randomly-generated (int), min/max rewards (float list), csv to store scores (String),
#  use of render() (boolean) and the way of computing SXp score (boolean)
#  Output: None
def SXpMetric(env, agents, wind_agents, k, model, wind_models, device, move="all", number_scenarios=1000, extremum_reward=[], csv_filename="", render=False, concise=False):
    #  ------------------------ HE-scenario ----------------------------------------
    print("HE-scenario -------------")
    reward_HE, step_limit_HE = E_scenario(env, agents, wind_agents[0], k, model, wind_models[0], device, concise, move=move, render=render)
    print("Reward obtained : {}, with step : {}".format(reward_HE, step_limit_HE))

    #  ------------------------ P-scenario ----------------------------------------
    print("P-scenario -------------")
    reward_P = P_scenario(env, agents, k, model, extremum_reward, device, move=move, render=render)
    print("Reward obtained : {}".format(reward_P))

    #  ------------------------ FE-scenario ----------------------------------------
    print("FE-scenario -------------")
    reward_FE, step_limit_FE = E_scenario(env, agents, wind_agents[1], k, model, wind_models[1], device, concise, move=move, render=render)
    print("Reward obtained : {}, with step : {}".format(reward_FE, step_limit_FE))

    #  ------------------------ Metrics F/H/PXp ----------------------------------------
    HE_score, P_score, FE_score = computeMetric([reward_HE, reward_P, reward_FE], env, agents, k, number_scenarios, model, move, device, concise, extremum_reward, step_limit_HE, step_limit_FE)
    print("For HE-scenario, percentage of worse scenarios over {} scenarios : {}".format(number_scenarios, HE_score))
    print("Cumulative reward difference between P-scenario and the mean reward of  {} scenarios : {}".format(number_scenarios, P_score))
    print("For FE-scenario, percentage of better scenarios over {} scenarios : {}".format(number_scenarios, FE_score))

    #  ------------------------ Store in CSV ----------------------------------------
    if csv_filename:
        with open(csv_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([P_score, HE_score, FE_score])

    return

#  Provide different SXps from a starting state depending on user's choice
#  Input: environment (DroneCoverageAreaEnv), agents (Agent list), wind agents (WindAgent list), the number of steps to
#  look forward (int), agent's model and wind agent's models (DQN), device (String), type of transition (String), number
#  of scenarios randomly-generated (int), min/max rewards (float list), use of render() (boolean)
#  Output: None
def SXp(env, agents, wind_agents, k, model, wind_models, device, move="all", number_scenarios=1000, extremum_reward=[], render=True):
    answer = False
    concise = False
    good_answers = ["yes","y"]
    while not answer:

        question = "Do you want an explanation?"
        explanation = input(question)

        if explanation in good_answers:

            #  ------------------------ HE-scenario ----------------------------------------
            question_HE = "Do you want a HE-scenario of the agent's move? "
            explanation_HE = input(question_HE)

            # Provide HE-scenario
            if explanation_HE in good_answers:

                question_boosted = "Do you want a mm_value version of HE-scenario?"
                boosted_case = input(question_boosted)
                mm_reward_HE = boosted_case in good_answers
                print("------------- HE-scenario -------------")
                reward_HE, step_limit_HE = E_scenario(env, agents, wind_agents[0], k, model, wind_models[0], device, mm_reward_HE, move=move, render=render)

            #  ------------------------ P-scenario ----------------------------------------
            question_P = "Do you want a P-scenario of the agent's move? "
            explanation_P = input(question_P)

            # Provide P-scenario
            if explanation_P in good_answers:
                print("------------- P-scenario -------------")
                reward_P = P_scenario(env, agents, k, model, extremum_reward, device, move=move, render=render)

            #  ------------------------ FE-scenario ----------------------------------------
            question_FE = "Do you want a FE-scenario of the agent's move? "
            explanation_FE = input(question_FE)

            # Provide FE-scenario
            if explanation_FE in good_answers:

                question_boosted = "Do you want a mm_value version of FE-scenario?"
                boosted_case = input(question_boosted)
                mm_reward_FE = boosted_case in good_answers

                print("------------- FE-scenario -------------")
                reward_FE, step_limit_FE = E_scenario(env, agents, wind_agents[1], k, model, wind_models[1], device, mm_reward_FE, move=move, render=render)

            #  ------------------------ Metrics ----------------------------------------
            if explanation_HE in good_answers and explanation_P in good_answers and explanation_FE in good_answers:
                question_metric = "Do you want a metric score for these explanations ?"
                answer_metric = input(question_metric)

                if answer_metric in good_answers:
                    HE_score, P_score, FE_score = computeMetric([reward_HE, reward_P, reward_FE], env, agents, k, number_scenarios, model, move, device, concise, extremum_reward, step_limit_HE, step_limit_FE)
                    print("For HE-scenario, percentage of worse scenarios over {} scenarios : {}".format(number_scenarios, HE_score))
                    print("Cumulative reward difference between P-scenario and the mean reward of  {} scenarios : {}".format(number_scenarios, P_score))
                    print("For FE-scenario, percentage of better scenarios over {} scenarios : {}".format(number_scenarios, FE_score))


            print("Go back to the current state of the problem!")
            env.render(agents)

        else:
            pass

        answer = True

    return

