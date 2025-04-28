#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: run_avalon_battle.py 
# @date: 2024/3/6 17:46 
#
# describe: Full HuggingFace version (TinyLlama for agents + extractors)
#
import os
os.environ["GROQ_API_KEY"] = "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"


import random
import argparse
from sentence_transformers import SentenceTransformer

from src.games.avalon.avalon import Avalon
from src.agents import SAPARAgent, CGAgent
from src.extractor.llm_extractor.hf_extractor import HFExtractor  # ✅ Replaces ChatGPTBasedExtractor
from src.utils import create_dir, read_json
from prompt.avalon_sapar_prompt import summary_prompt, plan_prompt, response_prompt, system_prompt, \
    action_prompt, suggestion_prompt, update_prompt, analysis_prompt, \
    strategy_prompt, candidate_actions, init_strategies, role_introduction, role_target
from prompt.avalon_cg_prompt import rule_role_prompt, select_question_prompt, ask_question_prompt, \
    generate_answer_prompt, reflection_prompt, extract_suggestion_prompt, generate_response_prompt, \
    informativeness_prompt, question_list
from src.games.avalon.extract_demos import number_extract_prompt, player_extractor_demos, vote_extractor_demos, \
    quest_extractor_demos, choose_identify_extractor_demos, select_merlin_extractor_demos, bool_extract_prompt, \
    quest_extract_prompt

# roles = ["Merlin", "Percival", "Loyal Servant", "Loyal Servant", "Morgana", "Assassin"]
roles = ["Merlin", "Percival", "Loyal Servant", "Morgana", "Assassin"]

bert_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1", device="cpu")


def run_game(game_output_dir: str, camp, game_idx):
    create_dir(game_output_dir.format(game_idx))

    mode = 'watch'
    language = 'english'
    # ai_model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    ai_model = "llama3-8b-8192"
    player_nums = 5
    player_mapping = {}
    random.shuffle(roles)

    if camp == "good":
        camp_role = ["Merlin", "Percival", "Loyal Servant"]
    else:
        camp_role = ["Morgana", "Assassin"]

    game = Avalon(player_nums, language, mode, ai_model, game_output_dir.format(game_idx))

    player_args = []
    for i in range(player_nums):
        log_dir = f"{game_output_dir.format(game_idx)}/player {i + 1}"
        create_dir(log_dir)
        name = f'player {i + 1}'
        role = roles[i]

        if role in camp_role:
            if game_idx == 0:
                role_strategy = init_strategies[role]
                other_strategy = "None"
                suggestion = "None"
            else:
                load_file = f"{game_output_dir.format(game_idx - 1)}/{role}_reflection.json"
                experience = read_json(load_file)
                role_strategy = experience.get("strategy", "None")
                other_strategy = experience.get("other_strategy", "None")
                suggestion = experience.get("suggestion", "None")

            role_system_prompt = system_prompt.format(name=name, role=role, strategy=role_strategy,
                                                      suggestion=suggestion, other_strategy=other_strategy)

            player_args.append((
                SAPARAgent,
                {"name": name, "role": role, "role_intro": role_introduction[role.lower()],
                 "game_goal": role_target[role], "strategy": role_strategy,
                 "system_prompt": role_system_prompt, "summary_prompt": summary_prompt,
                 "analysis_prompt": analysis_prompt, "plan_prompt": plan_prompt,
                 "action_prompt": action_prompt, "response_prompt": response_prompt, "model": ai_model,
                 "temperature": 0.3, "api_key": None, "output_dir": log_dir,
                 "suggestion_prompt": suggestion_prompt, "strategy_prompt": strategy_prompt,
                 "update_prompt": update_prompt, "suggestion": suggestion,
                 "other_strategy": other_strategy, "candidate_actions": candidate_actions}
            ))

        else:
            if game_idx == 0:
                previous_exp_pool = []
            else:
                load_file = f"{game_output_dir.format(game_idx - 1)}/{role}_reflection.json"
                previous_exp_pool = read_json(load_file)

            player_args.append((
                CGAgent,
                {"name": name, "role": role, "rule_role_prompt": rule_role_prompt,
                 "select_question_prompt": select_question_prompt,
                 "ask_question_prompt": ask_question_prompt,
                 "generate_answer_prompt": generate_answer_prompt,
                 "reflection_prompt": reflection_prompt,
                 "extract_suggestion_prompt": extract_suggestion_prompt,
                 "generate_response_prompt": generate_response_prompt,
                 "informativeness_prompt": informativeness_prompt,
                 "question_list": question_list.get(role, []), "retrival_model": bert_model,
                 "model": ai_model, "freshness_k": 15, "informativeness_n": 15, "experience_window": 50,
                 "temperature": 0.3, "api_key": "", "previous_exp_pool": previous_exp_pool,
                 "output_dir": log_dir}
            ))

    game.add_players(player_args)

    # ✅ Using HFExtractor instead of ChatGPT
    extractor_args = [
        (HFExtractor, {
            "extractor_name": "player extractor", "model_name": ai_model,
            "extract_prompt": number_extract_prompt, "system_prompt": "You are a helpful assistant.",
            "temperature": 0.0, "few_shot_demos": player_extractor_demos,
            "output_dir": game_output_dir.format(game_idx)
        }),
        (HFExtractor, {
            "extractor_name": "vote extractor", "model_name": ai_model,
            "extract_prompt": bool_extract_prompt, "system_prompt": "You are a helpful assistant.",
            "temperature": 0.0, "few_shot_demos": vote_extractor_demos,
            "output_dir": game_output_dir.format(game_idx)
        }),
        (HFExtractor, {
            "extractor_name": "quest extractor", "model_name": ai_model,
            "extract_prompt": quest_extract_prompt, "system_prompt": "You are a helpful assistant.",
            "temperature": 0.0, "few_shot_demos": quest_extractor_demos,
            "output_dir": game_output_dir.format(game_idx)
        }),
        (HFExtractor, {
            "extractor_name": "identify extractor", "model_name": ai_model,
            "extract_prompt": bool_extract_prompt, "system_prompt": "You are a helpful assistant.",
            "temperature": 0.0, "few_shot_demos": choose_identify_extractor_demos,
            "output_dir": game_output_dir.format(game_idx)
        }),
        (HFExtractor, {
            "extractor_name": "merlin extractor", "model_name": ai_model,
            "extract_prompt": number_extract_prompt, "system_prompt": "You are a helpful assistant.",
            "temperature": 0.0, "few_shot_demos": select_merlin_extractor_demos,
            "output_dir": game_output_dir.format(game_idx)
        }),
    ]

    game.init_extractor(
        player_extractor=extractor_args[0],
        vote_extractor=extractor_args[1],
        quest_extractor=extractor_args[2],
        choose_identify_extractor=extractor_args[3],
        select_merlin_extractor=extractor_args[4]
    )

    game.start()

    for player_i, agent_i in game.players.items():
        if isinstance(agent_i, (SAPARAgent, CGAgent)):
            agent_i.reflection(
                player_mapping,
                file_name=f"{game_output_dir.format(game_idx)}/{player_mapping.get(player_i)}_reflection.json",
                winners=game.winners,
                duration=game.game_round
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_count", type=int, default=10)
    parser.add_argument("--camp", type=str, default="good", choices=["good", "evil"])
    parser.add_argument("--exp_name", type=str, default="battle")
    parser.add_argument("--use_proxy", type=bool, default=False)
    parser.add_argument("--start_game_idx", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    for game_round in range(args.start_game_idx, args.game_count):
        output_dir = f"playing_log/avalon/battle/{args.exp_name}-{args.camp}" + "-game_{}"
        run_game(output_dir, camp=args.camp, game_idx=game_round)
        print(f"✅ Game {game_round} finished!")


if __name__ == '__main__':
    main()
    print("🏁 All done!")
