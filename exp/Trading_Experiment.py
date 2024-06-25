import os
from openai import OpenAI
from dotenv import load_dotenv
import json

import autogen
import autogen.trace as trace
from autogen.trace.optimizers import FunctionOptimizerV2Memory


p1_name = trace.node('Alice', trainable=False)
p2_name = trace.node('Bob', trainable=False)
p1_prompt = trace.node('STOCKPILE THESE RESOURCES: N/A', trainable=True)
p2_prompt = trace.node('STOCKPILE THESE RESOURCES: N/A', trainable=True)

system_prompt = "RULES of the TRADING GAME between two players named "+p1_name.data+" and "+p2_name.data+".\n\n"
system_prompt += "Each player's inventory is private and consists of three resources, WOOD, STONE, and GOLD.\n"
system_prompt += "The higher the quantity of a resource a player has, the higher the value of that resource.\n"
system_prompt += "The value of a resource is determined by a scale that increases exponentially with quantity.\n"
system_prompt += "The goal of the game is to maximize the total value of the resources in all players' inventories (OVERALL SCORE).\n\n"
system_prompt += "The game is played in turns, with each player taking one action per turn.\n"
system_prompt += "Turns alternate between the two players, starting with "+p1_name.data+".\n"
system_prompt += "Trading is the only way to exchange resources between players.\n"
system_prompt += "Players can choose to end the game when they think subsequent trades will be rejected or not beneficial.\n\n"
system_prompt += "Each player can do one of 4 actions: propose a trade, accept a trade, reject a trade or end the game.\n"
system_prompt += "Propose a trade: A proposed trade must barter the same quantity of one resource for another.\n"
system_prompt += "A player can only propose a trade if they have sufficient quantity of the resource they are trading away.\n"
system_prompt += "Accept a trade: A player can only accept a trade if they have sufficient quantity of the resource that they in turn are trading away.\n"
system_prompt += "Reject a trade: A player can reject a trade if they do not have sufficient quantity of the resource, or if it would lead to lower OVERALL SCORE.\n"
system_prompt += "End game: If both players select the end game action, the game ends and the overall value of both players' inventories are tallied up to produce the OVERALL SCORE.\n"
system_prompt += "NOTE: BOTH players must select the end game action during their respective turns for the game to end.\n\n"
system_prompt += "Each of the four actions must be formatted as a valid json object that can be parsed by python json.loads:\n"
system_prompt += "Example of proposing a trade = {'action': 'TRADE', 'sell_resource': 'WOOD', 'buy_resource': 'STONE', 'quantity': 5} \n"
system_prompt += "Example of accepting a trade = {'action': 'ACCEPT'} \n"
system_prompt += "Example of rejecting a trade = {'action': 'REJECT'} \n"
system_prompt += "Example of ending the game = {'action': 'END'} \n\n"


# GAMESTATE VARIABLES
p1_inventory = {'WOOD': 4, 'STONE': 3, 'GOLD': 2}
p2_inventory = {'WOOD': 1, 'STONE': 5, 'GOLD': 2}
proposed_trade = None
proposed_end = False
conversation = []

load_dotenv(".env")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@trace.bundle(trainable=False)
def create_message(player, prompt, previous_message=None):
    """
    Formats the conversation history so that an LLM can act on behalf of the player using a specified strategy
    """

    global p1_inventory
    global p2_inventory

    player_prompt = f'In the trading game, you are named {player}.\n'
    messages = [{'role': 'system', 'content': player_prompt}, {'role': 'system', 'content': prompt}]
    
    current_inventory = p1_inventory if player == "Alice" else p2_inventory    # p1_name.data hardcoded as Alice
    inventory_message = f'Your inventory consists of {current_inventory["WOOD"]} WOOD, {current_inventory["STONE"]} STONE, and {current_inventory["GOLD"]} GOLD.'
    messages.append({'role': 'user', 'content': inventory_message})

    return messages


def parse(player, response_json):
    global p1_inventory
    global p2_inventory

    sell_resource = response_json['sell_resource']
    buy_resource = response_json['buy_resource']
    quantity = response_json['quantity']
    if player == "Alice":   # p1_name.data hardcoded as Alice
        if p1_inventory[sell_resource] < quantity:
            return None
        if p2_inventory[buy_resource] < quantity:
            return None
        return {"Alice": {sell_resource: -quantity, buy_resource: quantity}, 
                "Bob": {sell_resource: quantity, buy_resource: -quantity}}
    else:   # p2_name.data hardcoded as Bob
        if p2_inventory[sell_resource] < quantity:
            return None
        if p1_inventory[buy_resource] < quantity:
            return None
        return {"Alice": {sell_resource: quantity, buy_resource: -quantity}, 
                "Bob": {sell_resource: -quantity, buy_resource: quantity}}


def accept_trade():
    global proposed_trade
    global p1_inventory
    global p2_inventory

    current_dict = proposed_trade["Alice"]  # p1_name.data hardcoded as Alice
    for key in current_dict:
        p1_inventory[key] += current_dict[key]
    
    current_dict = proposed_trade["Bob"] # p2_name.data hardcoded as Bob
    for key in current_dict:
        p2_inventory[key] += current_dict[key]


@trace.bundle(trainable=False)
def chat(player, message):
    """
    Calls the OpenAI LLM API to produce a response for the given player and input message
    """

    global system_prompt
    global conversation
    global proposed_trade
    global proposed_end
    
    current_message = [{'role': 'system', 'content': system_prompt}] + message

    if len(conversation) > 0:
        current_message.append({'role': 'user', 'content': 'This is the transcript of the conversation so far.'})
        conversation_history = ""
        for i in conversation:
            conversation_history += f'{i["role"]} said: {i["content"]}\n'
        current_message.append({'role': 'user', 'content': conversation_history})

    chat = client.chat.completions.create(
            model='gpt-4-0125-preview',
            messages=current_message,
            temperature=0,
            max_tokens=200,
            seed=42,
            response_format={ "type": "json_object" }
        )
    
    response = chat.choices[0].message.content
    response_json = json.loads(response)
    
    action = response_json['action']
    
    if action == 'END':
        if proposed_end:
            return 'TERMINATE'
        else:
            proposed_end = True
    elif action == 'REJECT':
        proposed_trade = None
        if proposed_end:
            proposed_end = False
    elif action == 'ACCEPT':
        if proposed_trade is not None:
            accept_trade()
        elif proposed_end:
            return 'TERMINATE'
    elif action == 'TRADE':
        proposed_trade = parse(player,response_json)
        if proposed_end:
            proposed_end = False
    
    return response


def end_game():
    global p1_inventory
    global p2_inventory
    
    value_scale = [1, 2, 4, 7, 12, 20, 33, 54, 88, 143, 250]

    p1_value = 0
    if p1_inventory['WOOD'] > 0:
        p1_value += value_scale[p1_inventory['WOOD']-1 if p1_inventory['WOOD'] <= 11 else 10]
    if p1_inventory['STONE'] > 0:
        p1_value += value_scale[p1_inventory['STONE']-1 if p1_inventory['STONE'] <= 11 else 10]
    if p1_inventory['GOLD'] > 0:
        p1_value += value_scale[p1_inventory['GOLD']-1 if p1_inventory['GOLD'] <= 11 else 10]

    p2_value = 0
    if p2_inventory['WOOD'] > 0:
        p2_value += value_scale[p2_inventory['WOOD']-1 if p2_inventory['WOOD'] <= 11 else 10]
    if p2_inventory['STONE'] > 0:
        p2_value += value_scale[p2_inventory['STONE']-1 if p2_inventory['STONE'] <= 11 else 10]
    if p2_inventory['GOLD'] > 0:
        p2_value += value_scale[p2_inventory['GOLD']-1 if p2_inventory['GOLD'] <= 11 else 10]

    return p1_value + p2_value, p1_value, p2_value


optimizer = FunctionOptimizerV2Memory(
                [p1_prompt, p2_prompt], memory_size=5, config_list=autogen.config_list_from_json("OAI_CONFIG_LIST")
            )
#optimizer.objective = (
#                "Suggest prompts to specialize the two players so that the OVERALL SCORE is as large as possible.\n"
#                + optimizer.default_objective
#        )

for i in range(10):
    p1_inventory = {'WOOD': 4, 'STONE': 3, 'GOLD': 2}
    p2_inventory = {'WOOD': 1, 'STONE': 5, 'GOLD': 2}
    proposed_trade = None
    proposed_end = False
    conversation = []

    current_message = None
    current_player = p2_name
    while (current_message is None) or (current_message.data != 'TERMINATE'):
        current_player = p1_name if current_player == p2_name else p2_name
        current_prompt = p1_prompt if current_player == p1_name else p2_prompt
        #print(p1_name, p1_inventory)
        #print(p2_name, p2_inventory)
        #print(current_player, current_prompt)
        message_prompt = create_message(current_player, current_prompt, current_message)
        #print(message_prompt)
        current_message = chat(current_player, message_prompt)
        #print(current_message)
        if current_message.data != 'TERMINATE':
            conversation.append({'role': current_player.data, 'content': current_message.data})
        
    #dot = current_message.backward(visualize=True)
    #dot.render(directory='.', view=True)
    
    result_value, p1_value, p2_value = end_game()
    feedback = 'The game has ended. ' + \
                p1_name.data + f' has inventory with value of {p1_value} and ' + \
                p2_name.data + f' has inventory with value of {p2_value}.\n'
    feedback += 'OVERALL SCORE: ' + str(result_value)
    if result_value < 73:
        feedback += '\nOVERALL SCORE is less than optimal. Find better trades to increase the OVERALL SCORE.'

    print("ITERATION", i+1)
    print(p1_name.data, p1_prompt.data)
    print(p2_name.data, p2_prompt.data)
    print(feedback)

    optimizer.zero_feedback()
    optimizer.backward(current_message, feedback, visualize=False)
    optimizer.step(verbose=False)
    