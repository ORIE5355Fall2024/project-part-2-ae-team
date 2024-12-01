import random
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        
        self.project_part = params['project_part'] 

        ### starting remaining inventory and inventory replenish rate are provided
        ## every time the inventory is replenished, it is set to the inventory limit
        ## the inventory_replenish rate is how often the inventory is replenished
        ## for example, we will run with inventory_replenish = 20, with the limit of 11. Then, the inventory will be replenished every 20 time steps (time steps 0, 20, 40, ...) and the inventory will be set to 11 at those time steps. 
        self.remaining_inventory = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']

        ### useful if you want to use a more complex price prediction model
        ### note that you will need to change the name of the path and this agent file when submitting
        ### complications: pickle works with any machine learning models defined in sklearn, xgboost, etc.
        ### however, this does not work with custom defined classes, due to the way pickle serializes objects
        ### refer to './yourteamname/create_model.ipynb' for a quick tutorial on how to use pickle
        self.filename = './agents/ae-team/trained_model'
        self.trained_model = pickle.load(open(self.filename, 'rb'))
        #self.scaler = pickle.load(open('./agents/ae-team/trained_model', 'rb'))

        ### potentially useful for Part 2 -- When competition is between two agents
        ### and you want to keep track of the opponent's status
        # self.opponent_number = 1 - agent_number  # index for opponent
        self.time_step = 0
        self.pricing_history = []

        self.competitor_pricing_history = []

    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        '''
        This function updates your internal state based on the last sale that occurred.
        This template shows you several ways you can keep track of important metrics.
        '''
        ### keep track of who, if anyone, the customer bought from
        #did_customer_buy_from_me = (last_sale[0] == self.this_agent_number)
        ### potentially useful for Part 2
        # did_customer_buy_from_opponent = (last_sale[0] == self.opponent_number)

        ### keep track of the prices that were offered in the last sale
        #my_last_prices = last_sale[1][self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_last_prices = last_sale[1][self.opponent_number]

        ### keep track of the profit for this agent after the last sale
        #my_current_profit = state[self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_current_profit = state[self.opponent_number]

        ### keep track of the inventory levels after the last sale
        #self.remaining_inventory = inventories[self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_inventory = inventories[self.opponent_number]

        ### keep track of the time until the next replenishment
        #time_until_replenish = time_until_replenish

        ### TODO - add your code here to potentially update your pricing strategy 
        ### based on what happened in the last round
        self.remaining_inventory = inventories[self.this_agent_number]
        self.time_step += 1

        # Record competitor's last price
        if self.project_part == 2 and last_sale[1] is not None:
            competitor_index = 1 - self.this_agent_number
            competitor_price = last_sale[1][competitor_index]
            self.competitor_pricing_history.append(competitor_price)
    
    def predict_competitor_price(self):
        if self.competitor_pricing_history:
            # Simple heuristic: use the average of competitor's past prices
            return np.mean(self.competitor_pricing_history[-5:])
        else:
            # Default to a reasonable price if no history
            return 50
    
    # def adjust_price_for_inventory(self, price):
    #     """
    #     Adjust price to account for inventory constraints.
    #     """
    #     remaining_customers = 20 - self.time_step % 20
    #     if remaining_customers <= self.remaining_inventory:
    #         return price
    #     # increase price to ration inventory
    #     return price * (1 + 0.2 * (1 - self.remaining_inventory / remaining_customers))

    def action(self, obs):
        '''
        This function is called every time the agent needs to choose an action by the environment.

        The input 'obs' is a 5 tuple, containing the following information:
        -- new_buyer_covariates: a vector of length 3, containing the covariates of the new buyer.
        -- last_sale: a tuple of length 2. The first element is the index of the agent that made the last sale, if it is NaN, then the customer did not make a purchase. The second element is a numpy array of length n_agents, containing the prices that were offered by each agent in the last sale.
        -- state: a vector of length n_agents, containing the current profit of each agent.
        -- inventories: a vector of length n_agents, containing the current inventory level of each agent.
        -- time_until_replenish: an integer indicating the time until the next replenishment, by which time your (and your opponent's, in part 2) remaining inventory will be reset to the inventory limit.

        The expected output is a single number, indicating the price that you would post for the new buyer.
        '''
        # unpack observation
        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        # if inventory is empty return very high price
        if self.remaining_inventory == 0:
            return np.inf

        # define price range
        price_range = np.linspace(10, 100, 50)

        # scale new buyer's covariates
        #scaled_covariates = self.scaler.transform([new_buyer_covariates])

        # Vectorize features for all prices
        n_prices = len(price_range)
        features_with_price = np.column_stack((
            np.repeat(new_buyer_covariates[0], n_prices),
            np.repeat(new_buyer_covariates[1], n_prices),
            np.repeat(new_buyer_covariates[2], n_prices),
            price_range
        ))

        # Predict purchase probabilities for all prices at once
        purchase_probs = self.trained_model.predict_proba(features_with_price)[:, 1]

        # Calculate expected revenues
        revenues = price_range * purchase_probs

        # Find the price with the maximum expected revenue without competition
        max_idx = np.argmax(revenues)
        best_price = price_range[max_idx]

        # Adjust price based on competitor's expected price
        if self.project_part == 2:
            competitor_price = self.predict_competitor_price()
            # If competitor prices lower, consider lowering price slightly below
            if competitor_price < best_price:
                best_price = max(competitor_price - 0.01, 10)  # Ensure price doesn't go below 10
            else:
                # If competitor prices higher, we can maintain or increase our price
                pass  # Keep best_price as is

        # Adjust price based on inventory constraints
        final_price = self.adjust_price_for_inventory(best_price)

        self.pricing_history.append(final_price)
        return final_price
    

        # # compute expected revenue for each price
        # best_price = price_range[0]
        # max_revenue = 0

        # for price in price_range:
        #     # predict purchase probability for given price
        #     #features = np.append(new_buyer_covariates, [[price]], axis=1)
        #     # append price 
        #     features_with_price = np.array([new_buyer_covariates[0], 
        #                                     new_buyer_covariates[1], 
        #                                     new_buyer_covariates[2], 
        #                                     price]).reshape(1, -1)

        #     # column names for features dataframe
        #     feature_names = ['Covariate1', 'Covariate2', 'Covariate3', 'price_item']

        #     # create features dataframe
        #     features_df = pd.DataFrame(features_with_price, columns=feature_names)

        #     purchase_prob = self.trained_model.predict_proba(features_df)[0, 1]

        #     # calcualte expected revenue
        #     revenue = price * purchase_prob

        #     # update best price if revenue is higher
        #     if revenue > max_revenue:
        #         max_revenue = revenue
        #         best_price = price

        # final_price = self.adjust_price_for_inventory(best_price)

        # # log pricing for history and return final price
        # self.pricing_history.append(final_price)
        # return final_price

        ### currently output is just a deterministic price for the item
        ### but you are expected to use the new_buyer_covariates
        ### combined with models you come up with using the training data 
        ### and history of prices from each team to set a better price for the item
        # return 30.123 #112.358

