import pandas as pd

from procsimulator.CommunityManager import CommunityManager
from MinimizeCommunityCosts import MinimizeCommunityCosts


class CommunityManagement(CommunityManager):

    def __init__(self, cg, path_steps_minutes, path_steps_after_first):
        """
        This class is a community management strategy that minimizes the community energy costs.
        Taking in consideration the objective functions and constraints, it shifts the consumption of the activities.

        Args:
          cg: Consumption Generator instance (to allow to use its functions)
          path_steps_minutes: path of the resampled consumption profiles (at 1/60Hz)
          path_steps_after_first: path of the consumption profiles after the optimization
        """
        self.cg = cg
        self.path_steps_minutes = path_steps_minutes
        self.path_steps_after_first = path_steps_after_first
        self.dataframes = {}


    def execute(self, export_prices_hour = [0]*24, import_prices_hour = [0]*24, save_to_file=False):
        """
        Executes the optimization process (implemented strategy using Multiple Knapsack):
        1) First step
        - Prepares the input (arrays) for the process (bin_capacities, bin_maximums, timeslots_number, flexibitilies, items_max, etc)
        - Calls the Knapsack class with the input processed (execute_knapsack function)
        - Updates the consumption profiles based on the output of the knapsack
        2) Second Step
        - Prepares the input (arrays) for the process (bin_capacities, bin_maximums, timeslots_number, flexibitilies, items_max, etc)
        - Calls the Knapsack class with the input processed (execute_knapsack function)
        - Updates the consumption profiles based on the output of the knapsack
        """

        print("Optimization of the community using the implemented strategy")


        fact = 60
        inputs = self.prepare_inputs(fact, save_to_file=save_to_file, export_prices_hour=export_prices_hour, import_prices_hour=import_prices_hour)


        # EVs inputs from EVs simulator
        evs_data = {}
        evs_data['EVs_Inputs'] = pd.read_csv('inputs/EVs_Inputs.csv')
        evs_data['availability'] = pd.read_csv('inputs/alpha.csv')
        evs_data['is_traveling'] = pd.read_csv('inputs/S.csv')


        initial_soc = evs_data['EVs_Inputs'].to_numpy()[:,0] * 1000
        evs_min = evs_data['EVs_Inputs'].to_numpy()[:,1] * 1000
        evs_max = evs_data['EVs_Inputs'].to_numpy()[:,2] * 1000
        evs_trip = evs_data['EVs_Inputs'].to_numpy()[:,3] * 1000
        evs_availability = evs_data['availability'].to_numpy()
        is_traveling = evs_data['is_traveling'].to_numpy()
        efficiency = 0.97
        p_charger = 7200
        #p_charger = contracted_power*0.95
        p_grid_max = 10000
        degradation_cost = 0.08
        num_evs = 2
        num_ess = 1
        community = self.cg.get_community()
        self.production_baseload = 0.85 * float(inputs.contracted_power)


        exec = MinimizeCommunityCosts(inputs.dates, inputs.items, inputs.bins_capacities, inputs.timeslot_numbers, inputs.bins_maximum, inputs.items_max, self.production_baseload, fact, inputs.n_bins_per_hour, inputs.flexibilities, inputs.bins_export_prices, inputs.bins_import_prices, num_evs, evs_max, evs_min, evs_trip, initial_soc, evs_availability, is_traveling, efficiency, p_charger, degradation_cost, p_grid_max, num_ess, inputs.s_max, inputs.s_min, inputs.s_initial_soc, inputs.num_houses, inputs.houses_production, inputs.house_items, inputs.house_items_max, inputs.house_items_date, inputs.house_items_num, inputs.house_items_flex, inputs.house_s_soc, inputs.house_s_max, inputs.house_s_min)
        otimization = exec.execute_knapsack()
        self.dataframes = exec.dataframes

        # Remove all the consumption (all timeslots - placed and not placed ones)
        # Add the consumption of the placed timeslots (just the ones that were placed by the optimization process)
        self.placed_timeslots = otimization[1]
        self.not_placed_timeslots = otimization[2]

        updt = self.create_profiles_after_strategy(self.placed_timeslots, self.timeslots, self.path_steps_minutes, self.path_steps_after_first, self.path_steps_minutes.split("/")[-1], self.path_steps_after_first.split("/")[-1], True, inputs.n_bins_per_hour, fact)