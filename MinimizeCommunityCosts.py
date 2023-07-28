import pandas as pd
import datetime
import numpy as np
import pyomo
import pyomo.opt
import pyomo.environ as pyo

class MinimizeCommunityCosts:

    def __init__(self, dates, items, bin_capacities, numbers, bins_maximum, items_maximum, baseload, fact, n_bins_per_hour, flexibilities, export_prices, import_prices, num_evs, evs_max, evs_min, evs_trip, initial_soc, ev_p_charger, evs_availability, evs_travelling, efficiency, p_charger, degradation_cost, p_grid_max, num_ess, s_max, s_min, s_initial_soc, num_houses, houses_production, house_items, house_items_max, house_items_date, house_items_num, house_items_flex, house_s_soc, house_s_max, house_s_min):
        """
        This class receives as input some arrays with information about the timeslots and the bins, as well as some global data (baseload, fact and n_bins_per_hour)

        Args:
          dates: array containing the bin in which the user wants to place the timeslot
          items: array containing the timeslot energy for each item of each timeslot
          bin_capacities: array containing the bin capactity of each bin
          numbers: array containing the number of the timeslot of each item of each timeslot
          bins_maximum: array containing the maximum power of the production of each bin
          items_maximum: array containing the maximum power of each item of each timeslot
          baseload: value that represents the value of energy that can be acquired from the grid in the 2nd step
          fact: minutes of each bin (e.g. if bins of 30 minutes, fact = 30)
          n_bins_per_hour: number of bins per hour (parameter of the strategy) to know the quantity of bins in a day (e.g. if bins of 30 minutes, n_bins_per_hour = 2)
          flexibilities: array contanining the flexibility of each item of each timeslot
          energy_prices: array containing the energy prices of each bin
        """
        self.dates = dates
        self.items = items
        self.numbers = numbers
        self.baseload = baseload
        self.bin_capacities = bin_capacities
        self.bins_maximum = bins_maximum
        self.items_maximum = items_maximum
        self.fact = fact
        self.n_bins_per_hour = n_bins_per_hour
        self.num_bins = n_bins_per_hour * 24
        self.flexibilities = flexibilities
        self.export_prices = export_prices
        self.import_prices = import_prices
        self.num_evs = num_evs
        self.maximum_soc = evs_max
        self.minimum_soc = evs_min
        self.etrip = evs_trip
        self.initial_soc = initial_soc
        self.ev_p_charger = ev_p_charger
        self.availability = evs_availability
        self.travelling = evs_travelling
        self.efficiency = efficiency
        self.p_charger = p_charger
        self.degradation_cost = degradation_cost
        self.p_grid_max = p_grid_max
        self.num_ess = num_ess
        self.s_maximum_soc = s_max
        self.s_minimum_soc = s_min
        self.s_initial_soc = s_initial_soc
        self.num_houses = num_houses
        self.houses_production = houses_production
        self.house_items = house_items
        self.house_items_max = house_items_max
        self.house_items_date = house_items_date
        self.house_items_num = house_items_num
        self.house_items_flex = house_items_flex
        self.house_s_soc = house_s_soc
        self.house_s_max = house_s_max
        self.house_s_min = house_s_min
        self.h_prod = houses_production
        self.dataframes = {}


    def _auxDictionary(self, a):
        temp_dictionary = {}
        if len(a.shape) == 3:
            for dim0 in np.arange(a.shape[0]):
                for dim1 in np.arange(a.shape[1]):
                    for dim2 in np.arange(a.shape[2]):
                        temp_dictionary[(dim0+1, dim1+1, dim2+1)] = a[dim0, dim1, dim2]
        elif len(a.shape) == 2:
            for dim0 in np.arange(a.shape[0]):
                for dim1 in np.arange(a.shape[1]):
                    temp_dictionary[(dim0+1, dim1+1)] = a[dim0, dim1]
        else:
            for dim0 in np.arange(a.shape[0]):
                temp_dictionary[(dim0+1)] = a[dim0]
        return temp_dictionary



    def ext_pyomo_vals(self, vals):
        # make a pd.Series from each
        s = pd.Series(vals.extract_values(),
                      index=vals.extract_values().keys())
        # if the series is multi-indexed we need to unstack it...
        if type(s.index[0]) == tuple:    # it is multi-indexed
            s = s.unstack(level=1)
        else:
            # force transition from Series -> df
            s = pd.DataFrame(s)
        return s


    def convert_to_tuple_dict(self, arr, data):
        # Convert 3-dimentional array to dict
        new_dict = {}
        for i, house in enumerate(arr):
            for j in np.arange(data['max_tims']):
                if (j >= len(house)):
                    for k in np.arange(data['max_len']):
                        new_dict[(i+1, j+1, k+1)] = 0
                else:
                    tim = house[j]
                    for k in np.arange(data['max_len']):
                        if (k < len(tim)):
                            new_dict[(i+1, j+1, k+1)] = tim[k]
                        else:
                            new_dict[(i+1, j+1, k+1)] = 0
        return new_dict


    def create_data_model(self):
        """
        Creates the data model for the Multi Knapsack problem, according to the input received in the constructor

        Returns:
          data model (list with different arrays and values)
        """
        data = {}
        #data['tim_lens'] = [len(i) for i in self.items]
        #data['max_len'] = max(data['tim_lens'])

        data['tim_lens'] = []
        for i in self.house_items:
            tmp = []
            for j in i:
                tmp.append(len(j))
            data['tim_lens'].append(tmp)

        data['max_len'] = max([max(i) if len(i) > 0 else 0 for i in data['tim_lens']]) # Maximum number of timeslots
        data['max_tims'] = max([len(i) for i in data['tim_lens']]) # Maximum number of items
        data['tim_lens'] = pd.DataFrame(data['tim_lens']).fillna(0)
        data['weights'] = pd.DataFrame(self.items).fillna(0)
        data['dates'] = pd.DataFrame(self.dates).fillna(0)
        data['numbers'] = pd.DataFrame(self.numbers).fillna(0)
        data['items'] = list(range(len(self.items)))
        data['num_items'] = len(self.items)
        num_bins = len(self.bin_capacities)
        data['bins'] = list(range(num_bins))
        data['bin_capacities'] = pd.DataFrame(self.bin_capacities)
        data['bins_maximum'] = pd.DataFrame(self.bins_maximum)
        data['items_maximum'] = pd.DataFrame(self.items_maximum).fillna(0)
        data['baseload'] = self.baseload
        data['fact'] = self.fact
        data['n_bins_per_hour'] = self.n_bins_per_hour
        data['num_bins'] = self.num_bins
        data['flexibilities'] = pd.DataFrame(self.flexibilities).fillna(0)
        data['export_prices'] = pd.DataFrame(self.export_prices)
        data['import_prices'] = pd.DataFrame(self.import_prices)
        data['num_evs'] = self.num_evs
        data['maximum_soc'] = self.maximum_soc
        data['minimum_soc'] = self.minimum_soc
        data['ev_trip'] = self.etrip
        data['initial_soc'] = self.initial_soc
        data['ev_p_charger'] = self.ev_p_charger
        data['availability'] = self.availability
        data['travelling'] = self.travelling
        data['efficiency'] = self.efficiency
        data['p_charger'] = self.p_charger
        data['degradation_cost'] = self.degradation_cost
        data['p_grid_max'] = self.p_grid_max
        data['num_ess'] = self.num_ess
        data['s_maximum_soc'] = self.s_maximum_soc
        data['s_minimum_soc'] = self.s_minimum_soc
        data['s_initial_soc'] = self.s_initial_soc
        data['num_houses'] = self.num_houses
        data['tim_st'] = [len(i) for i in self.house_s_soc]
        data['max_st'] = max(data['tim_st'])
        data['houses_production'] = self.houses_production
        data['house_items'] = self.house_items
        data['house_items_max'] = self.house_items_max
        data['house_items_date'] = self.house_items_date
        data['house_items_num'] = self.house_items_num
        data['house_items_flex'] = self.house_items_flex
        data['house_s_soc'] = pd.DataFrame(self.house_s_soc).fillna(0) # Need to use dataframes instead of numpy arrays when the shape is not the same elements (for instance, we have a timeslot with 5 items and another one with 3 items, or a house with 2 timeslots and another one with 4)
        data['house_s_max'] = pd.DataFrame(self.house_s_max).fillna(0)
        data['house_s_min'] = pd.DataFrame(self.house_s_min).fillna(0)
        data['h_prod'] = self.h_prod
        return data


    def create_sets(self, model, data):

        model.b = pyo.Set(initialize = np.arange(1, data['num_bins'] + 1))
        model.t = pyo.Set(initialize = np.arange(1, data['max_tims'] + 1))
        model.i = pyo.Set(initialize = np.arange(1, data['max_len'] + 1))
        model.ev = pyo.Set(initialize = np.arange(1, data['num_evs'] + 1))
        model.s = pyo.Set(initialize = np.arange(1, data['num_ess'] + 1))
        model.h = pyo.Set(initialize = np.arange(1, data['num_houses'] + 1))
        model.hs = pyo.Set(initialize = np.arange(1, data['max_st'] + 1))


    def create_parameters(self, model, data):
        model.tim_lens = pyo.Param(model.h, model.t, initialize = self._auxDictionary(np.array(data['tim_lens'])))
        model.weights = pyo.Param(model.h, model.t, model.i, initialize = self.convert_to_tuple_dict(data['house_items'], data))
        model.dates = pyo.Param(model.h, model.t, model.i, initialize = self.convert_to_tuple_dict(data['house_items_date'], data))
        model.numbers = pyo.Param(model.h, model.t, model.i, initialize = self.convert_to_tuple_dict(data['house_items_num'], data))
        model.bin_capacities = pyo.Param(model.b, initialize = self._auxDictionary(np.array(data['bin_capacities']).ravel()))
        model.bins_maximum = pyo.Param(model.b, initialize = self._auxDictionary(np.array(data['bins_maximum']).ravel()))
        model.items_maximum = pyo.Param(model.h, model.t, model.i, initialize = self.convert_to_tuple_dict(data['house_items_max'], data))
        model.baseload = pyo.Param(initialize = data['baseload'])
        model.fact = pyo.Param(initialize = data['fact'])
        model.n_bins_per_hour = pyo.Param(initialize = data['n_bins_per_hour'])
        model.num_bins = pyo.Param(initialize = data['num_bins'])
        model.flexibilities = pyo.Param(model.h, model.t, model.i, initialize = self.convert_to_tuple_dict(data['house_items_flex'], data))
        model.export_prices = pyo.Param(model.b, initialize = self._auxDictionary(np.array(data['export_prices']).ravel()))
        model.import_prices = pyo.Param(model.b, initialize = self._auxDictionary(np.array(data['import_prices']).ravel()))
        model.ev_soc_min = pyo.Param(model.ev, initialize = self._auxDictionary(np.array(data['minimum_soc'])))
        model.ev_soc_max = pyo.Param(model.ev, initialize = self._auxDictionary(np.array(data['maximum_soc'])))
        model.ev_trip = pyo.Param(model.ev, initialize = self._auxDictionary(np.array(data['ev_trip'])))
        model.ev_initial_soc = pyo.Param(model.ev, initialize = self._auxDictionary(np.array(data['initial_soc'])))
        model.ev_p_charger = pyo.Param(model.b, model.ev, initialize = self._auxDictionary(np.array(data['ev_p_charger'].transpose())))
        model.availability = pyo.Param(model.b, model.ev, initialize = self._auxDictionary(np.array(data['availability'].transpose())))
        model.travelling = pyo.Param(model.b, model.ev, initialize = self._auxDictionary(np.array(data['travelling'].transpose())))
        model.n = data['efficiency'] # Efficiency
        model.p_charger = data['p_charger'] # Charging station power
        model.degradation_cost = data['degradation_cost']
        model.p_grid_max = data['p_grid_max']
        model.s_min = pyo.Param(model.s, initialize = self._auxDictionary(np.array(data['s_minimum_soc'])))
        model.s_max = pyo.Param(model.s, initialize = self._auxDictionary(np.array(data['s_maximum_soc'])))
        model.s_initial_soc = pyo.Param(model.s, initialize = self._auxDictionary(np.array(data['s_initial_soc'])))
        model.h_s_min = pyo.Param(model.h, model.hs, initialize = self._auxDictionary(np.array(data['house_s_min'])))
        model.h_s_max = pyo.Param(model.h, model.hs, initialize = self._auxDictionary(np.array(data['house_s_max'])))
        model.h_s_initial_soc = pyo.Param(model.h, model.hs, initialize = self._auxDictionary(np.array(data['house_s_soc'])))
        model.h_prod = pyo.Param(model.b, model.h, initialize = self._auxDictionary(np.array(data['h_prod'])))


    def create_variables(self, model):

        model.x = pyo.Var(model.b, model.h, model.t, model.i, domain=pyo.Binary, initialize=0)
        model.pImp = pyo.Var(model.b, domain=pyo.NonNegativeReals, initialize=0)
        model.pExp = pyo.Var(model.b, domain=pyo.NonNegativeReals, initialize=0)
        model.ev_charge = pyo.Var(model.b, model.ev, domain=pyo.NonNegativeReals, initialize=0)
        model.ev_discharge = pyo.Var(model.b, model.ev, domain=pyo.NonNegativeReals, initialize=0)
        model.ev_soc = pyo.Var(model.b, model.ev, domain=pyo.NonNegativeReals, initialize=0)
        model.is_importing = pyo.Var(model.b, domain=pyo.Binary, initialize=0)
        model.s_charge = pyo.Var(model.b, model.s, domain=pyo.NonNegativeReals, initialize=0)
        model.s_discharge = pyo.Var(model.b, model.s, domain=pyo.NonNegativeReals, initialize=0)
        model.s_soc = pyo.Var(model.b, model.s, domain=pyo.NonNegativeReals, initialize=0)
        model.ev_tripn = pyo.Var(model.b, model.ev, domain = pyo.Reals, initialize = 0)
        model.h_s_charge = pyo.Var(model.b, model.h, model.hs, domain=pyo.NonNegativeReals, initialize=0)
        model.h_s_discharge = pyo.Var(model.b, model.h, model.hs, domain=pyo.NonNegativeReals, initialize=0)
        model.h_s_soc = pyo.Var(model.b, model.h, model.hs, domain=pyo.NonNegativeReals, initialize=0)
        model.h_pImp = pyo.Var(model.b, model.h, domain=pyo.NonNegativeReals, initialize=0)
        model.h_pExp = pyo.Var(model.b, model.h, domain=pyo.NonNegativeReals, initialize=0)


    def create_constraints(self, model, data):

        # An item of a timeslot can't be in more than one bin and all items have to be placed
        # The first condition is for those items that belongs to the timeslot
        # THe second condition is for those items that don't belong to the timeslot
        def _unique_bin(m,h,t,i):
            if (i <= m.tim_lens[h,t]): # Items used
                return sum(m.x[b,h,t,i] for b in m.b) == 1
            else: # Items not used
                return sum(m.x[b,h,t,i] for b in m.b) == 0
            #return sum([m.x[k, t, i] for k in np.arange(1, data['num_bins'] + 1)]) <= 1
        model.unique_bin = pyo.Constraint(model.h, model.t, model.i, rule = _unique_bin)


        # All the items of the timeslots have to be placed in the bins
        def _all_items_placed(m,t):
            return sum([m.x[k, t, i] for i in np.arange(1, data['max_len'] + 1) for k in np.arange(1, data['num_bins'] + 1)]) == m.tim_lens[t]
        #model.all_items_placed = pyo.Constraint(model.t, rule = _all_items_placed)


        # A bin can't contain more than one item of a timeslot (the timeslot items can't be in the same bin)
        def _one_item_per_bin(m, h, t, b):
            return sum([m.x[b, h, t, i] for i in np.arange(1, data['max_len'] + 1)]) <= 1
        model.one_item_per_bin = pyo.Constraint(model.h, model.t, model.b, rule = _one_item_per_bin)


        # The items of a timeslot have to be placed in consecutive bins (in an ascendent order)
        # Item 2 should be placed in the next bin of Item 1 and so on
        # Regarding the condition b == 1, in the first bin it is not possible to have Item higher than 1 (because the previous condition is just for b > 1)
        def _ascendent_order(m,h,t,i,b):
            if (i > 1 and i <= m.tim_lens[h,t] and b > 1):
                return m.x[b, h, t, i ] * b - m.x[b-1, h, t, i-1] * (b-1) <= 1
            elif (b == 1):
                return m.x[b, h, t, i] * i <= 1
            else:
                return pyo.Constraint.Skip
        model.ascendent_order = pyo.Constraint(model.h, model.t, model.i, model.b, rule = _ascendent_order)


        # The items of a timeslot have to be placed in consecutive bins (in an ascendent order)
        # Item 2 should be placed in the next bin of Item 1 and so on
        def _ascendent_order2(m,h, t,i,b):
            if (i > 1 and i <= m.tim_lens[h,t] and b > 1):
                return m.x[b, h, t, i ] * b - m.x[b-1, h, t, i-1] * (b-1) >= 0
            else:
                return pyo.Constraint.Skip
        model.ascendent_order2 = pyo.Constraint(model.h, model.t, model.i, model.b, rule = _ascendent_order2)



        # The items of a timeslot have to be placed in consecutive bins (in a descendent order)
        # Item 1 should be placed in the previous bin of Item 2 and so on
        def _descendent_order(m,h,t,i,b):
            if (i < m.tim_lens[h,t] and m.tim_lens[h,t] > 1 and b < m.num_bins):
                return m.x[b+1, h, t, i+1] * (b+1) - m.x[b, h, t, i] * b <= 1
            else:
                return pyo.Constraint.Skip
        model.descendent_order = pyo.Constraint(model.h, model.t, model.i, model.b, rule = _descendent_order)



        # The items of a timeslot have to be placed in consecutive bins (in a descendent order)
        # Item 1 should be placed in the previous bin of Item 2 and so on
        def _descendent_order2(m,h,t,i,b):
            if (i < m.tim_lens[h,t] and m.tim_lens[h,t] > 1 and b < m.num_bins):
                return m.x[b+1, h, t, i+1] - m.x[b, h, t, i] >= 0
            else:
                return pyo.Constraint.Skip
        model.descendent_order2 = pyo.Constraint(model.h, model.t, model.i, model.b, rule = _descendent_order2)



        # The house and appliance flexibilities have to be respected (min limit)
        def _flexibility_max(m,h, t,i,b):
            if (i <= m.tim_lens[h,t]):
                #diff = b - m.dates[t,i]
                #return m.x[b, t, i] * diff >= -1 * m.flexibilities[t,i] * m.n_bins_per_hour
                return m.x[b, h, t, i] * b - m.x[b, h, t, i] * m.dates[h,t,i] >= -1 * m.flexibilities[h,t,i] * m.n_bins_per_hour
            else:
                return pyo.Constraint.Skip
        model.flexibility_max = pyo.Constraint(model.h, model.t, model.i, model.b, rule = _flexibility_max)


        # The house and appliance flexibilities have to be respected (max limit)
        def _flexibility_min(m,h,t,i,b):
            if (i <= m.tim_lens[h,t]):
                return m.x[b, h, t, i] * b - m.x[b, h, t, i] * m.dates[h,t,i] <= m.flexibilities[h,t,i] * m.n_bins_per_hour
            else:
                return pyo.Constraint.Skip
        model.flexibility_min = pyo.Constraint(model.h, model.t, model.i, model.b, rule = _flexibility_min)


        # Calculate trip energy for all EVs for each hour
        def _balance_etripn(m,ev,b):
            return m.ev_tripn[b,ev] == m.ev_trip[ev]*m.travelling[b,ev]/(sum([m.travelling[k,ev] for k in m.b]))
        model.balance_etripn = pyo.Constraint(model.ev, model.b, rule = _balance_etripn)


        # Balance Community Load
        def _balance(m,b):
            return sum(m.h_pImp[b,h] for h in m.h) + m.pExp[b] + sum(m.ev_charge[b, ev] for ev in m.ev) + sum(m.s_charge[b, s] for s in m.s) == sum(m.h_pExp[b,h] for h in m.h) + m.bin_capacities[b] + m.pImp[b] + sum(m.ev_discharge[b, ev] for ev in m.ev) + sum(m.s_discharge[b, s] for s in m.s)
        model.balance = pyo.Constraint(model.b, rule = _balance)


        # Balance House Load
        def _house_balance(m,b,h):
            return sum(m.weights[h,t,i] * m.x[b, h, t, i] for t in m.t for i in m.i if i <= m.tim_lens[h,t]) + m.h_pExp[b,h] + sum(m.h_s_charge[b, h, hs] for hs in m.hs) == m.h_prod[b,h] + m.h_pImp[b,h] + sum(m.h_s_discharge[b, h, hs] for hs in m.hs)
        model.house_balance = pyo.Constraint(model.b, model.h, rule = _house_balance)



        # Calculation of the EV SOC
        def _soc_ev(m,b,ev):
            if (b == 1):
                return m.ev_soc[b,ev] == m.ev_initial_soc[ev] + m.ev_charge[b,ev]*m.n - m.ev_discharge[b,ev]/m.n - m.ev_tripn[b,ev]
            else:
                return m.ev_soc[b,ev] == m.ev_soc[b-1,ev] + m.ev_charge[b,ev]*m.n - m.ev_discharge[b,ev]/m.n - m.ev_tripn[b,ev]
        model.soc_ev = pyo.Constraint(model.b, model.ev, rule = _soc_ev)


        # Calculation of the storage (battery) SOC
        def _soc_s(m,b,s):
            if (b == 1):
                return m.s_soc[b,s] == m.s_initial_soc[s] + m.s_charge[b,s]*m.n - m.s_discharge[b,s]/m.n
            else:
                return m.s_soc[b,s] == m.s_soc[b-1,s] + m.s_charge[b,s]*m.n - m.s_discharge[b,s]/m.n
        model.soc_s = pyo.Constraint(model.b, model.s, rule = _soc_s)



        # Calculation of the house storage (battery) SOC
        def _h_soc_s(m,b,h,hs):
            if (b == 1):
                return m.h_s_soc[b,h,hs] == m.h_s_initial_soc[h,hs] + m.h_s_charge[b,h,hs]*m.n - m.h_s_discharge[b,h,hs]/m.n
            else:
                return m.h_s_soc[b,h,hs] == m.h_s_soc[b-1,h,hs] + m.h_s_charge[b,h,hs]*m.n - m.h_s_discharge[b,h,hs]/m.n
        model.h_soc_s = pyo.Constraint(model.b, model.h, model.hs, rule = _h_soc_s)


        # Charging considering the EVs availability
        def _charge_available(m,b,ev):
            return m.ev_charge[b,ev] <= m.ev_p_charger[b,ev] * m.availability[b,ev]
        model.ch_available = pyo.Constraint(model.b, model.ev, rule = _charge_available)


        # Discharging considering the EVs availability
        def _discharge_available(m,b,ev):
            return m.ev_discharge[b,ev] <= m.ev_p_charger[b,ev] * m.availability[b,ev]
        model.dch_available = pyo.Constraint(model.b, model.ev, rule = _discharge_available)



        # Storage Charging limit
        def _s_charge_available(m,b,s):
            return m.s_charge[b,s] <= m.p_charger
        model.s_ch_available = pyo.Constraint(model.b, model.s, rule = _s_charge_available)


        # Storage Charging limit
        def _s_discharge_available(m,b,s):
            return m.s_discharge[b,s] <= m.p_charger
        model.s_dch_available = pyo.Constraint(model.b, model.s, rule = _s_discharge_available)


        # Limit the power imported from the grid
        def _limit_pImp(m,b):
            #return m.pImp[b] <= min(0, sum(m.weights[t,i] * m.x[b, t, i] for t in m.t for i in m.i if i <= m.tim_lens[t]) - m.bin_capacities[b])
            return m.pImp[b] <= m.p_grid_max*m.is_importing[b];
        model.limit_pImp = pyo.Constraint(model.b, rule = _limit_pImp)


        # Limit for the power exported to the grid
        def _limit_pExp(m,b):
            return m.pExp[b] <= m.p_grid_max*(1-m.is_importing[b]);
        model.limit_pExp = pyo.Constraint(model.b, rule = _limit_pExp)


        # EV SOC minimum
        def _soc_min(m,b,ev):
            return m.ev_soc[b,ev] >= m.ev_soc_min[ev]
        model.soc_min = pyo.Constraint(model.b, model.ev, rule = _soc_min)


        # EV SOC maximum
        def _soc_max(m,b,ev):
            return m.ev_soc[b,ev] <= m.ev_soc_max[ev]
        model.soc_max = pyo.Constraint(model.b, model.ev, rule = _soc_max)



        # Storage SOC minimum
        def _s_soc_min(m,b,s):
            return m.s_soc[b,s] >= m.s_min[s]
        model.s_soc_min = pyo.Constraint(model.b, model.s, rule = _s_soc_min)


        # Storage SOC maximum
        def _s_soc_max(m,b,s):
            return m.s_soc[b,s] <= m.s_max[s]
        model.s_soc_max = pyo.Constraint(model.b, model.s, rule = _s_soc_max)


        # House Storage SOC minimum
        def _h_s_soc_min(m,b,h,hs):
            return m.h_s_soc[b,h,hs] >= m.h_s_min[h,hs]
        model.h_s_soc_min = pyo.Constraint(model.b, model.h, model.hs, rule = _h_s_soc_min)


        # House Storage SOC maximum
        def _h_s_soc_max(m,b,h,hs):
            return m.h_s_soc[b,h,hs] <= m.h_s_max[h,hs]
        model.h_s_soc_max = pyo.Constraint(model.b, model.h, model.hs, rule = _h_s_soc_max)



    def create_objective_function(self, model):

        def _FOag(m):
            return sum(m.pImp[j]/1000*m.import_prices[j] - m.pExp[j]/1000*m.export_prices[j] + sum((m.ev_charge[j,ev]+m.ev_discharge[j,ev])/1000 for ev in m.ev)*m.degradation_cost for j in m.b)
            #return sum(m.pImp[j] for j in m.b)

        model.FOag = pyo.Objective(rule = _FOag, sense = pyo.minimize)


    def get_dataframes(self, model):

        dataframes = {}

        dataframes['mDf'] = self.ext_pyomo_vals(model.x).transpose()
        dataframes['weight_df'] = self.ext_pyomo_vals(model.weights).transpose()
        dataframes['dates_df'] = self.ext_pyomo_vals(model.dates).transpose()
        dataframes['number_df'] = self.ext_pyomo_vals(model.numbers).transpose()
        dataframes['item_max_df'] = self.ext_pyomo_vals(model.items_maximum).transpose()
        dataframes['flex_df'] = self.ext_pyomo_vals(model.flexibilities).transpose()
        #dataframes['timLens_df'] = self.ext_pyomo_vals(model.tim_lens)[0]
        dataframes['timLens_df'] = self.ext_pyomo_vals(model.tim_lens).transpose()
        dataframes['production_df'] = self.ext_pyomo_vals(model.bin_capacities)[0]
        dataframes['pImp_df'] = self.ext_pyomo_vals(model.pImp)[0]
        dataframes['pExp_df'] = self.ext_pyomo_vals(model.pExp)[0]
        dataframes['evCharge_df'] = self.ext_pyomo_vals(model.ev_charge).transpose()
        dataframes['evDischarge_df'] = self.ext_pyomo_vals(model.ev_discharge).transpose()
        dataframes['isImporting_df'] = self.ext_pyomo_vals(model.is_importing)[0]
        dataframes['evSoc_df'] = self.ext_pyomo_vals(model.ev_soc).transpose()
        dataframes['importPrices_df'] = self.ext_pyomo_vals(model.import_prices)[0]
        dataframes['exportPrices_df'] = self.ext_pyomo_vals(model.export_prices)[0]
        dataframes['sCharge_df'] = self.ext_pyomo_vals(model.s_charge).transpose()
        dataframes['sDischarge_df'] = self.ext_pyomo_vals(model.s_discharge).transpose()
        dataframes['sSoc_df'] = self.ext_pyomo_vals(model.s_soc).transpose()
        dataframes['h_pImp_df'] = self.ext_pyomo_vals(model.h_pImp).transpose()
        dataframes['h_pExp_df'] = self.ext_pyomo_vals(model.h_pExp).transpose()
        dataframes['h_prod_df'] = self.ext_pyomo_vals(model.h_prod).transpose()
        dataframes['h_s_charge_df'] = self.ext_pyomo_vals(model.h_s_charge)
        dataframes['h_s_discharge_df'] = self.ext_pyomo_vals(model.h_s_discharge)
        dataframes['h_s_soc_df'] = self.ext_pyomo_vals(model.h_s_soc)
        dataframes['ev_tripn_df'] = self.ext_pyomo_vals(model.ev_tripn)
        dataframes['ev_connected_df'] = self.ext_pyomo_vals(model.availability)
        dataframes['ev_travelling_df'] = self.ext_pyomo_vals(model.travelling)

        return dataframes


    def get_results(self, model, data):
        print("Results")
        #print(ext_pyomo_vals(model.x))

        self.dataframes = self.get_dataframes(model)

        # Create dataframes
        mDf = self.dataframes['mDf']
        weight_df = self.dataframes['weight_df']
        dates_df = self.dataframes['dates_df']
        number_df = self.dataframes['number_df']
        item_max_df = self.dataframes['item_max_df']
        flex_df = self.dataframes['flex_df']
        timLens_df = self.dataframes['timLens_df']
        production_df = self.dataframes['production_df']
        pImp_df = self.dataframes['pImp_df']
        pExp_df = self.dataframes['pExp_df']
        evCharge_df = self.dataframes['evCharge_df']
        evDischarge_df = self.dataframes['evDischarge_df']
        isImporting_df = self.dataframes['isImporting_df']
        evSoc_df = self.dataframes['evSoc_df']
        importPrices_df = self.dataframes['importPrices_df']
        exportPrices_df = self.dataframes['exportPrices_df']
        sCharge_df = self.dataframes['sCharge_df']
        sDischarge_df = self.dataframes['sDischarge_df']
        sSoc_df = self.dataframes['sSoc_df']
        h_pImp_df = self.dataframes['h_pImp_df']
        h_pExp_df = self.dataframes['h_pExp_df']
        h_prod_df = self.dataframes['h_prod_df']
        h_s_charge_df = self.dataframes['h_s_charge_df']
        h_s_discharge_df = self.dataframes['h_s_discharge_df']
        h_s_soc_df = self.dataframes['h_s_soc_df']

        placed_timeslots = []
        demand_array = []

        for b in np.arange(1, data['num_bins']+1):

            tmp_array = []

            print("------------------------------------------------------------")
            print("Bin {}".format(b))
            print("------------------------------------------------------------")

            for h in np.arange(1, data['num_houses']+1):

                for t in np.arange(1, data['max_tims']+1):
                    for i in np.arange(1, data['max_len']+1):


                        if mDf[b][t][i][h] == 1:
                            #print("Timeslot ", t, "Item ", i, " - weight: ", data['weights'][t-1][i-1])
                            print("House ", h, "Timeslot ", t, "Item ", i)

                            firstItemDate = b - (i-1)

                            placed_timeslots.append(str(int(number_df[h][i][t])) + "-" + str(int(i-1)) + "-" + str(weight_df[h][i][t]) + "-" + str(int(b)) + "-" + str(int(timLens_df[h][t])) + "-" + str(int(firstItemDate)) + "-" + str(item_max_df[h][i][t]) + "-" + str(int(dates_df[h][i][t])) + "-" + str(flex_df[h][i][t]))


                demand = sum(weight_df[h][i][t] * mDf[b][t][i][h] for t in np.arange(1, data['max_tims']+1) for i in np.arange(1, data['max_len']+1) if i <= timLens_df[h][t])
                tmp_array.append(demand)

            demand_array.append(tmp_array)
            demand_df = pd.DataFrame(np.array(demand_array))
            demand_df.columns += 1

            self.dataframes['demand_df'] = demand_df

            for h in np.arange(1, data['num_houses']+1):
                print("----- House " + str(h) + " -----")
                print("Prod: ", h_prod_df[b][h])
                print("pImp: ", h_pImp_df[b][h])
                print("pExp: ", h_pExp_df[b][h])
                print("sCharge: ", h_s_charge_df[h][b])
                print("sDischarge: ", h_s_discharge_df[h][b])
                print("sSoc: ", h_s_soc_df[h][b])

            print("Production: ", production_df[b])
            #print("Demand: ", demand_df[b])
            #print("Excess of Production: ", (production_df[b]-demand_df[b]))

            print("pImp: ", pImp_df[b])
            print("pExp: ", pExp_df[b])

            #print("Is Importing: " + str(isImporting_df[b]))
            #print("Is Exporting: " + str(1-int(isImporting_df[b])))

            print("ev_charge: ")
            print(evCharge_df[b])
            print("ev_discharge: ")
            print(evDischarge_df[b])

            print("ev_soc: ")
            print(evSoc_df[b])

            print("s_charge: ")
            print(sCharge_df[b])
            print("s_discharge: ")
            print(sDischarge_df[b])

            print("s_soc: ")
            print(sSoc_df[b])

            print("import prices: ", importPrices_df[b])
            print("export prices: ", exportPrices_df[b])

            print('cost (+): ', exportPrices_df[b] * pExp_df[b]/1000)
            print('cost (-):', importPrices_df[b] * pImp_df[b]/1000)

        return placed_timeslots


    def execute_knapsack(self):

        now = datetime.datetime.now()

        start_time = now.strftime("%H:%M:%S")
        print("Start Time =", start_time)

        data = self.create_data_model()
        model = pyo.ConcreteModel()

        self.create_sets(model, data)
        self.create_parameters(model, data)
        self.create_variables(model)

        self.create_constraints(model, data)
        self.create_objective_function(model)


        model.write('res_V4_EC.lp',  io_options={'symbolic_solver_labels': True})

        opt = pyo.SolverFactory('cplex', executable='C:/Program Files/IBM/ILOG/CPLEX_Studio221/cplex/bin/x64_win64/cplex.exe')
        opt.options['LogFile'] = 'res_V4_EC.log'

        results = opt.solve(model)#, tee=True)
        results.write()


        now = datetime.datetime.now()

        end_time = now.strftime("%H:%M:%S")
        print("End Time =", end_time)
        print("Dif: {}".format(datetime.datetime.strptime(end_time, "%H:%M:%S") - datetime.datetime.strptime(start_time, "%H:%M:%S")))

        placed_timeslots = self.get_results(model, data)
        all_timeslots = []

        for i in list(range(len(self.items))):
            for p in range(len(self.items[i])):
                w = self.items[i][p]
                d = self.dates[i][p]
                n = self.numbers[i][p]
                flex = self.flexibilities[i][p]
                max = self.items_maximum[i][p]
                if (p == 0):  # First item of the timeslot
                    first_item_date = d
                all_timeslots.append(str(n) + "-" + str(p) + "-" + str(w) + "-" + str(d) + "-" + str(len(self.items[i])) + "-" + str(first_item_date) + "-" + str(max) + "-" + str(d) + "-" + str(flex))


        return [all_timeslots, placed_timeslots, [], []]