import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np


from procsimulator.ConsumptionGenerator import ConsumptionGenerator
from procsimulator.DataFromSmile import DataFromSmile
from procsimulator.DataFromTomorrow import DataFromTomorrow
from procsimulator.RenewableEnergyGenerator import RenewableEnergyGenerator
from procsimulator.CommunityGenerator import CommunityGenerator
from procsimulator.CommunityManager import CommunityManager
from procsimulator.Evaluation import Evaluation

from CommunityManagement import CommunityManagement


def post_processing_pyomo(cm, reg, path_steps_minutes, path_steps_after_otimization):

    before = pd.read_csv(path_steps_minutes + '/netload.csv', sep=';')
    before.columns = ['Date', 'Demand', 'PV_Production', 'Wind_Production', 'Production', 'Netload']
    before['Date'] = pd.to_datetime(before['Date'])
    before.set_index('Date')

    before[:24*60*1]["Demand"].plot(legend=True, label='Demand')
    before[:24*60*1]["Production"].plot(legend=True, label='Production')
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.xlabel("Time (Hours)")
    plt.ylabel("Power (W)")
    #plt.savefig('before_opt_2.png')
    plt.show()



    # Getting the consumption profiles after the optimization
    opt = pd.read_csv(path_steps_after_otimization + '/netload.csv', sep=';')
    opt.columns = ['Date', 'Demand', 'PV_Production', 'Wind_Production', 'Production', 'Netload']
    opt['Date'] = pd.to_datetime(opt['Date'])
    opt.set_index('Date')


    opt[:24*60*1]["Demand"].plot(legend=True, label='Demand')
    opt[:24*60*1]["Production"].plot(legend=True, label='Production')
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.xlabel("Time (Hours)")
    plt.ylabel("Power (W)")
    #plt.savefig('after_opt_2.png')
    plt.show()


    # Calculate the difference between Demand and Production
    opt['Difference'] = opt['Demand'] - opt['Production']

    # Plot the difference
    opt[:24*60*1]["Difference"].plot(legend=True, label='Difference')
    plt.show()


    dfs = cm.dataframes

    demandd = opt[:24*60*1]
    demandd = demandd.set_index("Date")
    demandd = demandd.groupby(demandd.index.hour).mean()


    # ----- 1 -----


    #fig, ax = plt.subplots()
    #plt.stackplot(demandd.index, demandd["Production"], dfs['evSoc_df'].sum(), labels=['Production', 'SOC'])
    #demandd["Demand"].plot(color='green')
    #ax.legend(loc='upper left')
    #ax.set_title('After Optimization')
    #ax.set_xlabel('Hours')
    #ax.set_ylabel('Energy (Wh)')
    #plt.savefig('after_opt_2b.png')


    # ----- 2 -----

    #dfs["prod_used_df"] = dfs["production_df"] - dfs["demand_df"].transpose()[0][1:]
    #dfs["prod_used_df"].loc[dfs["prod_used_df"] >= 0] = dfs["production_df"] - dfs["pExp_df"] - dfs["evCharge_df"].sum() + dfs["evDischarge_df"].sum()
    #dfs["prod_used_df"].loc[dfs["prod_used_df"] < 0] = dfs["production_df"]

    #dfs["battery_used_df"] = dfs["demand_df"].transpose()[0][1:] - dfs["prod_used_df"] - dfs["pImp_df"]

    #fig, ax = plt.subplots()
    #plt.stackplot(dfs["pImp_df"].index, dfs["pImp_df"],  dfs["prod_used_df"], dfs["battery_used_df"], labels=['Grid', 'Production', 'EVs Discharge'])
    #dfs["demand_df"].transpose()[0][1:].plot(color='blue', label='Demand')
    #ax.legend(loc='upper left')
    #ax.set_title('After Optimization')
    #ax.set_xlabel('Hours')
    #ax.set_ylabel('Energy (Wh)')
    #plt.savefig('after_opt_2b.png')


    #----- 3 -----

    color_map = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]


    fig, ax = plt.subplots()
    plt.stackplot(dfs["pImp_df"].index, dfs["pImp_df"],  dfs["production_df"], dfs["evDischarge_df"].sum(), labels=['Grid Import', 'Production', 'EVs Discharge'], colors = color_map)
    dfs["demand_df"].transpose().sum().plot(color='green', label='Demand', linewidth=3)
    ax.legend(loc='upper left')
    ax.set_title('After 2nd Optimization')
    ax.set_xlabel('Hours')
    ax.set_ylabel('Energy (Wh)')
    ax.set_ylim(0, 17000)
    #plt.savefig('after_opt_2.png')
    plt.show()

    fig, ax = plt.subplots()
    plt.stackplot(dfs["pExp_df"].index, dfs["pExp_df"], dfs["demand_df"].transpose().sum(), dfs["evCharge_df"].sum(), labels=['Grid Export', 'Demand', 'EVs Charge'], colors = color_map)
    dfs["production_df"].plot(color='green', label='Production', linewidth=3)
    #dfs["evSoc_df"].sum().plot(color='black')
    ax.legend(loc='upper left')
    ax.set_title('After 2nd Optimization')
    ax.set_xlabel('Hours')
    ax.set_ylabel('Energy (Wh)')
    ax.set_ylim(0, 17000)
    #plt.savefig('after_opt_2b.png')
    plt.show()


    beforee = before[:24*60*1]
    beforee = beforee.set_index("Date")
    beforee = beforee.groupby(beforee.index.hour).mean()


    #fig, ax = plt.subplots()
    #ax.stackplot(beforee.index, beforee["Production"], labels=['Production'])
    #beforee["Demand"].plot(color='green')
    #ax.legend(loc='upper left')
    #ax.set_title('Before Optimization')
    #ax.set_xlabel('Hours')
    #ax.set_ylabel('Energy (Wh)')
    #plt.savefig('before_opt_2b.png')

    beforee["pImp"] = beforee["Demand"] - beforee["Production"]
    beforee["pImp"].loc[beforee["pImp"] < 0] = 0


    beforee["pExp"] = beforee["Production"] - beforee["Demand"]
    beforee["pExp"].loc[beforee["pExp"] < 0] = 0


    color_map = ["#e74c3c", "#9b59b6", "#34495e", "#2ecc71"]


    fig, ax = plt.subplots()
    plt.stackplot(beforee["Production"].index, beforee["Production"], beforee["pImp"], labels=['Production', 'Grid Import'], colors = color_map)
    beforee["Demand"].plot(color='green', label='Demand', linewidth=3)
    ax.legend(loc='upper left')
    ax.set_title('Before Optimization')
    ax.set_xlabel('Hours')
    ax.set_ylabel('Energy (Wh)')
    ax.set_ylim(0, 17000)
    #plt.savefig('before_opt_2.png')
    plt.show()


    fig, ax = plt.subplots()
    plt.stackplot(beforee["Demand"].index, beforee["Demand"], beforee["pExp"], labels=['Demand', 'Grid Export'], colors = color_map)
    beforee["Production"].plot(color='green', label='Production', linewidth=3)
    #dfs["evSoc_df"].sum().plot(color='black')
    ax.legend(loc='upper left')
    ax.set_title('Before Optimization')
    ax.set_xlabel('Hours')
    ax.set_ylabel('Energy (Wh)')
    ax.set_ylim(0, 17000)
    #plt.savefig('before_opt_2b.png')
    plt.show()


    before_df = before.iloc[:24*60]
    dt = pd.to_datetime(before_df.Date)
    before_demand_df = before_df.groupby([dt.dt.hour]).Demand.mean()
    before_demand_df.index = np.arange(1, len(before_demand_df) + 1)

    cost_df = dfs['importPrices_df']*before_demand_df/1000
    print(cost_df)


    # Calculate the metrics for the input
    evaluation_in = Evaluation(reg, before.iloc[:24*60], 0)
    print("Energy Used from Grid: " + "{:.2f}".format(evaluation_in.get_energy_used_from_grid()) + " kWh")
    print("Energy Used from Production: " + "{:.2f}".format(evaluation_in.get_energy_used_from_pv()*2) + " kWh")
    print("Energy Not Used from Production: " + "{:.2f}".format(evaluation_in.get_energy_not_used_from_pv()) + " kWh")
    print("Self Sufficiency (SS): " + "{:.2f}".format(evaluation_in.get_self_sufficiency()*100) + "%")
    print("Self Consumption (SC): " + "{:.2f}".format(evaluation_in.get_self_consumption()*100) + "%")
    print("Total Cost: " + "{:.2f}".format(cost_df.sum()) + "€")



    # Plot ev charge graph
    dfs['evCharge_df'].sum(axis=0).plot(legend=True, label='EV Charge')
    dfs['evDischarge_df'].sum(axis=0).plot(legend=True, label='EV Discharge')
    dfs['evSoc_df'].sum(axis=0).plot(legend=True, label='EV SOC')
    #dfs['demand_df'].transpose().plot(legend=True, label='Demand')
    plt.show()



    # Calculate the metrics for the output
    evaluation_out = Evaluation(reg, dfs, 0)

    print("Energy Used from Grid: " + "{:.2f}".format(evaluation_out.get_energy_imported_from_grid()) + " kWh")
    print("Energy Used from Production: " + "{:.2f}".format(evaluation_out.get_energy_used_from_production()) + " kWh")
    print("Energy Not Used from Production: " + "{:.2f}".format(evaluation_out.get_energy_exported_to_grid()) + " kWh")
    print("Self Sufficiency (SS): " + "{:.2f}".format(evaluation_out.get_ss_without_storage()) + "%")
    print("Self Sufficiency 2 (SS): " + "{:.2f}".format(evaluation_out.get_ss_with_storage()) + "%")
    print("Self Consumption (SC): " + "{:.2f}".format(evaluation_out.get_sc()) + "%")
    print("Total Cost: " + "{:.2f}".format(evaluation_out.get_costs(0.08)) + "€")



    # Prepend the column names with the name of the house
    h_demand_df = dfs["demand_df"].add_prefix('Demand H')
    h_demand_df.index = np.arange(1, 25)
    h_prod_df = dfs["h_prod_df"].transpose().add_prefix('Production H')
    h_prod_df.index = np.arange(1, 25)
    h_pImp_df = dfs["h_pImp_df"].transpose().add_prefix('Import H')
    h_pImp_df.index = np.arange(1, 25)
    h_pExp_df = dfs["h_pExp_df"].transpose().add_prefix('Export H')
    h_pExp_df.index = np.arange(1, 25)
    h_s_soc_df = dfs["h_s_soc_df"].groupby(level=[0]).sum().add_prefix('Total SOC H') # level 0 - group by hours / level 1 - group by house storages
    h_s_charge_df = dfs["h_s_charge_df"].groupby(level=[0]).sum().add_prefix('Total Charge H')
    h_s_discharge_df = dfs["h_s_discharge_df"].groupby(level=[0]).sum().add_prefix('Total DisCharge H')


    demand_df = convert_series_to_df(dfs["demand_df"].transpose().sum(), "Demand")
    production_df = convert_series_to_df(dfs["production_df"], "Production")
    pImp_df = convert_series_to_df(dfs["pImp_df"], "Import")
    pExp_df = convert_series_to_df(dfs["pExp_df"], "Export")
    sSoc_df = convert_series_to_df(dfs["sSoc_df"].sum(), "St Soc")
    sCharge_df = convert_series_to_df(dfs["sCharge_df"].sum(), "St Charge")
    sDischarge_df = convert_series_to_df(dfs["sDischarge_df"].sum(), "St Discharge")
    evSoc_df = convert_series_to_df(dfs["evSoc_df"].sum(), "EVs SOC")
    evCharge_df = convert_series_to_df(dfs["evCharge_df"].sum(), "EVs Charge")
    evDischarge_df = convert_series_to_df(dfs["evDischarge_df"].sum(), "EVs Discharge")
    ev_tripn_df = convert_series_to_df(dfs["ev_tripn_df"].transpose().sum(), "EVs Tripn")
    ev_connected_df = convert_series_to_df(dfs["ev_connected_df"].transpose().sum(), "EVs Connected")
    ev_travelling_df = convert_series_to_df(dfs["ev_travelling_df"].transpose().sum(), "EVs Travelling")

    output_df = pd.concat([h_demand_df, h_prod_df, h_pImp_df, h_pExp_df, h_s_soc_df, h_s_charge_df, h_s_discharge_df, demand_df, production_df, pImp_df, pExp_df, sSoc_df, sCharge_df, sDischarge_df, evSoc_df, evCharge_df, evDischarge_df, ev_tripn_df, ev_connected_df, ev_travelling_df], axis=1)
    print(output_df)

    output_df.to_csv("output.csv", sep=";")


    for house in np.arange(1, len(cg.get_community())+1):
        house_demand_df = dfs["demand_df"][house].to_frame()
        house_demand_df.columns = ["Demand"]
        house_demand_df.index = np.arange(1, 25)
        house_prod_df = dfs["h_prod_df"].transpose()[house].to_frame()
        house_prod_df.columns = ["Production"]
        house_pimp_df = dfs["h_pImp_df"].transpose()[house].to_frame()
        house_pimp_df.columns = ["Import"]
        house_pexp_df = dfs["h_pExp_df"].transpose()[house].to_frame()
        house_pexp_df.columns = ["Export"]
        house_soc_df = pd.DataFrame(dfs["h_s_soc_df"][house].transpose().to_numpy().reshape(24,2))
        house_soc_df.columns +=1
        house_soc_df = house_soc_df.add_prefix('SOC ')
        house_soc_df.index = np.arange(1, 25)
        house_charge_df = pd.DataFrame(dfs["h_s_charge_df"][house].transpose().to_numpy().reshape(24,2))
        house_charge_df.columns +=1
        house_charge_df = house_charge_df.add_prefix('Charge ')
        house_charge_df.index = np.arange(1, 25)
        house_discharge_df = pd.DataFrame(dfs["h_s_discharge_df"][house].transpose().to_numpy().reshape(24,2))
        house_discharge_df.columns +=1
        house_discharge_df = house_discharge_df.add_prefix('Discharge ')
        house_discharge_df.index = np.arange(1, 25)
        house_total_soc_df = pd.DataFrame(dfs["h_s_soc_df"].groupby(level=[0]).sum()[house])
        house_total_soc_df.columns = ["Total SOC"]
        house_total_charge_df = pd.DataFrame(dfs["h_s_charge_df"].groupby(level=[0]).sum()[house])
        house_total_charge_df.columns = ["Total Charge"]
        house_total_discharge_df = pd.DataFrame(dfs["h_s_discharge_df"].groupby(level=[0]).sum()[house])
        house_total_discharge_df.columns = ["Total Discharge"]
        output_house_df = pd.concat([house_demand_df, house_prod_df, house_pimp_df, house_pexp_df, house_soc_df, house_charge_df, house_discharge_df, house_total_soc_df, house_total_charge_df, house_total_discharge_df], axis=1)
        print(output_house_df)
        output_house_df.to_csv("output_h" + str(house) + ".csv", sep=";")


def convert_series_to_df(series, column):
    df = series.to_frame()
    df.index = np.arange(1, 25)
    df.columns = [column]
    return df


def pyomoStrategy(reg, path_steps_minute, path_steps_after_opt):

    buy_price_hour_kwh = [0.0918, 0.0918, 0.0918, 0.0918, 0.0918, 0.0918, 0.0918, 0.0918, 0.2417, 0.2417, 0.2417, 0.1484, 0.1484, 0.1484, 0.1484, 0.1484, 0.1484, 0.1484, 0.2417, 0.2417, 0.2417, 0.1484, 0.0918, 0.0918]
    sell_price_hour_kwh = [0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163,0.1163]

    cm = CommunityManagement(cg, path_steps_minutes, path_steps_after_opt)
    cm.execute(export_prices_hour = sell_price_hour_kwh, import_prices_hour=buy_price_hour_kwh, save_to_file=False)

    post_processing_pyomo(cm, reg, path_steps_minute, path_steps_after_opt)



if __name__ == "__main__":

    current_path = os.getcwd()
    print(current_path)

    path_steps_seconds = os.path.join(current_path)
    path_steps_minutes = "output/minute"
    path_steps_after_opt = "output/afterfirstoptimization"
    num_days = "1"
    generate_community = False


    # Defining the PROCSIM classes that will be used
    #cs = CommunitySpecificator("data.json")
    cg = ConsumptionGenerator("data.json", path_steps_seconds, path_steps_minutes)
    pv_dat = DataFromSmile("https://ems.prsma.com/solcast/public/Fazendinha_solcast-radiation-historical_30min.csv")
    wind_dat = DataFromTomorrow(
        "https://api.tomorrow.io/v4/timelines?location=-73.98529171943665,40.75872069597532&fields=pressureSurfaceLevel,pressureSeaLevel,precipitationIntensity,precipitationType,windSpeed,windGust,windDirection,temperature,temperatureApparent,cloudCover,cloudBase,cloudCeiling,weatherCode&timesteps=1h&units=metric&apikey=Yckmp3vREbJqyprWGGiTOC1pVaAYO0ZT")
    reg = RenewableEnergyGenerator(cg, pv_dat, wind_dat, cg.path_steps_minutes)
    cmg = CommunityGenerator(cg.path_steps_minutes)



    # Creating the house and user files as well as the consumption and production profiles
    #cs.execute()
    if (generate_community):
        cg.execute(num_days, "houses") # Consumption
        reg.execute(num_days) # Renewable Production
        cmg.execute() # Netload Calculation

    pyomoStrategy(reg, path_steps_minutes, path_steps_after_opt)