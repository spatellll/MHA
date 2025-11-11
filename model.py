from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
import pandas as pd


class AccessModel(object):
    def __init__(self, tradeOff=1e-6, distanceThreshold=45, noShow=10, intervention=0, output="model/result"):
        print("Loading data...")
        self.demandDict, self.totalDemand = self.load_demand()
        self.supplyDict = self.load_supply(intervention)
        self.distanceDict, self.tractReachableDict, self.providerReachableDict = self.load_distance(distanceThreshold)
        print("Setting up model...")
        self.model, self.variables = self.setup_model(noShow, tradeOff)
        print("Solving model...")
        isOptimal = self.solve_model()
        if isOptimal:
            output = f"{output}_intervention{intervention}.csv"
            print(f"Model is optimal; Result saved to {output}")
            solution = self.get_solution()
            solution.to_csv(output, index=False)
        else:
            raise Exception("Model is infeasible")

    def load_distance(self, distanceThreshold):
        distanceDF = pd.read_csv("result/distance.csv")
        providerID = self.supplyDict.keys()
        tractID = self.demandDict.keys()
        distanceDF = distanceDF[distanceDF['PROVIDER_ID'].isin(providerID)]
        distanceDF = distanceDF[distanceDF['TRACT_ID'].isin(tractID)]
        distanceDF = distanceDF[distanceDF['DISTANCE'] <= distanceThreshold]
        distanceDict = distanceDF.set_index(['TRACT_ID', 'PROVIDER_ID'])['DISTANCE'].to_dict()
        tractReachableDict = distanceDF.groupby('TRACT_ID')['PROVIDER_ID'].apply(list).to_dict()
        providerReachableDict = distanceDF.groupby('PROVIDER_ID')['TRACT_ID'].apply(list).to_dict()
        return distanceDict, tractReachableDict, providerReachableDict

    def load_demand(self):
        demandDF = pd.read_csv("result/need.csv")[['TRACT_ID', 'DEMAND']]
        totalDemand = demandDF['DEMAND'].sum()
        return demandDF.set_index('TRACT_ID')['DEMAND'].to_dict(), totalDemand

    def load_supply(self, intervention):
        supplyDF = pd.read_csv("result/supply.csv")[['PROVIDER_ID', 'CASELOAD']]
        supplyDF['CASELOAD'] = supplyDF['CASELOAD'] * (100.0 + intervention) / 100.0
        return supplyDF.set_index('PROVIDER_ID')['CASELOAD'].to_dict()

    def setup_model(self, noShow, tradeOff):
        demandLocations = self.tractReachableDict.keys()
        supplyLocations = self.providerReachableDict.keys()
        arcs = self.distanceDict.keys()

        model = LpProblem("Access_Model", LpMinimize)
        x = LpVariable.dicts("x", arcs, cat="Continuous", lowBound=0)
        model += tradeOff * lpSum(x[i] * self.distanceDict[i] for i in arcs) + (1 - tradeOff) *100 * (1 - (lpSum(x[i] for i in arcs) / self.totalDemand)), "BiObjectiveMinimization"
        for i in demandLocations:
            model += lpSum(x[i, j] for j in self.tractReachableDict[i]) <= self.demandDict[i], f"DemandCapacity_{i}"
        for j in supplyLocations:
            model += lpSum(x[i, j] for i in self.providerReachableDict[j]) <= self.supplyDict[j] * (100.0 + noShow) / 100.0, f"SupplyCapacity_{j}"
        return model, x

    def solve_model(self):
        self.model.solve()
        status = self.model.status
        if LpStatus[status] == "Optimal":
            return True
        else:
            return False

    def get_solution(self):
        solution = {"TRACT_ID": [], "PROVIDER_ID": [], "ASSIGNMENT": [], 'DISTANCE': []}
        for i in self.distanceDict.keys():
            value = self.variables[i].value()
            if value > 1e-6:
                solution["TRACT_ID"].append(i[0])
                solution["PROVIDER_ID"].append(i[1])
                solution["ASSIGNMENT"].append(value)
                solution["DISTANCE"].append(self.distanceDict[i])
        solution = pd.DataFrame(solution)
        return pd.DataFrame(solution)