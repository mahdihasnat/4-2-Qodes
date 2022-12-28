#include <bits/stdc++.h>
using namespace std;

#define ALL(a) a.begin(), a.end()
#define FastIO                   \
	ios::sync_with_stdio(false); \
	cin.tie(0);                  \
	cout.tie(0)
#define IN freopen("input.txt", "r+", stdin)
#define OUT freopen("output.txt", "w+", stdout)

#define DBG(a) cerr << "line " << __LINE__ << " : " << #a << " --> " << (a) << endl
#define NL cerr << endl

template <class T1, class T2>
ostream &operator<<(ostream &os, const pair<T1, T2> &p)
{
	os << "{" << p.first << "," << p.second << "}";
	return os;
}
template <class T, size_t N>
ostream &operator<<(ostream &os, const array<T, N> &a)
{
	os << "{";
	for (auto x : a)
		os << x << " ";
	os << "}";
	return os;
}

template <class T>
ostream &operator<<(ostream &os, const vector<T> &a)
{
	os << "{ ";
	for (auto x : a)
		os << x << " ";
	os << "}";
	return os;
}
#include "simulator.h"
#include "distributions.h"

using Ftype = double;
using URNG = mt19937;
auto seed = chrono::steady_clock::now().time_since_epoch().count();
URNG generator(seed);
Ftype avg(my_exponential_distribution<Ftype> ed)
{
	Ftype sum = 0;
	const int n = 1000000;
	for (int i = 0; i < n; i++)
	{
		sum += ed(generator);
	}
	return sum / n;
}

int main() /* Main function. */
{

	ofstream out("inv.out");
	out << setprecision(10);
	ifstream in("inv.in");
	
	// num_months, &num_policies, &num_values_demand,
    //        &mean_interdemand, &setup_cost, &incremental_cost, &holding_cost,
    //        &shortage_cost, &minlag, &maxlag);

	int initial_inv_level;
	int num_months;
	int num_policies;
	int num_values_demand;


	in >> initial_inv_level >> num_months >> 
				num_policies >> num_values_demand;

	Ftype mean_interdemand;
	Ftype setup_cost;
	Ftype incremental_cost;
	Ftype holding_cost;
	Ftype shortage_cost;
	Ftype minlag;
	Ftype maxlag;

	in >> mean_interdemand >> setup_cost >> incremental_cost 
			>> holding_cost >> shortage_cost >> minlag >> maxlag;

	// DBG(initial_inv_level);
	// DBG(num_months);
	// DBG(num_policies);
	// DBG(num_values_demand);
	// DBG(mean_interdemand);
	// DBG(setup_cost);
	// DBG(incremental_cost);
	// DBG(holding_cost);
	// DBG(shortage_cost);
	// DBG(minlag);
	// DBG(maxlag);

	vector<Ftype> prob_distrib_demand(num_values_demand+1,0);
	for(int i=1;i<=num_values_demand;i++)
	{
		in >> prob_distrib_demand[i];
	}
	// DBG(prob_distrib_demand);



	out<<"Single-product inventory system\n\n";
	out<<"Initial inventory level"<<setw(24)<<initial_inv_level<<" items\n\n";
	out<<"Number of demand sizes"<<setw(25)<<num_values_demand<<"\n\n";
	out<<"Distribution function of demand sizes  ";
	for(int i=1;i<=num_values_demand;i++)
	{
		out<<setw(8)<<prob_distrib_demand[i];
	}
	out<<"\n\nMean interdemand time"<<setw(26)<<mean_interdemand<<"\n\n";
	out<<"Delivery lag range"<<setw(29)<<minlag<<" to"<<setw(10)<<maxlag<<" months\n\n";
	out<<"Length of the simulation"<<setw(23)<<num_months<<" months\n\n";
	out<<"K ="<<setw(6)<<setup_cost<<"   i ="<<setw(6)<<incremental_cost<<"   h ="<<setw(6)<<holding_cost<<"   pi ="<<setw(6)<<shortage_cost<<"\n\n";
	out<<"Number of policies"<<setw(29)<<num_policies<<"\n\n";
	out<<"                 Average        Average";
	out<<"        Average        Average\n";
	out<<"  Policy       total cost    ordering cost";
	out<<"  holding cost   shortage cost\n\n";


	my_exponential_distribution<Ftype> inter_demand_dist(1.0/mean_interdemand);
	my_uniform_real_distribution<Ftype> lag_dist(minlag,maxlag);
	my_discrete_distribution<int,Ftype> demand_size_dist(prob_distrib_demand);

	Simulator<Ftype,URNG> sim(inter_demand_dist,lag_dist,demand_size_dist);

	sim.set_generator(generator);
	sim.set_initial_inv_level(initial_inv_level);
	sim.set_num_months(num_months);
	sim.set_holding_cost(holding_cost);
	sim.set_shortage_cost(shortage_cost);
	sim.set_setup_cost(setup_cost);
	sim.set_incremental_cost(incremental_cost);

	for(int i=1;i<=num_policies;i++)
	{
		int smalls,bigs;
		in >> smalls >> bigs;
		sim.simulate(smalls,bigs);

		out<<fixed<<"("<<setw(3)<<smalls<<","<<setw(3)<<bigs<<")"<<
			setw(15)<<setprecision(2)<<
			sim.get_avg_total_cost()<<setw(15)<<setprecision(2)<<
			sim.get_avg_ordering_cost()<<setw(15)<<setprecision(2)<<
			sim.get_avg_holding_cost()<<setw(15)<<setprecision(2)<<
			sim.get_avg_shortage_cost()<<"\n\n";
	}

	return 0;
}
