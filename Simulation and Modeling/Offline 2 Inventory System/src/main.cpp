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
Ftype avg(exponential_distribution<Ftype> ed)
{
	Ftype sum = 0;
	const int n = 10000;
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

	DBG(initial_inv_level);
	DBG(num_months);
	DBG(num_policies);
	DBG(num_values_demand);
	DBG(mean_interdemand);
	DBG(setup_cost);
	DBG(incremental_cost);
	DBG(holding_cost);
	DBG(shortage_cost);
	DBG(minlag);
	DBG(maxlag);

	vector<Ftype> prob_distrib_demand(num_values_demand+1,0);
	for(int i=1;i<=num_values_demand;i++)
	{
		in >> prob_distrib_demand[i];
	}
	DBG(prob_distrib_demand);

	my_discrete_distribution<int,Ftype> dd(prob_distrib_demand);
	DBG(dd(generator));


	return 0;
}
