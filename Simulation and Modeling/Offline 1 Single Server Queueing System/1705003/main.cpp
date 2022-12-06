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

	ofstream out("mm1.out");
	out << setprecision(10);
	ifstream in("mm1.in");

	Ftype mean_interarrival, mean_service;
	int num_delays_required;
	in >> mean_interarrival >> mean_service >> num_delays_required;

	out<<"Single-server queueing system"<<endl<<endl;
	out<<"Mean interarrival time = "<<mean_interarrival<<" minutes"<<endl;
	out<<"Mean service time = "<<mean_service<<" minutes"<<endl;
	out<<"Number of customers = "<<num_delays_required<<endl;
	out<<"Generator seed = "<<seed<<endl<<endl;

	exponential_distribution<Ftype> iad(1.0 / mean_interarrival);
	exponential_distribution<Ftype> std(1.0 / mean_service);

	Simulator<Ftype, URNG> s1(iad, std);
	s1.set_generator(generator);
	s1.simulate(num_delays_required);

	out << "Simulation results:" << endl
		<< endl;
	out << "Average delay in queue: " << s1.get_avg_delays_in_q() << endl;
	out << "Average number in queue: " << s1.get_avg_num_in_q() << endl;
	out << "Server utilization: " << s1.get_server_utilization() << endl;
	out << "Time simulation ended: " << s1.get_simulation_end_time() << endl
		<< endl;

	Ftype lambda = 1.0 / mean_interarrival;
	Ftype mu = 1.0 / mean_service;
	Ftype avg_delay_in_q = lambda / (mu * (mu - lambda));
	Ftype avg_num_in_q = lambda * lambda / (mu * (mu - lambda));
	Ftype server_utilization = lambda / mu;
	Ftype avg_delay_in_system = 1 / (mu - lambda);
	Ftype simulation_end_time = avg_delay_in_system * num_delays_required;

	out << "Theoretical values:" << endl
		<< endl;
	out << "Average delay in queue: " << avg_delay_in_q << endl;
	out << "Average number in queue: " << avg_num_in_q << endl;
	out << "Server utilization: " << server_utilization << endl;
	out << "Time simulation ended: " << simulation_end_time << endl;

	

	return 0;
}
