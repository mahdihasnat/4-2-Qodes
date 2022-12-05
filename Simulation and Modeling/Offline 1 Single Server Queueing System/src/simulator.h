#ifndef SIMULATOR_H
#define SIMULATOR_H
#include "bits/stdc++.h"
using namespace std;

template <class Ftype = double,
		  class InterArrivalDistribution = exponential_distribution<Ftype>,
		  class ServiceTimeDistribution = exponential_distribution<Ftype>>
class Simulator
{
	InterArrivalDistribution inter_arrival_distribution;
	ServiceTimeDistribution service_time_distribution;

public:
	Simulator(InterArrivalDistribution inter_arrival_distribution = InterArrivalDistribution(),
			  ServiceTimeDistribution service_time_distribution = ServiceTimeDistribution())
		: inter_arrival_distribution(inter_arrival_distribution),
		  service_time_distribution(service_time_distribution)
	{
	}

	void simulate(int num_of_customer)
	{

	}

};

#endif // SIMULATOR_H