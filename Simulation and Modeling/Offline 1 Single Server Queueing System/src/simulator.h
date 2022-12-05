#ifndef SIMULATOR_H
#define SIMULATOR_H
#include "bits/stdc++.h"
using namespace std;

#define DBG(a) cerr << "line " << __LINE__ << " : " << #a << " --> " << (a) << endl
#define NL cerr << endl

template <class Ftype = double,
		  class URNG = default_random_engine,
		  class IAD = exponential_distribution<Ftype>,
		  class STD = exponential_distribution<Ftype>>
class Simulator
{
	IAD iad;
	STD std;
	URNG generator;

	enum SERVER_STATUS
	{
		IDLE,
		BUSY
	};
	enum EVENT_TYPE
	{
		NONE = 0,
		ARRIVE = 1,
		DEPART = 2
	};
	Ftype sim_time;
	SERVER_STATUS server_status;
	queue<Ftype> q;
	Ftype time_last_event;
	int num_custs_delayed;
	Ftype total_of_delays;
	Ftype area_num_in_q;
	Ftype area_server_status;
	EVENT_TYPE next_event_type;

#define num_events 2
	Ftype time_next_event[num_events + 1];

	void initialize()
	{
		/* Initialize the simulation clock. */
		sim_time = 0.0;
		/* Initialize the state variables. */
		server_status = SERVER_STATUS::IDLE;
		while (!q.empty())
			q.pop();
		time_last_event = 0.0;
		/* Initialize the statistical counters. */
		num_custs_delayed = 0;
		total_of_delays = 0.0;
		area_num_in_q = 0.0;
		area_server_status = 0.0;
		/* Initialize event list. Since no customers are present, the departure
		(service completion) event is eliminated from consideration. */
		time_next_event[ARRIVE] = sim_time + iad(generator);
		time_next_event[DEPART] = 1.0e+30;
	}
	void timing(void) /* Timing function. */
	{
		Ftype min_time_next_event = numeric_limits<Ftype>::max();
		next_event_type = NONE;
		/* Determine the event type of the next event to occur. */
		for (const auto event : {ARRIVE, DEPART})
			if (time_next_event[event] < min_time_next_event)
			{
				min_time_next_event = time_next_event[event];
				next_event_type = event;
			}

		/* Check to see whether the event list is empty. */
		if (next_event_type == NONE)
		{
			/* The event list is empty, so stop the simulation. */
			assert(0);
			// fprintf(outfile, "\nEvent list empty at time %f", sim_time);
			// exit(1);
		}
		/* The event list is not empty, so advance the simulation clock. */
		sim_time = min_time_next_event;
	}
	void update_time_avg_stats(void) /* Update area accumulators for time-average
	statistics. */
	{
		Ftype time_since_last_event;
		/* Compute time since last event, and update last-event-time marker. */
		time_since_last_event = sim_time - time_last_event;
		time_last_event = sim_time;
		/* Update area under number-in-queue function. */
		area_num_in_q += Ftype(q.size()) * time_since_last_event;
		/* Update area under server-busy indicator function. */
		area_server_status += server_status * time_since_last_event;
	}


	void arrive(void) /* Arrival event function. */
	{
		Ftype delay;
		/* Schedule next arrival. */
		time_next_event[ARRIVE] = sim_time + iad(generator);
		/* Check to see whether server is busy. */
		if (server_status == BUSY)
		{
			/* Server is busy, so increment number of customers in queue. */
			q.push(sim_time);
			// /* Check to see whether an overflow condition exists. */
			// if (num_in_q > Q_LIMIT)
			// {
			// 	/* The queue has overflowed, so stop the simulation. */
			// 	fprintf(outfile, "\nOverflow of the array time_arrival at");
			// 	fprintf(outfile, " time %f", sim_time);
			// 	exit(2);
			// }
			// /* There is still room in the queue, so store the time of arrival of the
			// arriving customer at the (new) end of time_arrival. */
			// time_arrival[num_in_q] = sim_time;
		}
		else
		{
			/* Server is idle, so arriving customer has a delay of zero. (The
			following two statements are for program clarity and do not affect
			the results of the simulation.) */
			delay = 0.0;
			total_of_delays += delay;
			/* Increment the number of customers delayed, and make server busy. */
			++num_custs_delayed;
			server_status = BUSY;
			/* Schedule a departure (service completion). */
			time_next_event[DEPART] = sim_time + std(generator);
		}
	}

	void depart(void) /* Departure event function. */
	{
		int i;
		Ftype delay;
		/* Check to see whether the queue is empty. */
		if (q.empty())
		{
			/* The queue is empty so make the server idle and eliminate the
			departure (service completion) event from consideration. */
			server_status = IDLE;
			time_next_event[DEPART] = numeric_limits<Ftype>::max();
		}
		else
		{
			/* The queue is nonempty, so decrement the number of customers in
			queue. */
			Ftype time_arrival = q.front();
			q.pop();
			/* Compute the delay of the customer who is beginning service and update
			the total delay accumulator. */
			delay = sim_time - time_arrival;
			total_of_delays += delay;
			/* Increment the number of customers delayed, and schedule departure. */
			++num_custs_delayed;
			time_next_event[DEPART] = sim_time + std(generator);
		}
	}

	void report(void) /* Report generator function. */
	{
		/* Compute and write estimates of desired measures of performance. */
		avg_delays_in_q = total_of_delays / num_custs_delayed;
		avg_num_in_q = area_num_in_q / sim_time;
		server_utilization = area_server_status / sim_time;
		simulation_end_time = sim_time;
	}

public:
	Ftype avg_delays_in_q;
	Ftype avg_num_in_q;
	Ftype server_utilization;
	Ftype simulation_end_time;



	Simulator(IAD iad = IAD(),
			  STD std = STD())
		: iad(iad),
		  std(std)
	{
	}

	void set_generator(URNG generator)
	{
		this->generator = generator;
	}

	void simulate(int num_of_customer)
	{
		/* Initialize the simulation. */
		initialize();
		/* Run the simulation while more delays are still needed. */
		while (num_custs_delayed < num_of_customer)
		{
			/* Determine the next event. */
			timing();
			/* Update time-average statistical accumulators. */
			update_time_avg_stats();
			// /* Invoke the appropriate event function. */
			switch (next_event_type)
			{
			case ARRIVE:
				arrive();
				break;
			case DEPART:
				depart();
				break;
			}
		}
		/* Invoke the report generator and end the simulation. */
		report();
	}

	Ftype get_avg_delays_in_q() const
	{
		return avg_delays_in_q;
	}
	Ftype get_avg_num_in_q() const
	{
		return avg_num_in_q;
	}
	Ftype get_server_utilization() const
	{
		return server_utilization;
	}
	Ftype get_simulation_end_time() const
	{
		return simulation_end_time;
	}

};

#endif // SIMULATOR_H