#ifndef SIMULATOR_H
#define SIMULATOR_H
#include "bits/stdc++.h"
using namespace std;

#define DBG(a) cerr << "line " << __LINE__ << " : " << #a << " --> " << (a) << endl
#define NL cerr << endl

#include "distributions.h"

template <class Ftype = double,
		  class URNG = default_random_engine,
		  class IDD = my_exponential_distribution<Ftype>,	// inter demand distribution
		  class DLD = my_uniform_real_distribution<Ftype>,	// delivery lag distribution
		  class DSD = my_discrete_distribution<int, Ftype>> // delivery size distribution
class Simulator
{
	URNG generator;
	IDD idd;
	DLD dld;
	DSD dsd;

	enum EVENT_TYPE
	{
		NONE = 0,
		ORDER_ARRIVAL = 1,
		DEMAND = 2,
		END_SIMULATION = 3,
		EVALUATE = 4
	};

	// simulation clock
	Ftype sim_time;
	Ftype time_last_event;

	EVENT_TYPE next_event_type;

	// state variables
	int inv_level;

	// amount for next order arrival
	int amount;

	// statistical counters
	Ftype total_ordering_cost;
	Ftype area_holding;
	Ftype area_shortage;

	// final results
	Ftype avg_holding_cost;
	Ftype avg_ordering_cost;
	Ftype avg_shortage_cost;

	// event list
#define num_events 4
	Ftype time_next_event[num_events + 1];

	// simulation parameters
	int initial_inv_level;
	int num_months;

	// costs
	Ftype holding_cost;
	Ftype shortage_cost;
	Ftype setup_cost;
	Ftype incremental_cost;

	void initialize(void) /* Initialization function. */
	{
		/* Initialize the simulation clock. */

		sim_time = 0;

		/* Initialize the state variables. */

		inv_level = initial_inv_level;
		time_last_event = 0;

		/* Initialize the statistical counters. */

		total_ordering_cost = 0;
		area_holding = 0;
		area_shortage = 0;

		/* Initialize the event list.  Since no order is outstanding, the order-
		arrival event is eliminated from consideration. */

		time_next_event[ORDER_ARRIVAL] = numeric_limits<Ftype>::max();
		time_next_event[DEMAND] = sim_time + idd(generator);
		time_next_event[END_SIMULATION] = num_months;
		time_next_event[EVALUATE] = 0;
	}

	void timing(void) /* Timing function. */
	{
		Ftype min_time_next_event = numeric_limits<Ftype>::max();

		next_event_type = NONE;

		/* Determine the event type of the next event to occur. */

		for (const auto event : {ORDER_ARRIVAL, DEMAND, END_SIMULATION, EVALUATE})
			if (time_next_event[event] < min_time_next_event)
			{
				min_time_next_event = time_next_event[event];
				next_event_type = event;
			}

		/* Check to see whether the event list is empty. */

		if (next_event_type == NONE)
		{

			/* The event list is empty, so stop the simulation */
			cerr << "Event list empty at time " << sim_time << endl;
			assert(false);
			exit(1);
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

		/* Determine the status of the inventory level during the previous interval.
		If the inventory level during the previous interval was negative, update
		area_shortage.  If it was positive, update area_holding.  If it was zero,
		no update is needed. */

		if (inv_level < 0)
			area_shortage -= inv_level * time_since_last_event;
		else if (inv_level > 0)
			area_holding += inv_level * time_since_last_event;
	}

	void order_arrival(void) /* Order arrival event function. */
	{
		/* Increment the inventory level by the amount ordered. */

		inv_level += amount;

		/* Since no order is now outstanding, eliminate the order-arrival event from
		consideration. */

		time_next_event[ORDER_ARRIVAL] = numeric_limits<Ftype>::max();
	}
	void demand(void) /* Demand event function. */
	{
		/* Decrement the inventory level by a generated demand size. */

		inv_level -= dsd(generator);

		/* Schedule the time of the next demand. */

		time_next_event[DEMAND] = sim_time + idd(generator);
	}

	void evaluate(int smalls, int bigs) /* Inventory-evaluation event function. */
	{
		/* Check whether the inventory level is less than smalls. */

		if (inv_level < smalls)
		{

			/* The inventory level is less than smalls, so place an order for the
			appropriate amount. */

			amount = bigs - inv_level;
			total_ordering_cost += setup_cost + incremental_cost * amount;

			/* Schedule the arrival of the order. */
			// via delivery lag distribution (dld)
			time_next_event[ORDER_ARRIVAL] = sim_time + dld(generator);
		}

		/* Regardless of the place-order decision, schedule the next inventory
		evaluation. */

		time_next_event[EVALUATE] = sim_time + 1.0;
	}

	void report(void) /* Report generator function. */
	{
		/* Compute and write estimates of desired measures of performance. */
		avg_ordering_cost = total_ordering_cost / num_months;
		avg_holding_cost = holding_cost * area_holding / num_months;
		avg_shortage_cost = shortage_cost * area_shortage / num_months;
	}

public:
	Simulator(IDD idd, DLD dld, DSD dsd) : idd(idd), dld(dld), dsd(dsd)
	{
	}

	void set_initial_inv_level(int initial_inv_level)
	{
		this->initial_inv_level = initial_inv_level;
	}
	void set_num_months(int num_months)
	{
		this->num_months = num_months;
	}
	void set_holding_cost(Ftype holding_cost)
	{
		this->holding_cost = holding_cost;
	}
	void set_shortage_cost(Ftype shortage_cost)
	{
		this->shortage_cost = shortage_cost;
	}
	void set_setup_cost(Ftype setup_cost)
	{
		this->setup_cost = setup_cost;
	}
	void set_incremental_cost(Ftype incremental_cost)
	{
		this->incremental_cost = incremental_cost;
	}
	void set_generator(URNG generator)
	{
		this->generator = generator;
	}

	Ftype get_avg_holding_cost(void)
	{
		return avg_holding_cost;
	}
	Ftype get_avg_ordering_cost(void)
	{
		return avg_ordering_cost;
	}
	Ftype get_avg_shortage_cost(void)
	{
		return avg_shortage_cost;
	}
	Ftype get_avg_total_cost(void)
	{
		return avg_holding_cost + avg_ordering_cost + avg_shortage_cost;
	}

	void simulate(int smalls, int bigs)
	{

		initialize();

		/* Run the simulation until it terminates after an end-simulation event
		   (type 3) occurs. */

		do
		{

			/* Determine the next event. */

			timing();

			/* Update time-average statistical accumulators. */

			update_time_avg_stats();

			/* Invoke the appropriate event function. */

			switch (next_event_type)
			{
			case NONE:
				cerr << "ERROR: next_event_type is NONE.";
				assert(0);
				break;
			case ORDER_ARRIVAL:
				order_arrival();
				break;
			case DEMAND:
				demand();
				break;
			case EVALUATE:
				evaluate(smalls, bigs);
				break;
			case END_SIMULATION:
				report();
				break;
			}

			/* If the event just executed was not the end-simulation event (type 3),
			   continue simulating.  Otherwise, end the simulation for the current
			   (s,S) pair and go on to the next pair (if any). */

		} while (next_event_type != END_SIMULATION);
	}
};

#endif // SIMULATOR_H