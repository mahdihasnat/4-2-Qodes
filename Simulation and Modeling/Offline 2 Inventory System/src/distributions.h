#ifndef DISTRUBUTIONS_H
#define DISTRUBUTIONS_H

#include<bits/stdc++.h>
using namespace std;

template<class IntType = int, class FloatType = double>
class my_discrete_distribution
{
	vector<FloatType> q_probs;
	public:
	my_discrete_distribution(vector<FloatType> const & q_probs_): q_probs(q_probs_) {
		// cumulative probabilities must be non-decreasing
		assert(is_sorted(q_probs.begin(), q_probs.end()));

	}
	template<class URNG> IntType operator()(URNG& g)
	{
		FloatType r = FloatType(g()-g.min())/(g.max()-g.min());
		// DBG(g.min());
		// DBG(g.max());
		DBG(r);
		int x = lower_bound(q_probs.begin(), q_probs.end(), r) - q_probs.begin();
		assert(x < (int)q_probs.size());

		return x;
	}
};


template<class FloatType = double>
class my_uniform_real_distribution
{
	FloatType a,b;
	public:
	my_uniform_real_distribution(FloatType a, FloatType b): a(a), b(b) {
	}
	template<class URNG> FloatType operator()(URNG& g)
	{
		FloatType r = FloatType(g()-g.min())*(b-a)/(g.max()-g.min()) + a;
		return r;
	}
};



#endif