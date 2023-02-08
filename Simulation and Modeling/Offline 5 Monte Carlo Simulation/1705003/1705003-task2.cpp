#include<bits/stdc++.h>
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

#include "distributions.h"



using Ftype = double;
using URNG = mt19937;
auto seed = chrono::steady_clock::now().time_since_epoch().count();
URNG generator(seed);


vector<int> sample(int n,auto &d)
// generation [0..n-1]
{
	vector<int> ret(n,0);
	ret[0]=1;
	for(int i=0;i+1<n;i++)
	{
		int x = ret[i];
		if(x==0) break;
		while(x--)
		{
			ret[i+1]+=d(generator);
		}
	}
	return ret;
}


vector<Ftype> family_prob(int n)
{
	vector<Ftype> ret(n+1,0);
	if(n>=1)
	{
		ret[1] = 0.2126;
	}
	for(int i=2;i<=n;i++)
	{
		ret[i]=ret[i-1]*0.5893;
	}
	{
		Ftype t = 0;
		for(int i=1;i<=n;i++)
			t+=ret[i];
		ret[0]=1.0-t;
	}
	return ret;
}

int main()
{
	int n=4;
	vector<Ftype> prob(n+1,0);
	// cout<<"Enter the probability:[i=1 to "<<n<<"](except 0)]\n";
	// for(int i=1;i<=n;i++)
	// {
	// 	cout<<"p["<<i<<"]:";
	// 	// cin>>prob[i];
	// 	prob[i]=1.0/5;
	// }
	// cout<<"\n";
	// calc qsum
	{
		Ftype t = 0;
		for(int i=1;i<=n;i++)
			t+=prob[i];
		assert(t<1.0);
		prob[0]=1-t;
	}
	prob = family_prob(n);
	{
		for(int i=1;i<=n;i++)
			prob[i]+=prob[i-1];
	}
	auto d = my_discrete_distribution<int,Ftype>(prob);
	int x = d(generator);
	int iter = 1000000;
	int mx = 10;
	vector< vector<Ftype> > ans(n+1,vector<Ftype>(mx+1,0));
	for(int i=0;i<iter;i++)
	{
		auto s = sample(n+1,d);
		for(int j=0;j<=n;j++)
		{
			if(mx>=s[j])
				ans[j][s[j]]++;
		}
	}
	for(auto &i: ans)
		for(auto &j: i)
			j/=iter;
	cout<<"Total Iteration:"<<iter<<"\n";
	for(int i=0;i<=n;i++)
	{
		cout<<"---------------------------\n";
		cout<<"Generation "<<i<<" \n";
		for(int j=0;j<=mx;j++)
		{
			cout<<"Porobability of "<<setw(2)<<j<<" neutron in "<<i<<" 'th generation: ";
			cout<<fixed<<setprecision(10)<<ans[i][j]<<"\n";
		}
	}
}