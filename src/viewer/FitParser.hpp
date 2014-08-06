#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <vector>

struct Fits {
	std::vector<FitData> fits;
	void load(const std::string &filename);
};

void Fits::load(const std::string &filename)
{
    using boost::property_tree::ptree;
    ptree pt;
    read_json(filename, pt);
    BOOST_FOREACH(ptree::value_type& pfit, pt.get_child("fits")) {
    	FitData fit;
    	fit.status = pfit.second.get("status", -1.0f);
    	int i = 0;
    	BOOST_FOREACH(ptree::value_type& p, pfit.second.get_child("params")) {
    		fit.param[i] = p.second.get("",0.0f);
    		i++;
    	}
    	fits.push_back(fit);
    }
}

