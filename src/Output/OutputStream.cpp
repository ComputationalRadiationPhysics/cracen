#include "OutputStream.hpp"
#include <boost/property_tree/json_parser.hpp>

void OutputStream::run() {
	using boost::property_tree::ptree;
	ptree pt;
	ptree array;
	while(!oBuffer.isFinished()) {
		oBuffer.popTry([&array](Output& o){
			for(Output::value_type& i : o) i.save(array);
		});
	}
	pt.add_child("fits", array);
	write_json(file, pt);
}

OutputStream::OutputStream(const std::string& file, int producer) :
    oBuffer(CHUNK_BUFFER_COUNT, producer),
    file(file),
	done(false)
{
	pthread_create(&thread_tid, NULL, run_helper, this);
}

Ringbuffer<Output>* OutputStream::getBuffer() {
	return &oBuffer;
}

void OutputStream::join() {
	pthread_join(thread_tid, NULL);
}
